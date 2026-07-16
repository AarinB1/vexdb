// Smoke-test the browser demo stack in Node — the wasm pkg, the movies.vex
// snapshot, and the viz math — without opening a browser:
//
//   node scripts/smoke-demo.mjs
//
// Run it after rebuilding vex-wasm or regenerating the demo data; it loads
// the exact artifacts the landing page fetches and asserts the demo's
// invariants (trace == search, exact-scan oracle, projection sanity).
//
// TODO: this reads the snapshot from disk and does NOT exercise the browser's
// fetch/decode path (fetchWithProgress, Content-Encoding: gzip on Pages), which
// is why a boot-time buffer overflow on gzipped movies.vex passed this check.

import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";

const url = (p) => new URL(p, import.meta.url);
const { default: init, VexIndex } = await import(url("../docs/demo/pkg/vex_wasm.js"));
const viz = await import(url("../docs/demo/viz.js"));

await init({ module_or_path: await readFile(url("../docs/demo/pkg/vex_wasm_bg.wasm")) });
const snapshot = new Uint8Array(await readFile(url("../docs/demo/movies.vex")));

let t = performance.now();
const index = VexIndex.fromSnapshot(snapshot);
console.log(`snapshot: ${index.len()} vectors × dim ${index.dim()}, parsed in ${(performance.now() - t).toFixed(1)} ms`);
assert.ok(index.len() > 0 && index.dim() > 0);

// --- searchTrace must agree with search, and the trace must be coherent ---
const query = index.vectorOf(7n);
const traced = index.searchTrace(query, 9, 80, null);
const plain = index.search(query, 9, 80, null);
assert.deepEqual(
  traced.hits.map((h) => h.id),
  plain.map((h) => h.id),
  "searchTrace changed the results",
);
const { trace } = traced;
assert.ok(trace.visited > 0 && trace.visited <= index.len());
assert.ok(trace.distance_evals >= trace.visited);
assert.equal(trace.visited, trace.beam.length + 1);
assert.ok(trace.entry_id !== null && trace.max_layer >= 1);
console.log(`trace: entered layer ${trace.max_layer}, ${trace.descent.length} descent hops, visited ${trace.visited} (${((100 * trace.visited) / index.len()).toFixed(1)}%), ${trace.distance_evals} distance evals`);

// --- filtered traces must mark routed nodes and respect the filter ---
const filter = JSON.stringify({ contains: { key: "genres", value: "Horror" } });
const filtered = index.searchTrace(query, 9, 80, filter);
assert.ok(filtered.hits.every((h) => h.payload.genres.includes("Horror")));
assert.ok(filtered.trace.beam.some((e) => e.status === "routed"), "selective filter routed no nodes");

// --- the exact scan is a perfect-recall oracle ---
t = performance.now();
const exact = index.searchExact(query, 10, null);
const exactMs = performance.now() - t;
const recallWide = viz.recallOf(exact, index.search(query, 10, 400, null));
assert.equal(recallWide, 1, "ef=400 should reach recall 1.0 on 5k vectors");
console.log(`exact scan: ${exactMs.toFixed(2)} ms; recall@10 at ef=80: ${viz.recallOf(exact, index.search(query, 10, 80, null)).toFixed(2)}`);

// --- bulk export feeds the projection ---
const ids = index.exportIds();
const vectors = index.exportVectors();
const n = ids.length;
const dim = index.dim();
assert.equal(vectors.length, n * dim, "export arrays disagree");
assert.deepEqual([...vectors.slice(3 * dim, 3 * dim + 4)], [...index.vectorOf(ids[3]).slice(0, 4)]);

// --- PCA projection: finite, spread out, and stable for a fixed input ---
t = performance.now();
const basis = viz.fitPca2(vectors, n, dim);
const { xy, bounds } = viz.projectAll(vectors, n, dim, basis);
console.log(`projection: fitted + projected ${n} points in ${(performance.now() - t).toFixed(0)} ms`);
assert.equal(xy.length, n * 2);
let spreadX = 0;
let meanX = 0;
for (let i = 0; i < n; i++) meanX += xy[i * 2];
meanX /= n;
for (let i = 0; i < n; i++) spreadX += (xy[i * 2] - meanX) ** 2;
assert.ok(Math.sqrt(spreadX / n) > 0.05, "projection collapsed to a point");
for (const v of xy.slice(0, 64)) assert.ok(v >= 0 && v <= 1 && Number.isFinite(v));
const again = viz.fitPca2(vectors, n, dim);
assert.deepEqual([...again.c1.slice(0, 4)], [...basis.c1.slice(0, 4)], "PCA is not deterministic");
const [qx, qy] = viz.projectPoint(query, basis, bounds);
assert.ok(qx >= 0 && qx <= 1 && qy >= 0 && qy <= 1);

// --- timeline construction for the animation ---
const tl = viz.buildTimeline(trace, traced.hits);
assert.equal(tl.events.length, trace.descent.length + trace.beam.length);
assert.equal(tl.counts.admitted + tl.counts.routed + tl.counts.rejected, trace.beam.length);
for (const h of traced.hits) {
  assert.equal(tl.nodeState.get(h.id).kind, "result", `hit ${h.id} not marked as result`);
}
assert.ok(tl.beamStartId !== null && tl.beamStartId !== undefined);
console.log(`timeline: ${tl.events.length} events (${tl.counts.admitted} admitted / ${tl.counts.routed} routed / ${tl.counts.rejected} rejected)`);

console.log("\nsmoke-demo: all checks passed ✔");

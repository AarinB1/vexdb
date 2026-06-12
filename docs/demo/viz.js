// Pure math + timeline helpers for the landing-page search X-ray.
//
// No DOM in this module: the page imports it for the canvas visualization,
// and scripts/smoke-demo.mjs imports it in Node to test the same code that
// ships. Everything is deterministic for a fixed input.

/** Deterministic PRNG (mulberry32) so the PCA basis is stable across loads. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalize(v) {
  let norm = 0;
  for (let d = 0; d < v.length; d++) norm += v[d] * v[d];
  norm = Math.sqrt(norm) || 1;
  for (let d = 0; d < v.length; d++) v[d] /= norm;
  return v;
}

function dotCentered(vectors, row, dim, mean, v) {
  let acc = 0;
  const base = row * dim;
  for (let d = 0; d < dim; d++) acc += (vectors[base + d] - mean[d]) * v[d];
  return acc;
}

/**
 * Fit the top-2 principal components of `vectors` (flat, dim-strided) by
 * power iteration with deflation, on a stride sample of up to `sampleCap`
 * rows. Returns `{mean, c1, c2}` as Float64Arrays of length `dim`.
 *
 * O(sample × dim × iters) — about 50M flops for 2,500 × 384 × 28 × 2,
 * comfortably a one-time cost in a browser tab.
 */
export function fitPca2(vectors, n, dim, { sampleCap = 2500, iters = 28 } = {}) {
  if (n === 0 || dim === 0) throw new Error("fitPca2: empty input");
  const mean = new Float64Array(dim);
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    for (let d = 0; d < dim; d++) mean[d] += vectors[base + d];
  }
  for (let d = 0; d < dim; d++) mean[d] /= n;

  const step = Math.max(1, Math.floor(n / sampleCap));
  const sample = [];
  for (let i = 0; i < n; i += step) sample.push(i);

  const rand = mulberry32(0x5eed);
  const component = (deflate) => {
    let v = new Float64Array(dim);
    for (let d = 0; d < dim; d++) v[d] = rand() - 0.5;
    normalize(v);
    const acc = new Float64Array(dim);
    for (let it = 0; it < iters; it++) {
      acc.fill(0);
      for (const row of sample) {
        const proj = dotCentered(vectors, row, dim, mean, v);
        const base = row * dim;
        for (let d = 0; d < dim; d++) acc[d] += proj * (vectors[base + d] - mean[d]);
      }
      if (deflate) {
        let along = 0;
        for (let d = 0; d < dim; d++) along += acc[d] * deflate[d];
        for (let d = 0; d < dim; d++) acc[d] -= along * deflate[d];
      }
      v.set(acc);
      normalize(v);
    }
    return v;
  };

  const c1 = component(null);
  const c2 = component(c1);
  return { mean, c1, c2 };
}

function percentile(sorted, p) {
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[idx];
}

/**
 * Project every row onto the PCA basis and normalize into [0,1]², clamping
 * to the 1st–99th percentile so outliers can't squash the cloud. Returns
 * `{xy, bounds}`: `xy` is a Float32Array of n (x,y) pairs; `bounds` feeds
 * `projectPoint` so queries land in the same coordinate space.
 */
export function projectAll(vectors, n, dim, basis) {
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = dotCentered(vectors, i, dim, basis.mean, basis.c1);
    ys[i] = dotCentered(vectors, i, dim, basis.mean, basis.c2);
  }
  const sx = Float64Array.from(xs).sort();
  const sy = Float64Array.from(ys).sort();
  const bounds = {
    x0: percentile(sx, 0.01),
    x1: percentile(sx, 0.99),
    y0: percentile(sy, 0.01),
    y1: percentile(sy, 0.99),
  };
  if (bounds.x1 - bounds.x0 < 1e-12) bounds.x1 = bounds.x0 + 1;
  if (bounds.y1 - bounds.y0 < 1e-12) bounds.y1 = bounds.y0 + 1;
  const xy = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    xy[i * 2] = clamp01((xs[i] - bounds.x0) / (bounds.x1 - bounds.x0));
    xy[i * 2 + 1] = clamp01((ys[i] - bounds.y0) / (bounds.y1 - bounds.y0));
  }
  return { xy, bounds };
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** Project a single vector (e.g. the live query) into the same [0,1]² space. */
export function projectPoint(vec, basis, bounds) {
  let x = 0;
  let y = 0;
  for (let d = 0; d < vec.length; d++) {
    const c = vec[d] - basis.mean[d];
    x += c * basis.c1[d];
    y += c * basis.c2[d];
  }
  return [
    clamp01((x - bounds.x0) / (bounds.x1 - bounds.x0)),
    clamp01((y - bounds.y0) / (bounds.y1 - bounds.y0)),
  ];
}

/**
 * Turn a `searchTrace` result into an animation timeline:
 * - `events`: descent hops then beam evaluations, in traversal order;
 * - `nodeState`: final per-node verdict (`result` with rank > `admitted` >
 *   `routed` > `rejected`), for the settled frame;
 * - `beamStartId`: where the layer-0 beam began (the descent's last stop);
 * - `counts`: beam outcome totals for the explain readout.
 */
export function buildTimeline(trace, hits) {
  const events = [];
  for (const h of trace.descent) {
    events.push({ t: "descent", layer: h.layer, from: h.from, to: h.to });
  }
  for (const e of trace.beam) {
    events.push({ t: "beam", from: e.from, to: e.to, status: e.status });
  }

  const beamStartId = trace.descent.length
    ? trace.descent[trace.descent.length - 1].to
    : trace.entry_id;

  const rankOf = { rejected: 0, routed: 1, admitted: 2, descent: 3, result: 4 };
  const nodeState = new Map();
  const upgrade = (id, kind, rank) => {
    if (id === null || id === undefined) return;
    const cur = nodeState.get(id);
    if (!cur || rankOf[kind] > rankOf[cur.kind]) nodeState.set(id, { kind, rank });
  };
  for (const e of trace.beam) upgrade(e.to, e.status);
  upgrade(trace.entry_id, "descent");
  for (const h of trace.descent) upgrade(h.to, "descent");
  hits.forEach((h, i) => upgrade(h.id, "result", i));

  const counts = { admitted: 0, routed: 0, rejected: 0 };
  for (const e of trace.beam) counts[e.status]++;

  return { events, nodeState, beamStartId, counts };
}

/** recall@k of `hits` against the exact `truth` (both `{id}` arrays). */
export function recallOf(truth, hits) {
  if (!truth.length) return 1;
  const ids = new Set(truth.map((h) => h.id));
  return hits.filter((h) => ids.has(h.id)).length / truth.length;
}

/** Median of a non-empty numeric array (not in-place). */
export function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

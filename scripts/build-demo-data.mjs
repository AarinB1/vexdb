#!/usr/bin/env node
// Builds the browser demo dataset: fetches ~10k movies (title + plot
// overview) from the Hugging Face datasets server, embeds them with
// all-MiniLM-L6-v2 (384-dim, the same model the landing page uses for
// queries via transformers.js), and bakes a .vex HNSW snapshot with
// `vex build`.
//
// Usage:  node scripts/build-demo-data.mjs [--n 5000]
// Output: docs/demo/movies.vex, docs/demo/meta.json
//
// Run from the repo root. Requires `npm install` in scripts/ and a Rust
// toolchain for the `vex build` step.

import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, statSync } from "node:fs";
import { pipeline } from "@huggingface/transformers";

const N = (() => {
  const i = process.argv.indexOf("--n");
  return i >= 0 ? parseInt(process.argv[i + 1], 10) : 5000;
})();

const DATASET = "Pablinho/movies-dataset";
const MODEL = "Xenova/all-MiniLM-L6-v2";
const DIM = 384;
const PAGE = 100;

async function fetchRows() {
  const rows = [];
  for (let offset = 0; ; offset += PAGE) {
    const url =
      `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(DATASET)}` +
      `&config=default&split=train&offset=${offset}&length=${PAGE}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`datasets-server ${res.status} at offset ${offset}`);
    const page = await res.json();
    rows.push(...page.rows.map((r) => r.row));
    process.stdout.write(`\rfetched ${rows.length} rows`);
    if (page.rows.length < PAGE) break;
  }
  console.log();
  return rows;
}

function clean(rows) {
  const seen = new Set();
  const out = [];
  for (const r of rows) {
    const title = (r.Title || "").trim();
    const overview = (r.Overview || "").trim();
    const year = parseInt((r.Release_Date || "").slice(0, 4), 10);
    if (!title || overview.length < 40 || !Number.isFinite(year)) continue;
    if (r.Original_Language !== "en") continue; // keep MiniLM in-domain
    const key = `${title}|${year}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      title,
      overview,
      year,
      genres: (r.Genre || "").split(",").map((g) => g.trim()).filter(Boolean),
      rating: parseFloat(r.Vote_Average) || null,
      popularity: r.Popularity || 0,
    });
  }
  out.sort((a, b) => b.popularity - a.popularity);
  return out.slice(0, N);
}

async function embedAll(movies) {
  console.log(`loading ${MODEL}…`);
  const embed = await pipeline("feature-extraction", MODEL, { dtype: "q8" });
  const vectors = [];
  const BATCH = 64;
  const t0 = performance.now();
  for (let i = 0; i < movies.length; i += BATCH) {
    const batch = movies.slice(i, i + BATCH);
    const texts = batch.map((m) => `${m.title}. ${m.overview}`);
    const out = await embed(texts, { pooling: "mean", normalize: true });
    for (let j = 0; j < batch.length; j++) {
      vectors.push(Array.from(out.data.slice(j * DIM, (j + 1) * DIM)));
    }
    process.stdout.write(`\rembedded ${vectors.length}/${movies.length}`);
  }
  console.log(`  (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  return vectors;
}

const rows = await fetchRows();
const movies = clean(rows);
console.log(`kept ${movies.length} movies after cleaning`);

const vectors = await embedAll(movies);

mkdirSync("docs/demo", { recursive: true });
const jsonl = movies
  .map((m, i) =>
    JSON.stringify({
      id: i,
      vector: vectors[i].map((x) => +x.toFixed(6)),
      payload: {
        title: m.title,
        overview: m.overview.length > 280 ? m.overview.slice(0, 277) + "…" : m.overview,
        year: m.year,
        genres: m.genres,
        rating: m.rating,
      },
    }),
  )
  .join("\n");
writeFileSync("/tmp/movies.jsonl", jsonl);

console.log("building HNSW snapshot with vex build…");
execFileSync(
  "cargo",
  [
    "run", "--release", "-p", "vex-cli", "--",
    "build",
    "--input", "/tmp/movies.jsonl",
    "--output", "docs/demo/movies.vex",
    "--dim", String(DIM),
    "--metric", "cosine",
    "--index", "hnsw",
    "--hnsw-m", "16",
    "--ef-construction", "200",
  ],
  { stdio: "inherit" },
);

// Genre list for the demo's filter dropdown, most frequent first.
const genreCounts = new Map();
for (const m of movies)
  for (const g of m.genres) genreCounts.set(g, (genreCounts.get(g) || 0) + 1);
const genres = [...genreCounts.entries()].sort((a, b) => b[1] - a[1]).map(([g]) => g);

writeFileSync(
  "docs/demo/meta.json",
  JSON.stringify(
    {
      dataset: DATASET,
      model: MODEL,
      metric: "cosine",
      dim: DIM,
      count: movies.length,
      genres,
      yearRange: [
        Math.min(...movies.map((m) => m.year)),
        Math.max(...movies.map((m) => m.year)),
      ],
      built: new Date().toISOString().slice(0, 10),
    },
    null,
    2,
  ),
);

const mb = (statSync("docs/demo/movies.vex").size / 1024 / 1024).toFixed(1);
console.log(`done: docs/demo/movies.vex (${mb} MB), docs/demo/meta.json`);

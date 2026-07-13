# vexdb

**[▶ Live demo](https://aarinb1.github.io/vexdb/)** — 5,000 movie embeddings, HNSW search, and the traversal X-ray running in your browser.

A vector database written in Rust, from scratch — no faiss bindings, no
black boxes. A hand-rolled HNSW index, binary snapshot persistence, payload
filtering during graph traversal, SIMD distance kernels, and a Qdrant-style
HTTP API, built in phases and benchmarked honestly against faiss at the end
([BENCHMARKS.md](BENCHMARKS.md) — TL;DR: recall curves at parity with
`IndexHNSWFlat`, QPS within ~2.5×, and the gap is entirely in the distance
kernels, not the graph).

## Workspace layout

| crate        | what it is |
|--------------|------------|
| `vex-core`   | The engine: `FlatIndex` (exact), `HnswIndex` (approximate), distance metrics with AVX2 kernels, payload filters, `.vex` snapshot format. Synchronous, embeddable; `default-features = false` drops rayon for a lean build. |
| `vex-cli`    | `vex` binary: `ingest`, `build` (JSONL → `.vex`), `query` (JSONL or snapshot), `bench` (recall/QPS harness). |
| `vex-server` | `vex-server` binary: HTTP API + demo console over a directory of snapshots. All async deps (tokio/axum/tower) are isolated here. |
| `vex-wasm`   | wasm-bindgen bindings: the whole engine compiled to a 289 KB WebAssembly module. Powers the landing-page demo — snapshot loading, HNSW search, filters, traversal traces, and the exact-scan recall oracle, all running in the browser. |

## The browser demo (no server)

The landing page ([`docs/index.html`](https://aarinb1.github.io/vexdb/), live on GitHub Pages) embeds a live demo:
vex-core compiled to WebAssembly, searching **5,000 movie-plot embeddings**
(all-MiniLM-L6-v2, 384-dim, cosine) inside the visitor's tab. "Surprise me"
and "more like this" use stored vectors and need no model; typed queries
lazy-load the same MiniLM model via transformers.js so the query embeds
locally too. Measured in-browser: snapshot parse ~50 ms, HNSW search ~2–4 ms,
query embedding ~200 ms. There is no backend — search, filters, and the
snapshot parser are the same Rust code paths as the native build.

Two things on the page go beyond a typical demo:

- **The search X-ray.** Every search calls `searchTrace` (the engine's
  `search_traced` through wasm) and animates the *actual* traversal over a
  PCA projection of all 5,000 embeddings — fitted by power iteration in the
  tab (~0.5 s, `docs/demo/viz.js`), no precomputed layout. You watch the
  greedy descent through the upper layers, the ef-wide beam spreading over
  layer 0, and — with a genre filter on — hundreds of orange "routed" nodes:
  filtered-out vectors steering the beam without taking result slots.
  Dragging the `ef` slider re-runs the search and visibly widens the beam,
  and an explain strip reports visited nodes, distance evals, and hop counts
  from the trace.
- **The recall lab.** One click brute-forces the exact top-10
  (`searchExact`), sweeps `ef`, and charts measured recall@10 against
  measured latency — a small ann-benchmarks-style curve produced live on the
  visitor's hardware, not copied from a README.

Run it locally:

```sh
python3 -m http.server 3000 --directory docs   # → http://localhost:3000
```

Rebuild the wasm module after core changes (needs `wasm32-unknown-unknown`
target + [wasm-pack](https://rustwasm.github.io/wasm-pack/)):

```sh
cd crates/vex-wasm && wasm-pack build --release --target web --out-dir ../../docs/demo/pkg
```

Regenerate the dataset (fetches the movie corpus, embeds with MiniLM in
Node, bakes the snapshot with `vex build`):

```sh
cd scripts && npm install && cd ..
node scripts/build-demo-data.mjs --n 5000
```

Sanity-check the shipped demo artifacts (wasm pkg + snapshot + viz math)
without opening a browser:

```sh
node scripts/smoke-demo.mjs
```

## Quick start: the server

```sh
cargo run --release -p vex-server -- --addr 127.0.0.1:8080 --data-dir ./data
```

Open `http://127.0.0.1:8080/` for the built-in console (create a demo
collection, run filtered searches from the browser). Or drive it directly:

```sh
# Create a collection
curl -X POST localhost:8080/collections/products \
  -H 'content-type: application/json' \
  -d '{"dim": 8, "metric": "cosine", "index": "hnsw", "m": 16, "ef_construction": 200}'

# Upsert points (id exists → replaced)
curl -X POST localhost:8080/collections/products/points \
  -H 'content-type: application/json' \
  -d '{"points": [{"id": 1, "vector": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
                   "payload": {"category": "shoes", "price": 49.0}}]}'

# Filtered search
curl -X POST localhost:8080/collections/products/search \
  -H 'content-type: application/json' \
  -d '{"query": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], "k": 10, "ef": 64,
       "filter": {"and": [{"eq": {"key": "category", "value": "shoes"}},
                          {"range": {"key": "price", "lte": 99.0}}]}}'
```

### API

```
GET    /health                              liveness
GET    /                                    demo console
GET    /collections                         list collections
POST   /collections/{name}                  create (dim, metric, index, m, ef_construction, ef_search)
GET    /collections/{name}                  stats: count, dim, metric, index type
DELETE /collections/{name}                  drop (removes the snapshot)
POST   /collections/{name}/points           upsert batch: {"points": [{id, vector, payload?}]}
DELETE /collections/{name}/points/{id}      delete one vector
POST   /collections/{name}/search           {"query", "k", "ef"?, "filter"?, "with_payload"?}
POST   /collections/{name}/snapshot         force a flush to disk
```

Errors map to honest statuses: dimension mismatch / bad input → 400,
unknown collection → 404, duplicate create → 409, writes in read-only mode
→ 403. Input limits are enforced (64 MiB body cap, k ≤ 1024, 10k points per
batch).

### Persistence & lifecycle

Every collection maps to `<data_dir>/<name>.vex`. Snapshots found at
startup are loaded as collections; writes mark a collection dirty; a
background task (default every 30s), an explicit `POST .../snapshot`, and
graceful shutdown (SIGTERM/ctrl-C) flush dirty collections back to disk
with an atomic temp-file-and-rename. `--read-only` serves search over
prebuilt snapshots with every write endpoint disabled — the dramatically
simpler deployment when you don't need online writes.

Concurrency model: collections sit behind `RwLock` — searches run
concurrently, writes take the exclusive lock. `vex_core::search_batch`
(rayon) parallelizes query batches over a frozen index.

### Docker

```sh
docker build -t vexdb .
docker run -p 8080:8080 -v vexdata:/data vexdb
```

## The CLI

```sh
# Build a snapshot from JSONL ({"id": u64, "vector": [f32...], "payload": {...}?})
cargo run --release -p vex-cli -- build \
    --input data.jsonl --dim 128 --metric cosine --index hnsw --output index.vex

# Query the snapshot (no rebuild)
cargo run --release -p vex-cli -- query \
    --snapshot index.vex --query 0.1,0.2,0.3 --k 10 --ef 128

# Recall/QPS harness: flat ground truth vs HNSW across ef_search values
cargo run --release -p vex-cli -- bench --n 100000 --dim 32 --queries 200 --k 10
```

## The engine (vex-core)

- **`Index` trait** with one contract for every implementation: empty index
  → zero results, `k > len()` clamps, `remove` is idempotent (`Ok(false)`
  for missing ids), results sorted ascending by distance for every metric.
- **`HnswIndex`** from the Malkov & Yashunin paper: exponential level
  sampling, greedy upper-layer descent, ef-bounded beam search, Algorithm 4
  neighbor selection with `keepPrunedConnections`. Arena storage (`u32`
  neighbor indices into a flat node array). Deletes are tombstones: removed
  nodes still route the beam but never surface.
- **Filtered search happens during traversal**, not as a post-cull:
  non-matching nodes route the beam but don't occupy result slots, so a 1%
  filter widens the search instead of starving it (the Qdrant approach).
- **Tracing and ground truth built in**: `search_traced` returns the same
  results as `search` plus a full record of the traversal (descent hops,
  every beam evaluation with its admitted/routed/rejected verdict, work
  counters) — the plain search path threads a no-op sink through the same
  code, so tracing costs nothing unless asked for. `search_exact` is the
  brute-force oracle for measuring the index's own recall. Both power the
  landing-page X-ray.
- **Distance kernels**: cosine / L2 / dot, all normalized to
  smaller-is-closer. Runtime-detected AVX2+FMA paths with
  auto-vectorizable scalar fallbacks; the scalar versions are the oracle in
  property tests (SIMD reorders FP summation, so equality is approximate).
- **`.vex` snapshots**: hand-rolled little-endian format — magic bytes,
  version field, then the arena dumped as flat arrays. Every length and
  index read from disk is validated; corrupt or truncated files fail with a
  typed error, never a panic. Round-trips are search-identical, including
  tombstones and the RNG state (construction continues deterministically
  after a load).

## Tests

```sh
cargo test --workspace
```

67 tests (the proptest ones each run 256 generated cases): unit + proptest
invariants on both indexes, recall@10 against flat ground truth,
SIMD-vs-scalar oracle properties, snapshot round-trip and corruption tests,
trace-vs-plain-search equivalence and trace invariants, exact-scan-vs-flat
agreement, CLI end-to-end runs, and HTTP API integration tests (CRUD,
filters, error statuses, read-only mode, persistence across restart).
`scripts/smoke-demo.mjs` additionally exercises the built wasm pkg, the
demo snapshot, and the projection math in Node.

## Lint

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Phases

The project was built in checkpointed phases — each one shipped with tests
and a "done when":

1. **Foundation** — core types, metrics, `FlatIndex`, CLI, CI ✅
2. **HNSW** — the graph index + recall harness ✅
3. **Persistence** — the `.vex` snapshot format ✅
4. **Concurrent reads** — `Arc`/`RwLock`, rayon `search_batch` ✅
5. **Server v1** — read-only HTTP API over snapshots ✅
6. **Writes over HTTP** — upsert/delete, flush lifecycle, input limits ✅
7. **Metadata & filtering** — payloads + traversal-time filters ✅
8. **SIMD** — AVX2 kernels, scalar oracle ✅
9. **The faiss benchmark** — [BENCHMARKS.md](BENCHMARKS.md) ✅
10. **The browser build** — vex-wasm + the in-tab semantic search demo ✅
11. **The X-ray** — zero-cost trace instrumentation (`search_traced`), the
    exact-scan recall oracle (`search_exact`), and a landing page that
    animates the real traversal and measures live recall curves in the
    visitor's tab ✅

Future work, in rough order of payoff: batched/binary query protocol (the
HTTP overhead measurement makes the case), mmap-backed snapshot loading,
software prefetch in the beam search, NEON kernels, per-node locking for
concurrent writes.

Requires stable Rust 1.82+.

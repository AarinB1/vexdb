# vexdb

A vector database written in Rust, from scratch, as a learning project. The
end goal is a hand-rolled HNSW index benchmarked against faiss; this repository
is being built out in phases.

## Phase 2 — HNSW (this checkpoint)

What's new:

- **`HnswIndex`**: Hierarchical Navigable Small World approximate
  nearest-neighbor index, implemented from the Malkov & Yashunin paper —
  layered graph with exponentially-decaying level sampling, greedy descent
  through the upper layers, ef-bounded beam search at layer 0, and the
  Algorithm 4 neighbor-selection heuristic (with `keepPrunedConnections`).
  Tunables exposed via `HnswConfig`: `m`, `ef_construction`, `ef_search`,
  plus a `seed` for deterministic construction. Storage is arena-style flat
  arrays (`u32` neighbor indices), which sets up the Phase 3 on-disk format.
- **Deletion via tombstones**: `remove` marks nodes deleted; they keep
  routing queries but never appear in results. True graph repair is
  deliberately out of scope (hnswlib and faiss avoid it too). `len()`
  reports live vectors only.
- **A unified `Index` contract** (see the trait docs): searching an empty
  index returns zero results (no longer an error), `k > len()` clamps,
  `remove` of a missing id is `Ok(false)`. The dead `IdNotFound` and
  `EmptyIndex` error variants are gone.
- **Recall tests**: property tests for structural invariants, plus a
  recall@10 test against `FlatIndex` ground truth (the strongest kind of
  test for a probabilistic structure).
- **Benchmarks**: criterion baselines (`cargo bench -p vex-core`) and a
  `vex bench` harness subcommand that sweeps `ef_search` and reports
  recall@k, QPS, and build time against the flat baseline.

What is **not** built yet (deliberately deferred):

- On-disk persistence: Phase 3.
- Filtering / metadata: Phase 4.
- SIMD / vectorized distance: Phase 5.
- Concurrency: Phase 6.
- The faiss comparison on SIFT1M: Phase 7.

## Phase 1 — Foundation

- A `Vector` / `VectorId` core type pair with dimension validation.
- `DistanceMetric` enum (`Cosine`, `L2`, `Dot`), naive scalar implementations.
  All metrics return *distance* (smaller = more similar) so search code is
  uniform; `Cosine` returns `1 - cos_sim`, `Dot` returns `-dot(a, b)`.
- An `Index` trait and `FlatIndex`: brute-force linear scan with a bounded
  max-heap for top-k selection (O(n log k)).
- `thiserror`-based `VexError`.
- A `vex` CLI with `ingest` and `query` subcommands over JSONL input.

## Build

```sh
cargo build --workspace
```

Requires stable Rust (1.75+).

## Run the CLI

The CLI consumes JSONL where each line is `{"id": <u64>, "vector": [<f32>, ...]}`.

```sh
# Ingest and report counts:
cargo run -p vex-cli -- ingest --input data.jsonl --dim 128

# Query (flat = exact, hnsw = approximate):
cargo run -p vex-cli -- query \
    --input data.jsonl \
    --query 0.1,0.2,0.3 \
    --k 10 \
    --metric cosine \
    --index hnsw
```

`--metric` accepts `cosine`, `l2`, or `dot`; `--index` accepts `flat` or `hnsw`.

## Benchmark harness

```sh
cargo run --release -p vex-cli -- bench --n 100000 --dim 32 --queries 200 --k 10
```

Sample output (synthetic uniform vectors, scalar distance kernels, single
thread):

```
dataset: 100000 synthetic vectors, dim 32, 200 queries, k=10, metric L2
build:   flat   17.465ms   hnsw(M=16, efc=200)    88.610s

 ef_search     recall@k          QPS    vs flat
      flat        1.000          381       1.0x
        10        0.431         9122      23.9x
        20        0.602         5995      15.7x
        40        0.773         3613       9.5x
        80        0.916         1936       5.1x
       160        0.973         1080       2.8x
       320        0.995          574       1.5x
```

Two honest caveats on these numbers. First, *uniform* random vectors at high
dimension are close to the worst case for any graph index — distances
concentrate and there is no low-dimensional structure to navigate (at dim 128
recall drops sharply; at dim 16 it is ~1.0 almost immediately). Real
embedding datasets (SIFT, GloVe) have far lower intrinsic dimensionality,
which is what the Phase 7 faiss comparison will use. Second, the speedup
column understates HNSW: the scalar distance kernels (no SIMD until Phase 5)
make the brute-force baseline artificially cheap to beat per-distance, and
the gap widens with n.

Criterion micro-benchmarks (insert throughput, query latency for both
indexes at several scales):

```sh
cargo bench -p vex-core
```

## Tests

```sh
cargo test --workspace
```

The suite includes property tests (via `proptest`) over both indexes and a
recall@10 test for `HnswIndex` against `FlatIndex` ground truth.

## Lint

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

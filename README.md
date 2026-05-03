# vexdb

A vector database written in Rust, from scratch, as a learning project. The
end goal is a hand-rolled HNSW index benchmarked against faiss; this repository
is being built out in phases.

## Phase 1 — Foundation (this checkpoint)

What works:

- A `Vector` / `VectorId` core type pair with dimension validation.
- `DistanceMetric` enum (`Cosine`, `L2`, `Dot`), naive scalar implementations.
  All metrics return *distance* (smaller = more similar) so search code is
  uniform; `Cosine` returns `1 - cos_sim`, `Dot` returns `-dot(a, b)`.
- An `Index` trait and one implementation, `FlatIndex`: brute-force linear
  scan with a bounded max-heap for top-k selection (O(n log k)).
- `thiserror`-based `VexError`.
- A `vex` CLI with `ingest` and `query` subcommands over JSONL input.

What is **not** built yet (deliberately deferred):

- HNSW (and any approximate index): Phase 2.
- On-disk persistence: Phase 3.
- Filtering / metadata: later.
- SIMD / vectorized distance: Phase 5.
- Concurrency / multi-tenant indexing.

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

# Query against an in-memory load of the same file:
cargo run -p vex-cli -- query \
    --input data.jsonl \
    --query 0.1,0.2,0.3 \
    --k 10 \
    --metric cosine
```

`--metric` accepts `cosine`, `l2`, or `dot`.

## Tests

```sh
cargo test --workspace
```

The test suite includes property tests (via `proptest`) over `FlatIndex`.

## Lint

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

# Benchmarks

All numbers from one run on the same machine (x86_64 with AVX2+FMA, single
thread unless stated), release builds. Synthetic uniform vectors, generated
by the same deterministic splitmix64 stream in every harness — vexdb and
faiss see bit-identical data.

Reproduce with:

```sh
cargo run --release -p vex-cli -- bench --n 100000 --dim 32 --queries 200 --k 10
python3 benches/faiss_compare.py --n 100000 --dim 32 --queries 200   # pip install faiss-cpu
```

## The headline: vexdb vs. faiss

`HnswIndex` vs. `faiss.IndexHNSWFlat`, 100k vectors × dim 32, k=10,
M=16, efConstruction=200, identical data, both single-threaded:

| ef_search | vexdb recall@10 | vexdb QPS | faiss recall@10 | faiss QPS |
|----------:|----------------:|----------:|----------------:|----------:|
| 10        | 0.431           | 19,610    | 0.459           | 41,351    |
| 20        | 0.602           | 11,753    | 0.630           | 29,235    |
| 40        | 0.773           | 6,930     | 0.801           | 17,608    |
| 80        | 0.916           | 4,022     | 0.933           | 9,628     |
| 160       | 0.973           | 1,989     | 0.986           | 5,190     |
| 320       | 0.995           | 1,033     | 0.998           | 2,624     |

Build time: vexdb 48.4s, faiss 31.1s. Exact (flat) baseline: vexdb 504 QPS,
faiss 1,058 QPS.

### Where the gap lives

Two observations, one conclusion:

1. **The recall curves are nearly identical.** At every ef, faiss's recall
   is within ~0.03 of vexdb's, which means the *graphs* are equivalent —
   level sampling, the neighbor-selection heuristic, and the beam search are
   doing the same thing the reference implementation does. The algorithm is
   right.
2. **The flat baselines differ by 2.1×** (1,058 vs 504 QPS) — pure distance
   throughput, no graph involved. The HNSW QPS gap at matched recall is
   ~2–2.5×. The entire gap is explained by per-distance cost, not traversal
   quality: faiss's hand-scheduled kernels, batched query layout, and
   prefetching beat our AVX2 loop, especially at small dims where loop
   overhead matters most.

Closing the remaining 2× would mean batched/blocked distance evaluation and
software prefetch of neighbor vectors during traversal — interesting future
work, but for a from-scratch implementation, "graph at parity, kernels 2×
behind" is exactly the gap worth understanding.

### Caveat: uniform data

Uniform random vectors at high dimension are close to the worst case for
graph indexes (distances concentrate; there is no low-dimensional manifold
to navigate). At dim 128 both vexdb's and faiss's recall drop sharply at the
same ef values — it's a property of the data, not the implementation. Real
embedding datasets (SIFT, GloVe) have much lower intrinsic dimensionality
and reach 0.95+ recall at far lower ef. The synthetic protocol here is for
*comparing implementations* on identical data, not for predicting absolute
recall on real workloads.

## SIMD speedup (Phase 8)

Same workload before/after the SIMD kernels (50k × dim 128, AVX2+FMA with
runtime detection, scalar fallback property-tested as the oracle):

|                         | scalar | SIMD  | speedup |
|-------------------------|-------:|------:|--------:|
| flat scan QPS           | 151    | 260   | 1.7×    |
| HNSW build (efc=200)    | 112s   | 45s   | 2.5×    |
| HNSW QPS @ ef=10        | 5,192  | 11,181| 2.2×    |
| HNSW QPS @ ef=320       | 314    | 562   | 1.8×    |

The win grows with dimension (more lanes amortizing loop overhead); at
dim 32 it is roughly 1.3–2×.

## HTTP layer overhead (vex-server)

10k vectors × dim 32, ef=64 (recall ≈ 0.975), same machine, JSON over
HTTP/1.1 via a Node client:

| path                        | QPS    |
|-----------------------------|-------:|
| in-process (`vex bench`)    | 15,381 |
| HTTP, 1 connection          | 1,213  |
| HTTP, 32 concurrent         | 3,236  |

When a query takes 65µs in-process, the request path — TCP, HTTP framing,
JSON parse/serialize on both ends — dominates end-to-end latency by ~10×.
Concurrency claws back throughput (readers share an `RwLock`), but the
lesson is the classic one: for cheap queries, the network layer *is* the
cost, which is why production vector DBs amortize it with batched query
endpoints and binary protocols (gRPC). Both are natural follow-ups.

## Criterion micro-benchmarks

`cargo bench -p vex-core` tracks insert throughput and query latency for
both indexes at 1k/10k/100k × dim 128, useful for catching regressions in
the kernels or the heap logic.

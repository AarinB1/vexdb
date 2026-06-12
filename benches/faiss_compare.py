#!/usr/bin/env python3
"""faiss head-to-head with the same protocol as `vex bench`.

Reproduces vexdb's splitmix64 data generation bit-for-bit (same seed, same
order: N base vectors then Q queries from one RNG stream), builds
IndexHNSWFlat with the same M / efConstruction, and sweeps efSearch.
Single-threaded so QPS is comparable with vexdb's single-threaded bench.

Usage: python3 benches/faiss_compare.py [--n 100000] [--dim 32] [--queries 200]
"""

import argparse
import time

import faiss
import numpy as np

MASK = (1 << 64) - 1


def splitmix64_vectors(n, dim, state):
    """Bit-exact clone of vexdb's Rng (crates/vex-cli/src/main.rs)."""
    total = n * dim
    out = np.empty(total, dtype=np.float32)
    for i in range(total):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        out[i] = (z >> 40) / float(1 << 24) * 2.0 - 1.0
    return out.reshape(n, dim), state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", default="10,20,40,80,160,320")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    faiss.omp_set_num_threads(1)
    print(f"faiss {faiss.__version__}, single thread")
    print(
        f"dataset: {args.n} synthetic vectors, dim {args.dim}, "
        f"{args.queries} queries, k={args.k}, metric L2 (same stream as vex bench --seed {args.seed})"
    )

    base, state = splitmix64_vectors(args.n, args.dim, args.seed)
    queries, _ = splitmix64_vectors(args.queries, args.dim, state)

    # Ground truth with exact flat search.
    flat = faiss.IndexFlatL2(args.dim)
    flat.add(base)
    t0 = time.perf_counter()
    _, truth = flat.search(queries, args.k)
    flat_qps = args.queries / (time.perf_counter() - t0)

    hnsw = faiss.IndexHNSWFlat(args.dim, args.m)
    hnsw.hnsw.efConstruction = args.ef_construction
    t0 = time.perf_counter()
    hnsw.add(base)
    build = time.perf_counter() - t0
    print(f"build:   flat-gt baseline   hnsw(M={args.m}, efc={args.ef_construction}) {build:.3f}s")
    print()
    print(f"{'ef_search':>10} {'recall@k':>12} {'QPS':>12} {'vs flat':>10}")
    print(f"{'flat':>10} {'1.000':>12} {flat_qps:>12.0f} {'1.0x':>10}")

    truth_sets = [set(row) for row in truth]
    for ef in (int(x) for x in args.ef_search.split(",")):
        hnsw.hnsw.efSearch = ef
        t0 = time.perf_counter()
        _, ids = hnsw.search(queries, args.k)
        qps = args.queries / (time.perf_counter() - t0)
        hits = sum(len(truth_sets[i] & set(row)) for i, row in enumerate(ids))
        recall = hits / (args.queries * args.k)
        print(f"{ef:>10} {recall:>12.3f} {qps:>12.0f} {qps / flat_qps:>9.1f}x")


if __name__ == "__main__":
    main()

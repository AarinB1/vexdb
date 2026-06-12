//! Criterion benchmarks: FlatIndex is the ground-truth baseline (recall 1.0
//! by definition); every HNSW number is read against it.
//!
//! Run with `cargo bench -p vex-core`. Sizes are kept moderate so a full run
//! finishes in minutes; bump `SIZES` locally for the 1M-scale numbers. For
//! the recall-vs-QPS table across ef_search values, use `vex bench` (the CLI
//! harness) — criterion measures latency only.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vex_core::{DistanceMetric, FlatIndex, HnswConfig, HnswIndex, Index, Vector, VectorId};

const DIM: usize = 128;
const SIZES: &[usize] = &[1_000, 10_000, 100_000];

/// splitmix64-based deterministic vector generator (no rand dependency).
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

fn make_vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.vector(DIM)).collect()
}

fn bench_flat_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_insert");
    for &n in SIZES {
        let vectors = make_vectors(n, 1);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &vectors, |b, vectors| {
            b.iter(|| {
                let mut idx = FlatIndex::new(DIM, DistanceMetric::L2);
                for (i, v) in vectors.iter().enumerate() {
                    idx.add(VectorId(i as u64), Vector::from_vec(v.clone()))
                        .unwrap();
                }
                idx
            })
        });
    }
    group.finish();
}

fn bench_flat_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_query_k10");
    for &n in SIZES {
        let vectors = make_vectors(n, 1);
        let mut idx = FlatIndex::new(DIM, DistanceMetric::L2);
        for (i, v) in vectors.iter().enumerate() {
            idx.add(VectorId(i as u64), Vector::from_vec(v.clone()))
                .unwrap();
        }
        let query = Rng(99).vector(DIM);
        group.bench_with_input(BenchmarkId::from_parameter(n), &idx, |b, idx| {
            b.iter(|| idx.search(&query, 10).unwrap())
        });
    }
    group.finish();
}

fn bench_hnsw_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_query_k10_ef64");
    for &n in SIZES {
        let vectors = make_vectors(n, 1);
        let mut idx = HnswIndex::new(DIM, DistanceMetric::L2, HnswConfig::default());
        for (i, v) in vectors.iter().enumerate() {
            idx.add(VectorId(i as u64), Vector::from_vec(v.clone()))
                .unwrap();
        }
        let query = Rng(99).vector(DIM);
        group.bench_with_input(BenchmarkId::from_parameter(n), &idx, |b, idx| {
            b.iter(|| idx.search(&query, 10).unwrap())
        });
    }
    group.finish();
}

fn bench_hnsw_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build");
    group.sample_size(10);
    // Build is expensive; bench the small size only. `vex bench` reports
    // build time at arbitrary scales.
    let n = 1_000;
    let vectors = make_vectors(n, 1);
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::from_parameter(n), &vectors, |b, vectors| {
        b.iter(|| {
            let mut idx = HnswIndex::new(DIM, DistanceMetric::L2, HnswConfig::default());
            for (i, v) in vectors.iter().enumerate() {
                idx.add(VectorId(i as u64), Vector::from_vec(v.clone()))
                    .unwrap();
            }
            idx
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_flat_insert,
    bench_flat_query,
    bench_hnsw_query,
    bench_hnsw_build
);
criterion_main!(benches);

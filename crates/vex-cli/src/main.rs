use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use vex_core::{
    AnyIndex, DistanceMetric, FlatIndex, HnswConfig, HnswIndex, Index, Vector, VectorId,
};

#[derive(Parser, Debug)]
#[command(name = "vex", version, about = "vex vector database CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Read JSONL of {"id": u64, "vector": [f32...]} and report count + dim.
    Ingest {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        dim: usize,
        #[arg(short = 'm', long, default_value = "l2")]
        metric: String,
    },
    /// Build an index from JSONL and save it as a .vex snapshot.
    Build {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long)]
        dim: usize,
        #[arg(short = 'm', long, default_value = "l2")]
        metric: String,
        /// Index type to build: "flat" (exact) or "hnsw" (approximate).
        #[arg(long, default_value = "hnsw")]
        index: String,
        #[arg(long, default_value_t = 16)]
        hnsw_m: usize,
        #[arg(long, default_value_t = 200)]
        ef_construction: usize,
    },
    /// Search an index built from JSONL (--input) or a .vex snapshot
    /// (--snapshot) against `--query`.
    Query {
        #[arg(short, long, conflicts_with = "snapshot")]
        input: Option<PathBuf>,
        /// Path to a .vex snapshot produced by `vex build`.
        #[arg(short, long)]
        snapshot: Option<PathBuf>,
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value_t = 5)]
        k: usize,
        /// Beam width for HNSW (ignored by flat indexes).
        #[arg(long)]
        ef: Option<usize>,
        #[arg(short = 'm', long, default_value = "l2")]
        metric: String,
        /// Index to build from --input: "flat" or "hnsw". Ignored with
        /// --snapshot (the snapshot records its own type).
        #[arg(long, default_value = "flat")]
        index: String,
    },
    /// Benchmark HNSW against FlatIndex ground truth on synthetic vectors:
    /// reports build time, then recall@k and QPS for each ef_search value.
    Bench {
        /// Number of vectors to index.
        #[arg(short, long, default_value_t = 10_000)]
        n: usize,
        /// Vector dimension.
        #[arg(short, long, default_value_t = 128)]
        dim: usize,
        /// Number of query vectors.
        #[arg(short, long, default_value_t = 100)]
        queries: usize,
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        #[arg(short = 'm', long, default_value = "l2")]
        metric: String,
        /// HNSW M (max neighbors per layer).
        #[arg(long, default_value_t = 16)]
        hnsw_m: usize,
        #[arg(long, default_value_t = 200)]
        ef_construction: usize,
        /// Comma-separated ef_search values to sweep.
        #[arg(long, default_value = "10,20,40,80,160,320")]
        ef_search: String,
        /// RNG seed for data generation.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

#[derive(Debug, Deserialize)]
struct Record {
    id: u64,
    vector: Vec<f32>,
    #[serde(default)]
    payload: Option<serde_json::Value>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Ingest { input, dim, metric } => {
            let metric: DistanceMetric = metric.parse().map_err(|e: String| anyhow!(e))?;
            let index = build_index(&input, dim, metric)?;
            println!(
                "ingested {} vectors of dim {} (metric: {:?})",
                index.len(),
                index.dim(),
                metric
            );
        }
        Command::Build {
            input,
            output,
            dim,
            metric,
            index,
            hnsw_m,
            ef_construction,
        } => {
            let metric: DistanceMetric = metric.parse().map_err(|e: String| anyhow!(e))?;
            let mut idx = new_index(&index, dim, metric, hnsw_m, ef_construction)?;
            let start = Instant::now();
            load_into(&input, &mut idx)?;
            let built = start.elapsed();
            idx.save(&output)
                .with_context(|| format!("saving snapshot to {}", output.display()))?;
            println!(
                "built {} index: {} vectors, dim {}, metric {:?} in {:.2?}; saved to {}",
                idx.type_name(),
                idx.len(),
                idx.dim(),
                metric,
                built,
                output.display()
            );
        }
        Command::Query {
            input,
            snapshot,
            query,
            k,
            ef,
            metric,
            index,
        } => {
            let q = parse_query(&query)?;
            let idx: AnyIndex = match (input, snapshot) {
                (_, Some(path)) => AnyIndex::load(&path)
                    .with_context(|| format!("loading snapshot {}", path.display()))?,
                (Some(path), None) => {
                    let metric: DistanceMetric = metric.parse().map_err(|e: String| anyhow!(e))?;
                    let mut idx = new_index(&index, q.len(), metric, 16, 200)?;
                    load_into(&path, &mut idx)?;
                    idx
                }
                (None, None) => return Err(anyhow!("provide --input or --snapshot")),
            };
            let results = idx.search_opts(&q, k, ef, None)?;
            println!("top {} results:", results.len());
            for r in results {
                println!("  id={} distance={:.6}", r.id, r.distance);
            }
        }
        Command::Bench {
            n,
            dim,
            queries,
            k,
            metric,
            hnsw_m,
            ef_construction,
            ef_search,
            seed,
        } => {
            let metric: DistanceMetric = metric.parse().map_err(|e: String| anyhow!(e))?;
            let efs = ef_search
                .split(',')
                .map(|t| t.trim().parse::<usize>().context("invalid ef_search value"))
                .collect::<Result<Vec<_>>>()?;
            run_bench(BenchParams {
                n,
                dim,
                queries,
                k,
                metric,
                m: hnsw_m,
                ef_construction,
                ef_search: efs,
                seed,
            })?;
        }
    }
    Ok(())
}

struct BenchParams {
    n: usize,
    dim: usize,
    queries: usize,
    k: usize,
    metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
    ef_search: Vec<usize>,
    seed: u64,
}

/// splitmix64-based deterministic vector generator.
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

fn run_bench(p: BenchParams) -> Result<()> {
    println!(
        "dataset: {} synthetic vectors, dim {}, {} queries, k={}, metric {:?}",
        p.n, p.dim, p.queries, p.k, p.metric
    );
    let mut rng = Rng(p.seed);
    let vectors: Vec<Vec<f32>> = (0..p.n).map(|_| rng.vector(p.dim)).collect();
    let queries: Vec<Vec<f32>> = (0..p.queries).map(|_| rng.vector(p.dim)).collect();

    // Flat baseline: ground truth (recall 1.0 by definition) + QPS floor.
    let start = Instant::now();
    let mut flat = FlatIndex::new(p.dim, p.metric);
    for (i, v) in vectors.iter().enumerate() {
        flat.add(VectorId(i as u64), Vector::from_vec(v.clone()))?;
    }
    let flat_build = start.elapsed();

    let start = Instant::now();
    let truth: Vec<HashSet<VectorId>> = queries
        .iter()
        .map(|q| Ok(flat.search(q, p.k)?.iter().map(|r| r.id).collect()))
        .collect::<Result<_>>()?;
    let flat_query = start.elapsed();
    let flat_qps = p.queries as f64 / flat_query.as_secs_f64();

    let start = Instant::now();
    let mut hnsw = HnswIndex::new(
        p.dim,
        p.metric,
        HnswConfig {
            m: p.m,
            ef_construction: p.ef_construction,
            ef_search: p.k, // overridden per sweep below
            seed: p.seed,
        },
    );
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(VectorId(i as u64), Vector::from_vec(v.clone()))?;
    }
    let hnsw_build = start.elapsed();

    println!(
        "build:   flat {:>10.3?}   hnsw(M={}, efc={}) {:>10.3?}",
        flat_build, p.m, p.ef_construction, hnsw_build
    );
    println!();
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "ef_search", "recall@k", "QPS", "vs flat"
    );
    println!(
        "{:>10} {:>12} {:>12.0} {:>9.1}x",
        "flat", "1.000", flat_qps, 1.0
    );
    for &ef in &p.ef_search {
        let start = Instant::now();
        let results: Vec<Vec<VectorId>> = queries
            .iter()
            .map(|q| {
                Ok(hnsw
                    .search_with_ef(q, p.k, ef)?
                    .iter()
                    .map(|r| r.id)
                    .collect())
            })
            .collect::<Result<_>>()?;
        let elapsed = start.elapsed();
        let qps = p.queries as f64 / elapsed.as_secs_f64();

        let hits: usize = results
            .iter()
            .zip(&truth)
            .map(|(res, t)| res.iter().filter(|id| t.contains(id)).count())
            .sum();
        let total: usize = truth.iter().map(|t| t.len()).sum();
        let recall = hits as f64 / total as f64;
        println!(
            "{:>10} {:>12.3} {:>12.0} {:>9.1}x",
            ef,
            recall,
            qps,
            qps / flat_qps
        );
    }
    Ok(())
}

fn parse_query(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|t| t.trim().parse::<f32>().context("invalid float in query"))
        .collect()
}

fn build_index(path: &Path, dim: usize, metric: DistanceMetric) -> Result<FlatIndex> {
    let mut index = FlatIndex::new(dim, metric);
    load_into(path, &mut index)?;
    Ok(index)
}

fn new_index(
    kind: &str,
    dim: usize,
    metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
) -> Result<AnyIndex> {
    match kind {
        "flat" => Ok(AnyIndex::Flat(FlatIndex::new(dim, metric))),
        "hnsw" => Ok(AnyIndex::Hnsw(HnswIndex::new(
            dim,
            metric,
            HnswConfig {
                m,
                ef_construction,
                ..HnswConfig::default()
            },
        ))),
        other => Err(anyhow!("unknown index type: {other} (expected flat|hnsw)")),
    }
}

fn load_into<I: Index>(path: &Path, index: &mut I) -> Result<()> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let dim = index.dim();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: Record =
            serde_json::from_str(&line).with_context(|| format!("parsing line {}", lineno + 1))?;
        let vector = Vector::new(rec.vector, dim)
            .with_context(|| format!("validating vector at line {}", lineno + 1))?;
        index
            .add_with_payload(VectorId(rec.id), vector, rec.payload)
            .with_context(|| format!("adding vector at line {}", lineno + 1))?;
    }
    Ok(())
}

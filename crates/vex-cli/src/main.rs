use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use vex_core::{DistanceMetric, FlatIndex, Index, Vector, VectorId};

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
    /// Load JSONL into a FlatIndex and run a search against `--query`.
    Query {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value_t = 5)]
        k: usize,
        #[arg(short = 'm', long, default_value = "l2")]
        metric: String,
    },
}

#[derive(Debug, Deserialize)]
struct Record {
    id: u64,
    vector: Vec<f32>,
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
        Command::Query {
            input,
            query,
            k,
            metric,
        } => {
            let metric: DistanceMetric = metric.parse().map_err(|e: String| anyhow!(e))?;
            let q = parse_query(&query)?;
            let index = build_index(&input, q.len(), metric)?;
            let results = index.search(&q, k)?;
            println!("top {} results:", results.len());
            for r in results {
                println!("  id={} distance={:.6}", r.id, r.distance);
            }
        }
    }
    Ok(())
}

fn parse_query(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|t| t.trim().parse::<f32>().context("invalid float in query"))
        .collect()
}

fn build_index(path: &Path, dim: usize, metric: DistanceMetric) -> Result<FlatIndex> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut index = FlatIndex::new(dim, metric);
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
            .add(VectorId(rec.id), vector)
            .with_context(|| format!("adding vector at line {}", lineno + 1))?;
    }
    Ok(index)
}

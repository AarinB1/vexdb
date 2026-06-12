//! End-to-end integration test driving the `vex` CLI.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn cli_bin() -> PathBuf {
    // CARGO_BIN_EXE_<name> is set by Cargo for integration tests of the
    // workspace's binaries.
    PathBuf::from(env!("CARGO_BIN_EXE_vex"))
}

fn write_jsonl(contents: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(".jsonl")
        .tempfile()
        .unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

#[test]
fn ingest_and_query_round_trip() {
    let jsonl = "\
{\"id\": 1, \"vector\": [1.0, 0.0, 0.0]}
{\"id\": 2, \"vector\": [0.0, 1.0, 0.0]}
{\"id\": 3, \"vector\": [0.0, 0.0, 1.0]}
{\"id\": 4, \"vector\": [1.0, 1.0, 0.0]}
";
    let f = write_jsonl(jsonl);

    let out = Command::new(cli_bin())
        .args([
            "ingest",
            "--input",
            f.path().to_str().unwrap(),
            "--dim",
            "3",
        ])
        .output()
        .expect("running vex ingest");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("ingested 4 vectors"), "stdout: {stdout}");

    let out = Command::new(cli_bin())
        .args([
            "query",
            "--input",
            f.path().to_str().unwrap(),
            "--query",
            "1.0,0.0,0.0",
            "--k",
            "2",
            "--metric",
            "l2",
        ])
        .output()
        .expect("running vex query");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("top 2 results"), "stdout: {stdout}");
    // id=1 should be the closest with distance 0.
    let first_line = stdout
        .lines()
        .find(|l| l.contains("id=1"))
        .expect("id=1 in output");
    assert!(first_line.contains("distance=0"), "got: {first_line}");
}

#[test]
fn build_snapshot_and_query_round_trip() {
    let jsonl = "\
{\"id\": 1, \"vector\": [1.0, 0.0, 0.0], \"payload\": {\"tag\": \"x\"}}
{\"id\": 2, \"vector\": [0.0, 1.0, 0.0]}
{\"id\": 3, \"vector\": [0.0, 0.0, 1.0]}
";
    let f = write_jsonl(jsonl);
    let dir = tempfile::tempdir().unwrap();
    let snap = dir.path().join("index.vex");

    let out = Command::new(cli_bin())
        .args([
            "build",
            "--input",
            f.path().to_str().unwrap(),
            "--output",
            snap.to_str().unwrap(),
            "--dim",
            "3",
            "--index",
            "hnsw",
        ])
        .output()
        .expect("running vex build");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("built hnsw index: 3 vectors"),
        "stdout: {stdout}"
    );
    assert!(snap.exists());

    let out = Command::new(cli_bin())
        .args([
            "query",
            "--snapshot",
            snap.to_str().unwrap(),
            "--query",
            "0.0,1.0,0.0",
            "--k",
            "1",
        ])
        .output()
        .expect("running vex query --snapshot");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("id=2"), "stdout: {stdout}");
}

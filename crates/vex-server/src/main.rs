use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use vex_server::{app, AppState};

#[derive(Parser, Debug)]
#[command(
    name = "vex-server",
    version,
    about = "vex vector database HTTP server"
)]
struct Args {
    /// Address to bind.
    #[arg(long, default_value = "0.0.0.0:8080", env = "VEX_ADDR")]
    addr: String,

    /// Directory holding .vex snapshots; every snapshot found at startup is
    /// loaded as a collection, and dirty collections are flushed back here.
    #[arg(long, default_value = "./data", env = "VEX_DATA_DIR")]
    data_dir: PathBuf,

    /// Serve search-only: every write endpoint returns 403. Useful for
    /// fronting a prebuilt snapshot with zero persistence concerns.
    #[arg(long, env = "VEX_READ_ONLY")]
    read_only: bool,

    /// Seconds between background flushes of dirty collections.
    #[arg(long, default_value_t = 30, env = "VEX_FLUSH_INTERVAL")]
    flush_interval: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();
    let state = Arc::new(
        AppState::load(args.data_dir.clone(), args.read_only)
            .context("loading collections from data dir")?,
    );

    // Background flush of dirty collections (skipped in read-only mode,
    // where nothing can become dirty).
    if !args.read_only {
        let flush_state = state.clone();
        let interval = Duration::from_secs(args.flush_interval.max(1));
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                let flushed = flush_state.flush_all(false);
                if flushed > 0 {
                    tracing::info!(flushed, "background flush");
                }
            }
        });
    }

    let listener = tokio::net::TcpListener::bind(&args.addr)
        .await
        .with_context(|| format!("binding {}", args.addr))?;
    tracing::info!(
        addr = %args.addr,
        data_dir = %args.data_dir.display(),
        read_only = args.read_only,
        "vex-server listening"
    );

    let shutdown_state = state.clone();
    axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Final flush so a clean shutdown never loses acknowledged writes.
    let flushed = shutdown_state.flush_all(false);
    tracing::info!(flushed, "shutdown flush complete");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("installing ctrl-c handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("installing SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received");
}

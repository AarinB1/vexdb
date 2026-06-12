//! vex-server: an HTTP API over vex-core indexes.
//!
//! The endpoint shape deliberately mirrors Qdrant's so it feels familiar:
//!
//! ```text
//! GET    /health                              liveness
//! GET    /                                    demo UI
//! GET    /collections                         list collections
//! POST   /collections/{name}                  create (dim, metric, index, m, ef_construction)
//! GET    /collections/{name}                  stats
//! DELETE /collections/{name}                  drop
//! POST   /collections/{name}/points           upsert vectors (+ payloads)
//! DELETE /collections/{name}/points/{id}      delete one vector
//! POST   /collections/{name}/search           {"query": [...], "k": 10, "ef": 64, "filter": {...}}
//! POST   /collections/{name}/snapshot         force a flush to disk
//! ```
//!
//! All async dependencies (tokio/axum/tower) are isolated in this crate;
//! vex-core stays synchronous and embeddable.

pub mod error;
pub mod routes;
pub mod state;

use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

pub use state::AppState;

/// Request bodies above this size are rejected with 413 before any JSON
/// parsing happens. 10k points × 1536 dims × ~10 bytes/float comfortably
/// fits; adjust per deployment if needed.
pub const BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

pub fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(routes::demo_page))
        .route("/health", get(routes::health))
        .route("/collections", get(routes::list_collections))
        .route(
            "/collections/:name",
            post(routes::create_collection)
                .get(routes::collection_stats)
                .delete(routes::drop_collection),
        )
        .route("/collections/:name/points", post(routes::upsert_points))
        .route(
            "/collections/:name/points/:id",
            delete(routes::delete_point),
        )
        .route("/collections/:name/search", post(routes::search))
        .route(
            "/collections/:name/snapshot",
            post(routes::snapshot_collection),
        )
        .layer(RequestBodyLimitLayer::new(BODY_LIMIT_BYTES))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

//! HTTP handlers. Deliberately thin: deserialize, validate limits, take the
//! lock, call the `Index` trait, serialize. The engineering lives in
//! vex-core; this layer is glue plus the things a network boundary forces
//! (status codes, input limits, upsert semantics).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Html;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;
use vex_core::{
    AnyIndex, DistanceMetric, Filter, FlatIndex, HnswConfig, HnswIndex, Index, Payload, Vector,
    VectorId,
};

use crate::error::ApiError;
use crate::state::{snapshot_path, validate_name, AppState, Collection};

/// Input limits. A vector DB endpoint takes arbitrary user bytes; these are
/// the guardrails (the request body size cap lives in the router layer).
pub const MAX_K: usize = 1_024;
pub const MAX_EF: usize = 100_000;
pub const MAX_BATCH: usize = 10_000;
pub const MAX_DIM: usize = 65_536;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateCollection {
    pub dim: usize,
    #[serde(default = "default_metric")]
    pub metric: DistanceMetric,
    /// "hnsw" (default) or "flat".
    #[serde(default = "default_index_type")]
    pub index: String,
    #[serde(default = "default_m")]
    pub m: usize,
    #[serde(default = "default_ef_construction")]
    pub ef_construction: usize,
    #[serde(default = "default_ef_search")]
    pub ef_search: usize,
}

fn default_metric() -> DistanceMetric {
    DistanceMetric::L2
}
fn default_index_type() -> String {
    "hnsw".into()
}
fn default_m() -> usize {
    HnswConfig::default().m
}
fn default_ef_construction() -> usize {
    HnswConfig::default().ef_construction
}
fn default_ef_search() -> usize {
    HnswConfig::default().ef_search
}

#[derive(Debug, Deserialize)]
pub struct UpsertPoints {
    pub points: Vec<PointInput>,
}

#[derive(Debug, Deserialize)]
pub struct PointInput {
    pub id: u64,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub payload: Option<Payload>,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: usize,
    /// HNSW beam width override; ignored by flat collections.
    #[serde(default)]
    pub ef: Option<usize>,
    #[serde(default)]
    pub filter: Option<Filter>,
    /// Include stored payloads in results (default true).
    #[serde(default = "default_true")]
    pub with_payload: bool,
}

fn default_k() -> usize {
    10
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub id: u64,
    pub distance: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Payload>,
}

#[derive(Debug, Serialize)]
pub struct CollectionStats {
    pub name: String,
    pub index: &'static str,
    pub dim: usize,
    pub metric: DistanceMetric,
    pub count: usize,
}

fn stats(c: &Collection) -> Result<CollectionStats, ApiError> {
    let index = lock_read(c)?;
    Ok(CollectionStats {
        name: c.name.clone(),
        index: index.type_name(),
        dim: index.dim(),
        metric: index.metric(),
        count: index.len(),
    })
}

fn lock_read(c: &Collection) -> Result<std::sync::RwLockReadGuard<'_, AnyIndex>, ApiError> {
    c.index
        .read()
        .map_err(|_| ApiError::Internal("index lock poisoned".into()))
}

fn lock_write(c: &Collection) -> Result<std::sync::RwLockWriteGuard<'_, AnyIndex>, ApiError> {
    c.index
        .write()
        .map_err(|_| ApiError::Internal("index lock poisoned".into()))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

pub async fn demo_page() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

pub async fn list_collections(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let collections = state
        .collections
        .read()
        .map_err(|_| ApiError::Internal("collections lock poisoned".into()))?
        .values()
        .cloned()
        .collect::<Vec<_>>();
    let mut out = Vec::with_capacity(collections.len());
    for c in &collections {
        out.push(stats(c)?);
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Json(json!({ "collections": out })))
}

pub async fn create_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<CreateCollection>,
) -> Result<(StatusCode, Json<CollectionStats>), ApiError> {
    state.ensure_writable()?;
    validate_name(&name)?;
    if req.dim == 0 || req.dim > MAX_DIM {
        return Err(ApiError::BadRequest(format!(
            "dim must be 1..={MAX_DIM}, got {}",
            req.dim
        )));
    }
    let index = match req.index.as_str() {
        "flat" => AnyIndex::Flat(FlatIndex::new(req.dim, req.metric)),
        "hnsw" => {
            if req.m == 0 || req.m > 256 {
                return Err(ApiError::BadRequest("m must be 1..=256".into()));
            }
            AnyIndex::Hnsw(HnswIndex::new(
                req.dim,
                req.metric,
                HnswConfig {
                    m: req.m,
                    ef_construction: req.ef_construction.clamp(1, MAX_EF),
                    ef_search: req.ef_search.clamp(1, MAX_EF),
                    ..HnswConfig::default()
                },
            ))
        }
        other => {
            return Err(ApiError::BadRequest(format!(
                "unknown index type '{other}' (expected flat|hnsw)"
            )))
        }
    };

    let mut collections = state
        .collections
        .write()
        .map_err(|_| ApiError::Internal("collections lock poisoned".into()))?;
    if collections.contains_key(&name) {
        return Err(ApiError::Conflict(format!(
            "collection '{name}' already exists"
        )));
    }
    let collection = Arc::new(Collection::new(name.clone(), index));
    // Persist immediately so an empty collection survives a restart.
    collection.flush(&state.data_dir, true)?;
    collections.insert(name, collection.clone());
    drop(collections);

    Ok((StatusCode::CREATED, Json(stats(&collection)?)))
}

pub async fn drop_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, ApiError> {
    state.ensure_writable()?;
    let mut collections = state
        .collections
        .write()
        .map_err(|_| ApiError::Internal("collections lock poisoned".into()))?;
    if collections.remove(&name).is_none() {
        return Err(ApiError::NotFound(format!(
            "collection '{name}' does not exist"
        )));
    }
    drop(collections);
    let path = snapshot_path(&state.data_dir, &name);
    if path.exists() {
        std::fs::remove_file(&path)
            .map_err(|e| ApiError::Internal(format!("removing snapshot: {e}")))?;
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn collection_stats(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<CollectionStats>, ApiError> {
    let collection = state.get(&name)?;
    Ok(Json(stats(&collection)?))
}

pub async fn upsert_points(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpsertPoints>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state.ensure_writable()?;
    if req.points.is_empty() {
        return Err(ApiError::BadRequest("points must not be empty".into()));
    }
    if req.points.len() > MAX_BATCH {
        return Err(ApiError::BadRequest(format!(
            "batch too large: {} points (max {MAX_BATCH})",
            req.points.len()
        )));
    }
    let collection = state.get(&name)?;
    let mut index = lock_write(&collection)?;
    // Validate the whole batch before touching the index, so a bad point
    // mid-batch can't leave a half-applied request.
    let dim = index.dim();
    for (i, p) in req.points.iter().enumerate() {
        if p.vector.len() != dim {
            return Err(ApiError::BadRequest(format!(
                "point {} (id {}): dimension mismatch: expected {dim}, got {}",
                i,
                p.id,
                p.vector.len()
            )));
        }
    }
    let count = req.points.len();
    for p in req.points {
        let id = VectorId(p.id);
        // Upsert semantics: replace silently if the id exists.
        index.remove(id)?;
        let vector = Vector::new(p.vector, dim).map_err(ApiError::from)?;
        index.add_with_payload(id, vector, p.payload)?;
    }
    drop(index);
    collection.dirty.store(true, Ordering::Release);
    Ok(Json(json!({ "upserted": count })))
}

pub async fn delete_point(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, u64)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state.ensure_writable()?;
    let collection = state.get(&name)?;
    let removed = lock_write(&collection)?.remove(VectorId(id))?;
    if removed {
        collection.dirty.store(true, Ordering::Release);
    }
    Ok(Json(json!({ "removed": removed })))
}

pub async fn search(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if req.k == 0 || req.k > MAX_K {
        return Err(ApiError::BadRequest(format!(
            "k must be 1..={MAX_K}, got {}",
            req.k
        )));
    }
    if req.ef.is_some_and(|ef| ef > MAX_EF) {
        return Err(ApiError::BadRequest(format!("ef must be <= {MAX_EF}")));
    }
    let collection = state.get(&name)?;
    let index = lock_read(&collection)?;
    let results = index.search_opts(&req.query, req.k, req.ef, req.filter.as_ref())?;
    let hits: Vec<SearchHit> = results
        .into_iter()
        .map(|r| SearchHit {
            id: r.id.0,
            distance: r.distance,
            payload: if req.with_payload {
                index.payload(r.id).cloned()
            } else {
                None
            },
        })
        .collect();
    Ok(Json(json!({ "results": hits })))
}

pub async fn snapshot_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state.ensure_writable()?;
    let collection = state.get(&name)?;
    collection.flush(&state.data_dir, true)?;
    Ok(Json(json!({ "saved": true })))
}

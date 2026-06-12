//! HTTP API integration tests, driving the router directly with
//! `tower::ServiceExt::oneshot` — no sockets, no flakiness.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use vex_server::{app, AppState};

fn test_state(dir: &std::path::Path, read_only: bool) -> Arc<AppState> {
    Arc::new(AppState::load(dir.to_path_buf(), read_only).unwrap())
}

async fn call(
    state: Arc<AppState>,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(match body {
            Some(v) => Body::from(v.to_string()),
            None => Body::empty(),
        })
        .unwrap();
    let res = app(state).oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, value)
}

fn seed_points(n: usize, dim: usize) -> Value {
    let points: Vec<Value> = (0..n)
        .map(|i| {
            let vector: Vec<f32> = (0..dim)
                .map(|d| ((i * 31 + d * 7) % 17) as f32 / 17.0)
                .collect();
            json!({"id": i, "vector": vector, "payload": {"bucket": i % 5}})
        })
        .collect();
    json!({ "points": points })
}

#[tokio::test]
async fn health_works() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(dir.path(), false);
    let (status, body) = call(state, "GET", "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn demo_page_served_at_root() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(dir.path(), false);
    let req = Request::builder().uri("/").body(Body::empty()).unwrap();
    let res = app(state).oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert!(String::from_utf8_lossy(&bytes).contains("vexdb console"));
}

#[tokio::test]
async fn full_crud_and_search_flow() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(dir.path(), false);

    // Create.
    let (status, body) = call(
        state.clone(),
        "POST",
        "/collections/test",
        Some(json!({"dim": 4, "metric": "l2", "index": "hnsw"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["dim"], 4);

    // Duplicate create → 409.
    let (status, _) = call(
        state.clone(),
        "POST",
        "/collections/test",
        Some(json!({"dim": 4})),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);

    // Upsert points.
    let (status, body) = call(
        state.clone(),
        "POST",
        "/collections/test/points",
        Some(seed_points(50, 4)),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["upserted"], 50);

    // Stats reflect the points.
    let (status, body) = call(state.clone(), "GET", "/collections/test", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], 50);
    assert_eq!(body["index"], "hnsw");

    // Search.
    let (status, body) = call(
        state.clone(),
        "POST",
        "/collections/test/search",
        Some(json!({"query": [0.1, 0.2, 0.3, 0.4], "k": 5, "ef": 32})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 5);
    assert!(results[0]["payload"]["bucket"].is_number());
    // Sorted ascending.
    let d: Vec<f64> = results
        .iter()
        .map(|r| r["distance"].as_f64().unwrap())
        .collect();
    assert!(d.windows(2).all(|w| w[0] <= w[1]));

    // Filtered search only returns the matching bucket.
    let (status, body) = call(
        state.clone(),
        "POST",
        "/collections/test/search",
        Some(json!({
            "query": [0.1, 0.2, 0.3, 0.4], "k": 5, "ef": 64,
            "filter": {"eq": {"key": "bucket", "value": 2}}
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    for r in body["results"].as_array().unwrap() {
        assert_eq!(r["payload"]["bucket"], 2);
    }

    // Upsert replaces an existing id.
    let (status, _) = call(
        state.clone(),
        "POST",
        "/collections/test/points",
        Some(json!({"points": [{"id": 0, "vector": [9.0, 9.0, 9.0, 9.0], "payload": {"bucket": 99}}]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, body) = call(state.clone(), "GET", "/collections/test", None).await;
    assert_eq!(body["count"], 50, "upsert must not grow the collection");

    // Delete a point.
    let (status, body) = call(state.clone(), "DELETE", "/collections/test/points/0", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["removed"], true);
    let (_, body) = call(state.clone(), "DELETE", "/collections/test/points/0", None).await;
    assert_eq!(body["removed"], false, "delete is idempotent");

    // Drop.
    let (status, _) = call(state.clone(), "DELETE", "/collections/test", None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = call(state.clone(), "GET", "/collections/test", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn validation_errors_map_to_http_statuses() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(dir.path(), false);

    // Bad collection name → 400.
    let (status, _) = call(
        state.clone(),
        "POST",
        "/collections/..%2Fevil",
        Some(json!({"dim": 4})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Unknown collection → 404.
    let (status, _) = call(
        state.clone(),
        "POST",
        "/collections/nope/search",
        Some(json!({"query": [1.0], "k": 1})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    call(
        state.clone(),
        "POST",
        "/collections/v",
        Some(json!({"dim": 4})),
    )
    .await;

    // Dimension mismatch → 400.
    let (status, body) = call(
        state.clone(),
        "POST",
        "/collections/v/search",
        Some(json!({"query": [1.0, 2.0], "k": 1})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"].as_str().unwrap().contains("dimension"));

    // k over the cap → 400.
    let (status, _) = call(
        state.clone(),
        "POST",
        "/collections/v/search",
        Some(json!({"query": [1.0, 2.0, 3.0, 4.0], "k": 100000})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Mismatched point dim in a batch → 400, atomically rejected.
    let (status, _) = call(
        state.clone(),
        "POST",
        "/collections/v/points",
        Some(json!({"points": [
            {"id": 1, "vector": [1.0, 2.0, 3.0, 4.0]},
            {"id": 2, "vector": [1.0]}
        ]})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (_, body) = call(state.clone(), "GET", "/collections/v", None).await;
    assert_eq!(body["count"], 0, "bad batch must not be half-applied");
}

#[tokio::test]
async fn read_only_mode_rejects_writes_but_serves_reads() {
    let dir = tempfile::tempdir().unwrap();

    // Seed a snapshot with a writable instance.
    {
        let state = test_state(dir.path(), false);
        call(
            state.clone(),
            "POST",
            "/collections/ro",
            Some(json!({"dim": 4, "index": "flat"})),
        )
        .await;
        call(
            state.clone(),
            "POST",
            "/collections/ro/points",
            Some(seed_points(10, 4)),
        )
        .await;
        let (status, _) = call(state, "POST", "/collections/ro/snapshot", None).await;
        assert_eq!(status, StatusCode::OK);
    }

    // Reopen read-only over the same data dir.
    let state = test_state(dir.path(), true);
    let (status, body) = call(
        state.clone(),
        "POST",
        "/collections/ro/search",
        Some(json!({"query": [0.5, 0.5, 0.5, 0.5], "k": 3})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["results"].as_array().unwrap().len(), 3);

    for (method, path, body) in [
        ("POST", "/collections/other", Some(json!({"dim": 2}))),
        ("POST", "/collections/ro/points", Some(seed_points(1, 4))),
        ("DELETE", "/collections/ro/points/1", None),
        ("DELETE", "/collections/ro", None),
        ("POST", "/collections/ro/snapshot", None),
    ] {
        let (status, _) = call(state.clone(), method, path, body).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{method} {path}");
    }
}

#[tokio::test]
async fn snapshot_persists_across_restart() {
    let dir = tempfile::tempdir().unwrap();

    {
        let state = test_state(dir.path(), false);
        call(
            state.clone(),
            "POST",
            "/collections/persist",
            Some(json!({"dim": 4, "index": "hnsw"})),
        )
        .await;
        call(
            state.clone(),
            "POST",
            "/collections/persist/points",
            Some(seed_points(25, 4)),
        )
        .await;
        // Explicit flush (a real shutdown would also flush).
        let (status, _) = call(state, "POST", "/collections/persist/snapshot", None).await;
        assert_eq!(status, StatusCode::OK);
    }

    // "Restart": brand-new state over the same data dir.
    let state = test_state(dir.path(), false);
    let (status, body) = call(state.clone(), "GET", "/collections/persist", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["count"], 25);
    let (status, body) = call(
        state,
        "POST",
        "/collections/persist/search",
        Some(json!({"query": [0.1, 0.1, 0.1, 0.1], "k": 5})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["results"].as_array().unwrap().len(), 5);
}

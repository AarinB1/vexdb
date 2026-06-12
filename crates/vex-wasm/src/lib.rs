//! WebAssembly bindings for vex-core: the whole engine — snapshot loading,
//! HNSW search, payload filters — running inside a browser tab.
//!
//! Built with `wasm-pack build --target web`; the landing page demo loads a
//! prebuilt `.vex` snapshot over `fetch` and searches it locally. No server.
//!
//! Layout note: all real logic lives in [`imp`] with plain Rust types so it
//! is unit-testable on the host; the `#[wasm_bindgen]` layer only converts
//! at the JS boundary.

use serde::Serialize;
use vex_core::Index;
use wasm_bindgen::prelude::*;

/// Serialize into plain JS objects/numbers (not `Map`s/BigInts): payloads
/// become `{...}` and u64 ids become JS numbers, which is what page code
/// expects. Demo-scale ids fit comfortably in f64's 2^53.
fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Host-testable core of the bindings.
pub mod imp {
    use serde::Serialize;
    use vex_core::{
        AnyIndex, DistanceMetric, Filter, HnswConfig, HnswIndex, Index, SnapshotError, Vector,
        VectorId,
    };

    #[derive(Debug, Serialize)]
    pub struct Hit {
        pub id: u64,
        pub distance: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub payload: Option<serde_json::Value>,
    }

    #[derive(Debug, Serialize)]
    pub struct Stats {
        pub index: &'static str,
        pub metric: String,
        pub dim: usize,
        pub count: usize,
    }

    pub struct Inner {
        pub index: AnyIndex,
    }

    impl Inner {
        pub fn from_snapshot(bytes: &[u8]) -> Result<Self, SnapshotError> {
            Ok(Self {
                index: AnyIndex::read_from(bytes)?,
            })
        }

        pub fn new_hnsw(
            dim: usize,
            metric: &str,
            m: usize,
            ef_construction: usize,
        ) -> Result<Self, String> {
            let metric: DistanceMetric = metric.parse()?;
            Ok(Self {
                index: AnyIndex::Hnsw(HnswIndex::new(
                    dim,
                    metric,
                    HnswConfig {
                        m,
                        ef_construction,
                        ..HnswConfig::default()
                    },
                )),
            })
        }

        pub fn add(
            &mut self,
            id: u64,
            vector: &[f32],
            payload_json: Option<&str>,
        ) -> Result<(), String> {
            let payload = payload_json
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| format!("payload is not valid JSON: {e}"))?;
            let vector =
                Vector::new(vector.to_vec(), self.index.dim()).map_err(|e| e.to_string())?;
            self.index
                .add_with_payload(VectorId(id), vector, payload)
                .map_err(|e| e.to_string())
        }

        pub fn search(
            &self,
            query: &[f32],
            k: usize,
            ef: Option<usize>,
            filter_json: Option<&str>,
        ) -> Result<Vec<Hit>, String> {
            let filter: Option<Filter> = filter_json
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| format!("filter is not valid JSON: {e}"))?;
            let results = self
                .index
                .search_opts(query, k, ef, filter.as_ref())
                .map_err(|e| e.to_string())?;
            Ok(results
                .into_iter()
                .map(|r| Hit {
                    id: r.id.0,
                    distance: r.distance,
                    payload: self.index.payload(r.id).cloned(),
                })
                .collect())
        }

        pub fn vector_of(&self, id: u64) -> Option<Vec<f32>> {
            self.index
                .vector(VectorId(id))
                .map(|v| v.as_slice().to_vec())
        }

        pub fn payload_of(&self, id: u64) -> Option<serde_json::Value> {
            self.index.payload(VectorId(id)).cloned()
        }

        pub fn stats(&self) -> Stats {
            Stats {
                index: self.index.type_name(),
                metric: format!("{:?}", self.index.metric()).to_lowercase(),
                dim: self.index.dim(),
                count: self.index.len(),
            }
        }

        pub fn snapshot_bytes(&self) -> Result<Vec<u8>, SnapshotError> {
            let mut buf = Vec::new();
            self.index.write_to(&mut buf)?;
            Ok(buf)
        }
    }
}

/// A vex index living in WebAssembly memory.
#[wasm_bindgen]
pub struct VexIndex {
    inner: imp::Inner,
}

#[wasm_bindgen]
impl VexIndex {
    /// Load an index from `.vex` snapshot bytes (e.g. a `fetch`ed file).
    #[wasm_bindgen(js_name = fromSnapshot)]
    pub fn from_snapshot(bytes: &[u8]) -> Result<VexIndex, JsError> {
        console_error_panic_hook::set_once();
        let inner = imp::Inner::from_snapshot(bytes).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(VexIndex { inner })
    }

    /// Create an empty HNSW index (metric: "cosine" | "l2" | "dot").
    #[wasm_bindgen(js_name = newHnsw)]
    pub fn new_hnsw(
        dim: usize,
        metric: &str,
        m: usize,
        ef_construction: usize,
    ) -> Result<VexIndex, JsError> {
        console_error_panic_hook::set_once();
        let inner =
            imp::Inner::new_hnsw(dim, metric, m, ef_construction).map_err(|e| JsError::new(&e))?;
        Ok(VexIndex { inner })
    }

    /// Insert a vector with an optional JSON payload string.
    pub fn add(
        &mut self,
        id: u64,
        vector: &[f32],
        payload_json: Option<String>,
    ) -> Result<(), JsError> {
        self.inner
            .add(id, vector, payload_json.as_deref())
            .map_err(|e| JsError::new(&e))
    }

    /// Top-k search. `ef` tunes the HNSW beam width (0 = index default);
    /// `filter_json` is the same filter language the HTTP API accepts.
    /// Returns `[{id, distance, payload?}, ...]` sorted ascending.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_json: Option<String>,
    ) -> Result<JsValue, JsError> {
        let ef = (ef > 0).then_some(ef);
        let hits = self
            .inner
            .search(query, k, ef, filter_json.as_deref())
            .map_err(|e| JsError::new(&e))?;
        to_js(&hits)
    }

    /// The stored vector for an id — search with it for "more like this".
    #[wasm_bindgen(js_name = vectorOf)]
    pub fn vector_of(&self, id: u64) -> Option<Vec<f32>> {
        self.inner.vector_of(id)
    }

    /// The stored payload for an id.
    #[wasm_bindgen(js_name = payloadOf)]
    pub fn payload_of(&self, id: u64) -> Result<JsValue, JsError> {
        to_js(&self.inner.payload_of(id))
    }

    /// `{index, metric, dim, count}`.
    pub fn stats(&self) -> Result<JsValue, JsError> {
        to_js(&self.inner.stats())
    }

    /// Serialize the index back to `.vex` bytes (e.g. to download an index
    /// built in the browser).
    #[wasm_bindgen(js_name = toSnapshot)]
    pub fn to_snapshot(&self) -> Result<Vec<u8>, JsError> {
        self.inner
            .snapshot_bytes()
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn len(&self) -> usize {
        self.inner.index.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.index.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.inner.index.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::imp::Inner;

    fn build_demo() -> Inner {
        let mut inner = Inner::new_hnsw(4, "cosine", 8, 50).unwrap();
        for i in 0..200u64 {
            let f = i as f32;
            let v = [f.sin(), f.cos(), (f * 0.1).sin(), 1.0];
            let payload = format!(r#"{{"n": {}, "bucket": {}}}"#, i, i % 4);
            inner.add(i, &v, Some(&payload)).unwrap();
        }
        inner
    }

    #[test]
    fn build_search_and_filter() {
        let inner = build_demo();
        assert_eq!(inner.stats().count, 200);

        let query = inner.vector_of(7).unwrap();
        let hits = inner.search(&query, 5, Some(64), None).unwrap();
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].id, 7, "self should be its own nearest neighbor");
        assert!(hits[0].payload.as_ref().unwrap()["n"] == 7);

        let filtered = inner
            .search(
                &query,
                5,
                Some(64),
                Some(r#"{"eq":{"key":"bucket","value":2}}"#),
            )
            .unwrap();
        assert_eq!(filtered.len(), 5);
        for h in &filtered {
            assert_eq!(h.id % 4, 2);
        }
    }

    #[test]
    fn snapshot_roundtrip_in_memory() {
        let inner = build_demo();
        let bytes = inner.snapshot_bytes().unwrap();
        let reloaded = Inner::from_snapshot(&bytes).unwrap();
        assert_eq!(reloaded.stats().count, 200);
        let q = reloaded.vector_of(3).unwrap();
        let a = inner.search(&q, 10, Some(80), None).unwrap();
        let b = reloaded.search(&q, 10, Some(80), None).unwrap();
        let ids_a: Vec<u64> = a.iter().map(|h| h.id).collect();
        let ids_b: Vec<u64> = b.iter().map(|h| h.id).collect();
        assert_eq!(ids_a, ids_b, "results must be identical after round-trip");
    }

    #[test]
    fn errors_are_strings_not_panics() {
        let mut inner = Inner::new_hnsw(4, "cosine", 8, 50).unwrap();
        assert!(inner.add(1, &[1.0, 2.0], None).is_err(), "dim mismatch");
        assert!(Inner::from_snapshot(b"garbage").is_err());
        assert!(Inner::new_hnsw(4, "bogus-metric", 8, 50).is_err());
        inner.add(1, &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert!(inner
            .search(&[1.0, 0.0, 0.0, 0.0], 5, None, Some("not json"))
            .is_err());
    }
}

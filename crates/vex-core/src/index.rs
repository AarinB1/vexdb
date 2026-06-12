use crate::error::Result;
use crate::vector::{Vector, VectorId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
}

/// Common interface for any index implementation.
///
/// # Contract
///
/// Every implementation follows the same edge-case philosophy so callers can
/// swap indexes without changing error handling:
///
/// - **Empty index / `k > len()`**: never an error. Searching an empty index
///   yields zero results; `k` larger than the number of stored vectors yields
///   `len()` results. The only `search` errors are `InvalidK` (k == 0) and
///   `DimensionMismatch`.
/// - **`remove` of a missing id**: returns `Ok(false)`, not an error.
///   Removal is idempotent. Implementations are free to use tombstones
///   internally (HNSW does — true graph repair is famously awkward), but
///   `len()` always reports *live* vectors and removed ids never appear in
///   search results.
/// - **Results** are sorted ascending by distance (smaller = more similar,
///   for every metric).
pub trait Index {
    fn add(&mut self, id: VectorId, vector: Vector) -> Result<()>;
    fn remove(&mut self, id: VectorId) -> Result<bool>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn dim(&self) -> usize;
}

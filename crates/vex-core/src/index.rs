use crate::error::Result;
use crate::payload::{Filter, Payload};
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
/// - **Filtered search** only returns vectors whose payload matches the
///   filter; vectors without a payload match only vacuous filters.
pub trait Index {
    fn add_with_payload(
        &mut self,
        id: VectorId,
        vector: Vector,
        payload: Option<Payload>,
    ) -> Result<()>;

    fn add(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        self.add_with_payload(id, vector, None)
    }

    fn remove(&mut self, id: VectorId) -> Result<bool>;

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>>;

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_filtered(query, k, None)
    }

    /// The payload stored with a live vector, if any.
    fn payload(&self, id: VectorId) -> Option<&Payload>;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn dim(&self) -> usize;
}

/// Run many queries in parallel across a thread pool. Reads on a frozen
/// index are embarrassingly parallel — every `search` takes `&self`.
///
/// Requires the `rayon` feature (on by default; build vex-core with
/// `default-features = false` for a dependency-lean embeddable core).
#[cfg(feature = "rayon")]
pub fn search_batch<I: Index + Sync>(
    index: &I,
    queries: &[Vec<f32>],
    k: usize,
) -> Result<Vec<Vec<SearchResult>>> {
    use rayon::prelude::*;
    queries.par_iter().map(|q| index.search(q, k)).collect()
}

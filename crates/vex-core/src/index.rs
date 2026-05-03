use crate::error::Result;
use crate::vector::{Vector, VectorId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
}

/// Common interface for any index implementation. `FlatIndex` is the only
/// implementor in Phase 1; HNSW etc. will plug in here in later phases.
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

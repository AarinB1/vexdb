//! Core types and index implementations for vexdb.
//!
//! Phase 1: `FlatIndex` brute force, scalar distance.
//! Phase 2: `HnswIndex` approximate nearest-neighbor search.

pub mod distance;
pub mod error;
pub mod flat;
pub mod hnsw;
pub mod index;
pub mod vector;

pub use distance::DistanceMetric;
pub use error::{Result, VexError};
pub use flat::FlatIndex;
pub use hnsw::{HnswConfig, HnswIndex};
pub use index::{Index, SearchResult};
pub use vector::{Vector, VectorId};

//! Core types and the `FlatIndex` brute-force implementation for vexdb.
//!
//! Phase 1: scalar distance, no SIMD, no approximate indexing.

pub mod distance;
pub mod error;
pub mod flat;
pub mod index;
pub mod vector;

pub use distance::DistanceMetric;
pub use error::{Result, VexError};
pub use flat::FlatIndex;
pub use index::{Index, SearchResult};
pub use vector::{Vector, VectorId};

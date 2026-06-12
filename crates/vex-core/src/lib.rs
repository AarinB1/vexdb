//! Core types and index implementations for vexdb.
//!
//! Phase 1: `FlatIndex` brute force. Phase 2: `HnswIndex` approximate
//! nearest-neighbor search. Phase 3: binary snapshot persistence.
//! Phase 4: parallel batch reads. Phase 7: payloads + filtered search.
//! Phase 8: SIMD distance kernels. Phase 11: search tracing
//! (`HnswIndex::search_traced`) + the exact-scan recall oracle.

pub mod distance;
pub mod error;
pub mod flat;
pub mod hnsw;
pub mod index;
pub mod payload;
pub mod persist;
pub mod vector;

pub use distance::DistanceMetric;
pub use error::{Result, VexError};
pub use flat::FlatIndex;
pub use hnsw::{BeamEdge, BeamStatus, DescentHop, HnswConfig, HnswIndex, SearchTrace};
#[cfg(feature = "rayon")]
pub use index::search_batch;
pub use index::{Index, SearchResult};
pub use payload::{Filter, Payload};
pub use persist::{AnyIndex, SnapshotError};
pub use vector::{Vector, VectorId};

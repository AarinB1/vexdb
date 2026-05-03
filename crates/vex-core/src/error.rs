use thiserror::Error;

use crate::vector::VectorId;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum VexError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("duplicate id: {0:?}")]
    DuplicateId(VectorId),

    #[error("id not found: {0:?}")]
    IdNotFound(VectorId),

    #[error("invalid k: must be > 0")]
    InvalidK,

    #[error("index is empty")]
    EmptyIndex,
}

pub type Result<T> = std::result::Result<T, VexError>;

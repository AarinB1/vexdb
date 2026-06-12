use thiserror::Error;

use crate::vector::VectorId;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum VexError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("duplicate id: {0:?}")]
    DuplicateId(VectorId),

    #[error("invalid k: must be > 0")]
    InvalidK,
}

pub type Result<T> = std::result::Result<T, VexError>;

use crate::error::{Result, VexError};

use serde::{Deserialize, Serialize};

/// Stable 64-bit identifier for a vector. `u64` (not `usize`) so the on-disk
/// format and IDs are portable across 32/64-bit targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VectorId(pub u64);

impl From<u64> for VectorId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl std::fmt::Display for VectorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A dense `f32` vector with a fixed dimension.
///
/// Intentionally not `Copy`: vectors can be large, and we want move semantics
/// to be the obvious cost.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<f32>,
}

impl Vector {
    /// Construct a `Vector`, validating it matches the expected dimension.
    pub fn new(data: Vec<f32>, expected_dim: usize) -> Result<Self> {
        if data.len() != expected_dim {
            return Err(VexError::DimensionMismatch {
                expected: expected_dim,
                actual: data.len(),
            });
        }
        Ok(Self { data })
    }

    /// Construct from a `Vec<f32>` without an external expected dim. The
    /// dim is whatever the vector's length is. Useful for tests / ad-hoc use.
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.data
    }
}

impl AsRef<[f32]> for Vector {
    fn as_ref(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_validates_dim() {
        let v = Vector::new(vec![1.0, 2.0, 3.0], 3).unwrap();
        assert_eq!(v.dim(), 3);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn new_rejects_mismatched_dim() {
        let err = Vector::new(vec![1.0, 2.0], 3).unwrap_err();
        assert_eq!(
            err,
            VexError::DimensionMismatch {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn vector_id_traits() {
        let a = VectorId(1);
        let b = VectorId::from(1u64);
        assert_eq!(a, b);
        assert!(VectorId(1) < VectorId(2));
    }
}

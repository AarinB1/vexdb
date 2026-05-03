use crate::error::{Result, VexError};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Distance metric. All variants return values where *smaller is more similar*,
/// so the search code can treat them uniformly. For `Cosine` we therefore
/// return `1 - cos_sim`. For `Dot` we return `-dot(a, b)` so larger dot
/// products yield smaller distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum DistanceMetric {
    Cosine,
    L2,
    Dot,
}

impl DistanceMetric {
    /// Compute the distance between two equal-length slices.
    ///
    /// Naive scalar implementation — SIMD is intentionally out of scope for
    /// Phase 1. Validates dimension equality.
    pub fn distance(self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VexError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        Ok(match self {
            DistanceMetric::L2 => l2(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Dot => -dot(a, b),
        })
    }
}

impl std::str::FromStr for DistanceMetric {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cosine" => Ok(Self::Cosine),
            "l2" | "euclidean" => Ok(Self::L2),
            "dot" | "ip" | "inner_product" => Ok(Self::Dot),
            other => Err(format!("unknown distance metric: {other}")),
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn l2(a: &[f32], b: &[f32]) -> f32 {
    // Squared Euclidean would be faster and produce the same ordering, but
    // returning the actual distance is friendlier for users inspecting raw
    // numbers. Phase 5 may swap to squared + take sqrt only when reporting.
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

#[inline]
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_ab = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot_ab += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    // A zero vector has undefined cosine; treat it as maximally distant
    // rather than NaN-poisoning the heap.
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - (dot_ab / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn l2_known_values() {
        let d = DistanceMetric::L2
            .distance(&[0.0, 0.0], &[3.0, 4.0])
            .unwrap();
        assert!(approx(d, 5.0), "got {d}");
    }

    #[test]
    fn dot_known_values() {
        // dot([1,2,3],[4,5,6]) = 32 -> distance = -32
        let d = DistanceMetric::Dot
            .distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .unwrap();
        assert!(approx(d, -32.0), "got {d}");
    }

    #[test]
    fn cosine_identical_is_zero() {
        let d = DistanceMetric::Cosine
            .distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
            .unwrap();
        assert!(approx(d, 0.0), "got {d}");
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        let d = DistanceMetric::Cosine
            .distance(&[1.0, 0.0], &[0.0, 1.0])
            .unwrap();
        assert!(approx(d, 1.0), "got {d}");
    }

    #[test]
    fn cosine_opposite_is_two() {
        let d = DistanceMetric::Cosine
            .distance(&[1.0, 0.0], &[-1.0, 0.0])
            .unwrap();
        assert!(approx(d, 2.0), "got {d}");
    }

    #[test]
    fn cosine_zero_vector_is_one() {
        let d = DistanceMetric::Cosine
            .distance(&[0.0, 0.0], &[1.0, 0.0])
            .unwrap();
        assert!(approx(d, 1.0));
    }

    #[test]
    fn dim_mismatch_errors() {
        let err = DistanceMetric::L2
            .distance(&[1.0], &[1.0, 2.0])
            .unwrap_err();
        assert!(matches!(err, VexError::DimensionMismatch { .. }));
    }

    #[test]
    fn from_str() {
        use std::str::FromStr;
        assert_eq!(
            DistanceMetric::from_str("cosine").unwrap(),
            DistanceMetric::Cosine
        );
        assert_eq!(DistanceMetric::from_str("L2").unwrap(), DistanceMetric::L2);
        assert_eq!(
            DistanceMetric::from_str("dot").unwrap(),
            DistanceMetric::Dot
        );
        assert!(DistanceMetric::from_str("manhattan").is_err());
    }
}

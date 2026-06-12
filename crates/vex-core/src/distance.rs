use crate::error::{Result, VexError};

use serde::{Deserialize, Serialize};

/// Distance metric. All variants return values where *smaller is more similar*,
/// so the search code can treat them uniformly. For `Cosine` we therefore
/// return `1 - cos_sim`. For `Dot` we return `-dot(a, b)` so larger dot
/// products yield smaller distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    Cosine,
    L2,
    Dot,
}

impl DistanceMetric {
    /// Stable single-byte tag for the on-disk snapshot format.
    pub(crate) fn to_tag(self) -> u8 {
        match self {
            DistanceMetric::Cosine => 0,
            DistanceMetric::L2 => 1,
            DistanceMetric::Dot => 2,
        }
    }

    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(DistanceMetric::Cosine),
            1 => Some(DistanceMetric::L2),
            2 => Some(DistanceMetric::Dot),
            _ => None,
        }
    }
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

// ---------------------------------------------------------------------------
// Distance kernels.
//
// Each metric has two implementations: a scalar kernel structured with
// 8-wide independent accumulators (float reductions can't be auto-vectorized
// in their naive form because FP addition isn't associative; making the
// lane structure explicit lets LLVM vectorize it), and an AVX2+FMA kernel
// selected at runtime on x86_64. The scalar kernels are the correctness
// oracle — property tests assert the SIMD paths agree to within reordering
// tolerance.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
fn use_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        // SAFETY: AVX2 + FMA presence checked above.
        return unsafe { simd_x86::dot_avx2(a, b) };
    }
    dot_scalar(a, b)
}

#[inline]
fn l2(a: &[f32], b: &[f32]) -> f32 {
    // Squared Euclidean would produce the same ordering, but returning the
    // actual distance is friendlier for users inspecting raw numbers.
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        // SAFETY: AVX2 + FMA presence checked above.
        return unsafe { simd_x86::l2_sq_avx2(a, b) }.sqrt();
    }
    l2_sq_scalar(a, b).sqrt()
}

#[inline]
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        // SAFETY: AVX2 + FMA presence checked above.
        let (dot_ab, norm_a, norm_b) = unsafe { simd_x86::cosine_terms_avx2(a, b) };
        return cosine_from_terms(dot_ab, norm_a, norm_b);
    }
    let (dot_ab, norm_a, norm_b) = cosine_terms_scalar(a, b);
    cosine_from_terms(dot_ab, norm_a, norm_b)
}

#[inline]
fn cosine_from_terms(dot_ab: f32, norm_a: f32, norm_b: f32) -> f32 {
    let denom = norm_a.sqrt() * norm_b.sqrt();
    // A zero vector has undefined cosine; treat it as maximally distant
    // rather than NaN-poisoning the heap.
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - (dot_ab / denom)
}

const LANES: usize = 8;

#[inline]
pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (a_chunks, a_rem) = a.split_at(a.len() - a.len() % LANES);
    let (b_chunks, b_rem) = b.split_at(b.len() - b.len() % LANES);
    for (ca, cb) in a_chunks
        .chunks_exact(LANES)
        .zip(b_chunks.chunks_exact(LANES))
    {
        for l in 0..LANES {
            acc[l] += ca[l] * cb[l];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in a_rem.iter().zip(b_rem) {
        sum += x * y;
    }
    sum
}

#[inline]
pub(crate) fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (a_chunks, a_rem) = a.split_at(a.len() - a.len() % LANES);
    let (b_chunks, b_rem) = b.split_at(b.len() - b.len() % LANES);
    for (ca, cb) in a_chunks
        .chunks_exact(LANES)
        .zip(b_chunks.chunks_exact(LANES))
    {
        for l in 0..LANES {
            let d = ca[l] - cb[l];
            acc[l] += d * d;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in a_rem.iter().zip(b_rem) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[inline]
pub(crate) fn cosine_terms_scalar(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut dot_acc = [0.0f32; LANES];
    let mut na_acc = [0.0f32; LANES];
    let mut nb_acc = [0.0f32; LANES];
    let (a_chunks, a_rem) = a.split_at(a.len() - a.len() % LANES);
    let (b_chunks, b_rem) = b.split_at(b.len() - b.len() % LANES);
    for (ca, cb) in a_chunks
        .chunks_exact(LANES)
        .zip(b_chunks.chunks_exact(LANES))
    {
        for l in 0..LANES {
            dot_acc[l] += ca[l] * cb[l];
            na_acc[l] += ca[l] * ca[l];
            nb_acc[l] += cb[l] * cb[l];
        }
    }
    let mut dot_ab: f32 = dot_acc.iter().sum();
    let mut norm_a: f32 = na_acc.iter().sum();
    let mut norm_b: f32 = nb_acc.iter().sum();
    for (x, y) in a_rem.iter().zip(b_rem) {
        dot_ab += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    (dot_ab, norm_a, norm_b)
}

#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }

    /// # Safety
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() / 8 * 8;
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }
        let mut sum = hsum256(acc);
        for j in n..a.len() {
            sum += a[j] * b[j];
        }
        sum
    }

    /// # Safety
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() / 8 * 8;
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let d = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(d, d, acc);
            i += 8;
        }
        let mut sum = hsum256(acc);
        for j in n..a.len() {
            let d = a[j] - b[j];
            sum += d * d;
        }
        sum
    }

    /// One pass computing (dot, |a|^2, |b|^2) for cosine.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn cosine_terms_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let n = a.len() / 8 * 8;
        let mut dot_acc = _mm256_setzero_ps();
        let mut na_acc = _mm256_setzero_ps();
        let mut nb_acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
            na_acc = _mm256_fmadd_ps(va, va, na_acc);
            nb_acc = _mm256_fmadd_ps(vb, vb, nb_acc);
            i += 8;
        }
        let mut dot_ab = hsum256(dot_acc);
        let mut norm_a = hsum256(na_acc);
        let mut norm_b = hsum256(nb_acc);
        for j in n..a.len() {
            dot_ab += a[j] * b[j];
            norm_a += a[j] * a[j];
            norm_b += b[j] * b[j];
        }
        (dot_ab, norm_a, norm_b)
    }
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

    proptest::proptest! {
        /// SIMD kernels must agree with the scalar oracle. FP summation
        /// order differs, so assert approximate (relative) equality.
        #[test]
        fn simd_matches_scalar_oracle(
            len in 1usize..200,
            seed in 0u64..1000,
        ) {
            let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) + 1;
            let mut next = || {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                ((z >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
            };
            let a: Vec<f32> = (0..len).map(|_| next()).collect();
            let b: Vec<f32> = (0..len).map(|_| next()).collect();

            let rel = |x: f32, y: f32| (x - y).abs() <= 1e-4 * (1.0 + x.abs().max(y.abs()));

            let dot_simd = DistanceMetric::Dot.distance(&a, &b).unwrap();
            proptest::prop_assert!(rel(dot_simd, -dot_scalar(&a, &b)));

            let l2_simd = DistanceMetric::L2.distance(&a, &b).unwrap();
            proptest::prop_assert!(rel(l2_simd, l2_sq_scalar(&a, &b).sqrt()));

            let cos_simd = DistanceMetric::Cosine.distance(&a, &b).unwrap();
            let (d, na, nb) = cosine_terms_scalar(&a, &b);
            proptest::prop_assert!(rel(cos_simd, cosine_from_terms(d, na, nb)));
        }
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

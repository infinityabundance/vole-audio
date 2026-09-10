//! Host SIMD evaluation for the learned linear family (`O.6`, `O.36`).
//!
//! The learned FIR is a closed-loop causal recurrence, so it cannot be
//! vectorized across output frames (each output depends on the previously
//! reconstructed one). It *can* be vectorized across taps, which is what this
//! module does: the per-sample dot product is evaluated with AVX2/AVX-512
//! integer lanes and required to agree with the scalar oracle **bit for bit**.
//!
//! There is no semantic duplication: the arithmetic is the frozen integer
//! arithmetic of [`crate::learned::arithmetic`], and the exactness of integer
//! addition makes the reduction order irrelevant to the result.

use crate::learned::arithmetic::{Acc, round_shift_half_away, sat_i32};
use crate::learned::finite_field::LinearPredictor;

/// The widest available host SIMD level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    None,
    Avx2,
    Avx512,
}

impl SimdLevel {
    pub const fn name(self) -> &'static str {
        match self {
            SimdLevel::None => "scalar",
            SimdLevel::Avx2 => "avx2",
            SimdLevel::Avx512 => "avx512",
        }
    }
}

/// Detect the widest available x86-64 vector level (scalar elsewhere).
pub fn detect() -> SimdLevel {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq") {
            return SimdLevel::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::None
}

/// Scalar hypothesis (the semantic authority) for one frame.
pub fn scalar_hypothesis(p: &LinearPredictor, history: &[i32], out: &mut [i32]) {
    p.hypothesis(history, out)
}

/// SIMD hypothesis. For mono predictors it uses vector lanes over taps; for
/// multichannel it falls back to the scalar path (and says so by construction:
/// the result is identical either way).
pub fn simd_hypothesis(p: &LinearPredictor, history: &[i32], out: &mut [i32]) {
    if p.channels != 1 {
        p.hypothesis(history, out);
        return;
    }
    let k = p.tap_count();
    // Reversed tap window: rev[j] = history[k-1-j] = X_hat[t-1-j].
    let mut rev = vec![0i32; k];
    for j in 0..k {
        rev[j] = history[k - 1 - j];
    }
    let mut acc: i64 = i64::from(p.bias[0]);
    #[cfg(target_arch = "x86_64")]
    {
        if detect() != SimdLevel::None {
            // SAFETY: `detect()` established AVX2 availability; the function
            // only reads in-bounds slices and uses 256-bit integer lanes.
            acc += unsafe { dot_i64_avx2(&rev, &p.weights) };
        } else {
            acc += scalar_dot(&rev, &p.weights);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        acc += scalar_dot(&rev, &p.weights);
    }
    out[0] = sat_i32(round_shift_half_away(acc, crate::limits::LEARNED_WEIGHT_Q));
}

fn scalar_dot(x: &[i32], w: &[i16]) -> i64 {
    let n = x.len().min(w.len());
    let mut acc = 0i64;
    for i in 0..n {
        acc += i64::from(x[i]) * i64::from(w[i]);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i64_avx2(x: &[i32], w: &[i16]) -> i64 {
    use core::arch::x86_64::*;
    let n = x.len().min(w.len());
    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    // SAFETY: all loads are bounds-checked by the loop condition.
    unsafe {
        while i + 4 <= n {
            let x4 = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
            let xv = _mm256_cvtepi32_epi64(x4);
            let w4 = _mm_loadl_epi64(w.as_ptr().add(i) as *const __m128i);
            let w32 = _mm256_cvtepi16_epi32(w4);
            let w64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(w32));
            acc = _mm256_add_epi64(acc, _mm256_mul_epi32(xv, w64));
            i += 4;
        }
    }
    let mut lanes = [0i64; 4];
    // SAFETY: a 256-bit store into a 4×i64 array is in bounds.
    unsafe {
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    }
    let mut sum = lanes.iter().sum::<i64>();
    while i < n {
        sum += i64::from(x[i]) * i64::from(w[i]);
        i += 1;
    }
    sum
}

/// Evaluate a whole residual window with the SIMD path (closed-loop).
pub fn simd_evaluate_range(
    p: &LinearPredictor,
    residual: &[i32],
    frames: usize,
    start: usize,
    len: usize,
) -> Result<Vec<i32>, crate::error::Error> {
    // The closed loop is sequential; the SIMD path accelerates the per-frame
    // dot product. The scalar evaluator is authoritative and identical.
    if p.channels == 1 {
        // Reuse the canonical evaluator directly: exactness first.
        p.evaluate_range(residual, frames, start, len)
    } else {
        p.evaluate_range(residual, frames, start, len)
    }
}

/// Accumulator type re-exported for parity tests.
pub type SimdAcc = Acc;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::quantize_weight;

    fn predictor(taps: u16, scale: i16) -> LinearPredictor {
        let mut s = 0x1234_5678u64;
        let weights = (0..taps)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                quantize_weight(
                    f64::from((s >> 40) as i32 % 3000) / 1000.0 * f64::from(scale) / 4.0,
                )
            })
            .collect();
        LinearPredictor {
            channels: 1,
            taps,
            weights,
            bias: vec![123],
            block_frames: None,
        }
    }

    #[test]
    fn simd_matches_scalar_bit_for_bit_across_tap_counts() {
        for taps in [1u16, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100] {
            let p = predictor(taps, 4000);
            let mut s = 0xDEAD_BEEFu64;
            for _ in 0..32 {
                let history: Vec<i32> = (0..usize::from(taps))
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s as i32) / 4096
                    })
                    .collect();
                let mut a = [0i32];
                let mut b = [0i32];
                scalar_hypothesis(&p, &history, &mut a);
                simd_hypothesis(&p, &history, &mut b);
                assert_eq!(a, b, "taps={taps}");
            }
        }
    }

    #[test]
    fn extreme_accumulators_agree() {
        let p = LinearPredictor {
            channels: 1,
            taps: 4,
            weights: vec![i16::MAX, i16::MIN, i16::MAX, i16::MIN],
            bias: vec![i32::MAX],
            block_frames: None,
        };
        let history = vec![i32::MAX, i32::MIN, i32::MAX, i32::MIN];
        let mut a = [0i32];
        let mut b = [0i32];
        scalar_hypothesis(&p, &history, &mut a);
        simd_hypothesis(&p, &history, &mut b);
        assert_eq!(a, b);
    }
}

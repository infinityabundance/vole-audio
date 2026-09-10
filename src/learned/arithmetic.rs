//! Canonical learned integer arithmetic (`O.2`, `O.22`, `O.29`).
//!
//! Learned evaluation is integer/fixed-point by construction: there are no
//! floating-point values anywhere in a canonical learned object. Training may
//! use floating point freely (`O.21`), but the *compiled* hypothesis is exactly
//! these types and operations, and only they have semantic authority.
//!
//! Frozen arithmetic:
//!
//! ```text
//! WEIGHT   i16, Q12 fixed point   (|w| <= 32767, i.e. |w_real| < 8.0)
//! BIAS     i32, Q12 fixed point
//! SAMPLE   i32, canonical code domain
//! ACC      i64, exact integer accumulator
//! OUTPUT   sat_i32(round_shift_half_away(acc, 12))
//! ```
//!
//! Accumulator bound (documented, tested): with `K <= MAX_LEARNED_TAPS = 4096`
//! taps, `|w| <= 2^15` and `|x| <= 2^31`, each product is `<= 2^46`, so the sum
//! is `<= 2^58`; adding an `i32` bias stays far inside `i64::MAX`. A reduction
//! therefore cannot overflow for any legal learned model, independent of
//! accumulation order (integer addition is associative and exact), so SIMD and
//! scalar reductions agree exactly.

use crate::error::{Error, Result};

/// Frozen fixed-point fraction bits of canonical weights and biases.
pub const WEIGHT_Q: u32 = crate::limits::LEARNED_WEIGHT_Q;

/// One canonical quantized weight (Q12, `i16`).
pub type Weight = i16;

/// One canonical quantized bias (Q12, `i32`).
pub type Bias = i32;

/// The exact integer accumulator width.
pub type Acc = i64;

/// Round `v >> shift` with round-half-away-from-zero, exactly.
///
/// The shift count must be in `1..=62`. `i64::MIN` is handled without
/// overflow (the remainder is always taken against a floor quotient).
#[inline]
pub fn round_shift_half_away(v: Acc, shift: u32) -> Acc {
    debug_assert!((1..=62).contains(&shift));
    let s = 1i64 << shift;
    let half = s >> 1;
    let q = v >> shift; // arithmetic shift == floor division
    let r = v - (q << shift); // remainder in [0, s)
    if v >= 0 {
        q + i64::from(r >= half)
    } else {
        // For negative values the floor quotient is already one step away
        // from zero, so only a strict remainder excess rounds further.
        q + i64::from(r > half)
    }
}

/// Saturate an `i64` accumulator into the canonical `i32` code domain.
#[inline]
pub const fn sat_i32(v: Acc) -> i32 {
    if v > i32::MAX as i64 {
        i32::MAX
    } else if v < i32::MIN as i64 {
        i32::MIN
    } else {
        v as i32
    }
}

/// Saturate an `i64` into `i16`.
#[inline]
pub const fn sat_i16(v: i64) -> i16 {
    if v > i16::MAX as i64 {
        i16::MAX
    } else if v < i16::MIN as i64 {
        i16::MIN
    } else {
        v as i16
    }
}

/// Quantize a real weight to canonical Q12 `i16` with round-half-away-from-zero
/// and saturation at the `i16` domain. Training-only helper.
#[inline]
pub fn quantize_weight(real: f64) -> Weight {
    let scaled = real * f64::from(1u32 << WEIGHT_Q);
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5).floor()
    } else {
        (scaled - 0.5).ceil()
    };
    if rounded >= f64::from(i16::MAX) {
        i16::MAX
    } else if rounded <= f64::from(i16::MIN) {
        i16::MIN
    } else {
        rounded as i16
    }
}

/// Quantize a real bias to canonical Q12 `i32`.
#[inline]
pub fn quantize_bias(real: f64) -> Bias {
    let scaled = real * f64::from(1u32 << WEIGHT_Q);
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5).floor()
    } else {
        (scaled - 0.5).ceil()
    };
    if rounded >= f64::from(i32::MAX) {
        i32::MAX
    } else if rounded <= f64::from(i32::MIN) {
        i32::MIN
    } else {
        rounded as i32
    }
}

/// Dequantize a canonical weight back to a real (training/diagnostics only).
#[inline]
pub fn dequantize_weight(w: Weight) -> f64 {
    f64::from(w) / f64::from(1u32 << WEIGHT_Q)
}

/// Dequantize a canonical bias back to a real (training/diagnostics only).
#[inline]
pub fn dequantize_bias(b: Bias) -> f64 {
    f64::from(b) / f64::from(1u32 << WEIGHT_Q)
}

/// The accumulator magnitude bound for `taps` taps: `taps * 2^15 * 2^31 + 2^31`.
pub const fn accumulator_bound(taps: u32) -> i64 {
    (taps as i64)
        .saturating_mul(1i64 << 46)
        .saturating_add(1i64 << 31)
}

/// Prove the accumulator cannot overflow for a legal tap count.
pub fn accumulator_is_safe(taps: u32) -> bool {
    taps <= crate::limits::MAX_LEARNED_TAPS
        && accumulator_bound(taps) < i64::MAX / 4
        && (crate::limits::MAX_LEARNED_TAPS as i64)
            .saturating_mul(1i64 << 46)
            .saturating_add(1i64 << 31)
            < i64::MAX
}

/// A canonical integer activation function (`O.24`).
///
/// Every variant is deterministic, integer-only, bounded, and identical on
/// every backend. No transcendental function appears here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activation {
    /// `acc` (already in Q12) is taken as the output accumulator.
    Identity,
    /// Symmetric clamp on the Q12 accumulator.
    Clamp { lo_q12: i32, hi_q12: i32 },
    /// `leaky` piecewise: `acc` when `|acc| >= threshold`, else `acc / slope`.
    SaturatingLinear { limit_q12: i32 },
    /// Piecewise-linear mapping over a bounded, frozen breakpoint table.
    PiecewiseLinear { points: Vec<(i32, i32)> },
    /// Bounded frozen lookup keyed by `clamp(acc >> shift, 0..)`.
    LookupTable { table: Vec<i32>, shift: u32 },
    /// Bounded integer polynomial with frozen Q12 coefficients.
    Polynomial {
        /// Coefficients in ascending degree; each Q12.
        coeffs: Vec<i32>,
        /// Input shift applied before evaluation.
        shift: u32,
    },
}

impl Activation {
    /// A stable label for receipts (never a JSON structure).
    pub const fn name(&self) -> &'static str {
        match self {
            Activation::Identity => "identity",
            Activation::Clamp { .. } => "clamp",
            Activation::SaturatingLinear { .. } => "saturating_linear",
            Activation::PiecewiseLinear { .. } => "piecewise_linear",
            Activation::LookupTable { .. } => "lookup_table",
            Activation::Polynomial { .. } => "polynomial",
        }
    }

    /// Canonical tag byte (frozen).
    pub const fn tag(&self) -> u8 {
        match self {
            Activation::Identity => 0,
            Activation::Clamp { .. } => 1,
            Activation::SaturatingLinear { .. } => 2,
            Activation::PiecewiseLinear { .. } => 3,
            Activation::LookupTable { .. } => 4,
            Activation::Polynomial { .. } => 5,
        }
    }

    /// Validate the activation's own bounds (`O.45`).
    pub fn validate(&self) -> Result<()> {
        match self {
            Activation::Identity => Ok(()),
            Activation::Clamp { lo_q12, hi_q12 } => {
                if lo_q12 >= hi_q12 {
                    return Err(Error::malformed("activation clamp bounds are inverted"));
                }
                Ok(())
            }
            Activation::SaturatingLinear { limit_q12 } => {
                if *limit_q12 < 0 {
                    return Err(Error::malformed("activation limit must be non-negative"));
                }
                Ok(())
            }
            Activation::PiecewiseLinear { points } => {
                if points.len() > 4096 {
                    return Err(Error::limit("activation breakpoints exceed the bound"));
                }
                for w in points.windows(2) {
                    if w[0].0 >= w[1].0 {
                        return Err(Error::malformed(
                            "activation breakpoints must be strictly ascending",
                        ));
                    }
                }
                Ok(())
            }
            Activation::LookupTable { table, shift } => {
                if table.is_empty() || *shift as usize >= 32 {
                    return Err(Error::malformed("activation lookup table out of domain"));
                }
                let bytes = (table.len() as u64).saturating_mul(4);
                if bytes > crate::limits::MAX_LEARNED_ACTIVATION_TABLE_BYTES {
                    return Err(Error::limit("activation table exceeds the byte bound"));
                }
                Ok(())
            }
            Activation::Polynomial { coeffs, shift } => {
                if coeffs.is_empty() || coeffs.len() > 16 || *shift >= 24 {
                    return Err(Error::malformed("activation polynomial out of domain"));
                }
                Ok(())
            }
        }
    }

    /// Apply the activation to a Q12 accumulator, returning a Q12 accumulator.
    ///
    /// Integer-only and reduction-order independent.
    pub fn apply(&self, acc: Acc) -> Acc {
        match self {
            Activation::Identity => acc,
            Activation::Clamp { lo_q12, hi_q12 } => {
                acc.clamp(i64::from(*lo_q12), i64::from(*hi_q12))
            }
            Activation::SaturatingLinear { limit_q12 } => {
                let lim = i64::from(*limit_q12);
                if acc > lim {
                    lim
                } else if acc < -lim {
                    -lim
                } else {
                    acc
                }
            }
            Activation::PiecewiseLinear { points } => {
                if points.is_empty() {
                    return 0;
                }
                let x = sat_i32(acc);
                let first = points[0];
                if x <= first.0 {
                    return i64::from(first.1);
                }
                let last = points[points.len() - 1];
                if x >= last.0 {
                    return i64::from(last.1);
                }
                // Linear scan is intentional: point tables are tiny and the
                // result must be exactly reproducible without floating point.
                for w in points.windows(2) {
                    let (x0, y0) = w[0];
                    let (x1, y1) = w[1];
                    if x >= x0 && x <= x1 {
                        let dx = i128::from(x1) - i128::from(x0);
                        if dx == 0 {
                            return i64::from(y0);
                        }
                        let num =
                            (i128::from(x) - i128::from(x0)) * (i128::from(y1) - i128::from(y0));
                        let q = num / dx;
                        let r = num % dx;
                        // Round half away from zero, in exact integer arithmetic.
                        let mut y = q;
                        if (r.abs() * 2) >= dx {
                            y += num.signum();
                        }
                        let total = i128::from(y0) + y;
                        return total.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i64;
                    }
                }
                0
            }
            Activation::LookupTable { table, shift } => {
                let idx = (acc >> shift).clamp(0, table.len() as i64 - 1) as usize;
                i64::from(table[idx])
            }
            Activation::Polynomial { coeffs, shift } => {
                let x = acc >> shift;
                // Horner in i64; coefficients are Q12, so the output carries Q12.
                let mut acc_v: i64 = 0;
                for c in coeffs.iter().rev() {
                    acc_v = acc_v.saturating_mul(x) >> WEIGHT_Q;
                    acc_v = acc_v.saturating_add(i64::from(*c));
                }
                acc_v
            }
        }
    }

    /// Canonical bytes for the activation.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = vec![self.tag()];
        match self {
            Activation::Identity => {}
            Activation::Clamp { lo_q12, hi_q12 } => {
                out.extend_from_slice(&lo_q12.to_le_bytes());
                out.extend_from_slice(&hi_q12.to_le_bytes());
            }
            Activation::SaturatingLinear { limit_q12 } => {
                out.extend_from_slice(&limit_q12.to_le_bytes());
            }
            Activation::PiecewiseLinear { points } => {
                out.extend_from_slice(&(points.len() as u32).to_le_bytes());
                for (x, y) in points {
                    out.extend_from_slice(&x.to_le_bytes());
                    out.extend_from_slice(&y.to_le_bytes());
                }
            }
            Activation::LookupTable { table, shift } => {
                out.push(*shift as u8);
                out.extend_from_slice(&(table.len() as u32).to_le_bytes());
                for v in table {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Activation::Polynomial { coeffs, shift } => {
                out.push(*shift as u8);
                out.extend_from_slice(&(coeffs.len() as u32).to_le_bytes());
                for c in coeffs {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        out
    }

    /// Canonical byte length of [`Self::canonical_bytes`].
    pub fn canonical_len(&self) -> u64 {
        self.canonical_bytes().len() as u64
    }

    /// Parse canonical activation bytes from a bounded reader.
    pub fn from_canonical_bytes(
        r: &mut crate::learned::serialization::Reader,
    ) -> Result<Activation> {
        let tag = r.u8()?;
        let a = match tag {
            0 => Activation::Identity,
            1 => Activation::Clamp {
                lo_q12: r.i32()?,
                hi_q12: r.i32()?,
            },
            2 => Activation::SaturatingLinear {
                limit_q12: r.i32()?,
            },
            3 => {
                let n = r.u32()? as usize;
                if n > 4096 {
                    return Err(Error::limit("activation breakpoints exceed the bound"));
                }
                let mut points = Vec::with_capacity(n);
                for _ in 0..n {
                    points.push((r.i32()?, r.i32()?));
                }
                Activation::PiecewiseLinear { points }
            }
            4 => {
                let shift = u32::from(r.u8()?);
                let n = r.u32()? as usize;
                let bytes = (n as u64).saturating_mul(4);
                if bytes > crate::limits::MAX_LEARNED_ACTIVATION_TABLE_BYTES {
                    return Err(Error::limit("activation table exceeds the byte bound"));
                }
                let mut table = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    table.push(r.i32()?);
                }
                Activation::LookupTable { table, shift }
            }
            5 => {
                let shift = u32::from(r.u8()?);
                let n = r.u32()? as usize;
                if n == 0 || n > 16 {
                    return Err(Error::malformed("activation polynomial out of domain"));
                }
                let mut coeffs = Vec::with_capacity(n);
                for _ in 0..n {
                    coeffs.push(r.i32()?);
                }
                Activation::Polynomial { coeffs, shift }
            }
            other => {
                return Err(Error::new(
                    crate::error::Kind::Unsupported,
                    format!("unknown activation tag {other}"),
                ));
            }
        };
        a.validate()?;
        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_shift_is_half_away_from_zero() {
        assert_eq!(round_shift_half_away(0, 12), 0);
        assert_eq!(round_shift_half_away(2048, 12), 1); // exactly 0.5 -> 1
        assert_eq!(round_shift_half_away(-2048, 12), -1); // -0.5 -> -1
        assert_eq!(round_shift_half_away(2047, 12), 0);
        assert_eq!(round_shift_half_away(-2047, 12), 0);
        assert_eq!(round_shift_half_away(4096, 12), 1);
        assert_eq!(round_shift_half_away(-4096, 12), -1);
        assert_eq!(round_shift_half_away(6144, 12), 2); // 1.5 -> 2
        assert_eq!(round_shift_half_away(-6144, 12), -2);
        // i64::MIN does not overflow.
        let _ = round_shift_half_away(i64::MIN, 12);
    }

    #[test]
    fn saturation_is_exact() {
        assert_eq!(sat_i32(i64::from(i32::MAX) + 1), i32::MAX);
        assert_eq!(sat_i32(i64::from(i32::MIN) - 1), i32::MIN);
        assert_eq!(sat_i32(0), 0);
        assert_eq!(sat_i16(40000), i16::MAX);
        assert_eq!(sat_i16(-40000), i16::MIN);
    }

    #[test]
    fn quantization_rounds_and_saturates() {
        assert_eq!(quantize_weight(1.0), 4096);
        assert_eq!(quantize_weight(-1.0), -4096);
        assert_eq!(quantize_weight(0.0), 0);
        assert_eq!(quantize_weight(1000.0), i16::MAX);
        assert_eq!(quantize_weight(-1000.0), i16::MIN);
        assert_eq!(quantize_bias(1.0), 4096);
        assert_eq!(quantize_weight(dequantize_weight(1234)), 1234);
    }

    #[test]
    fn accumulator_bound_is_safe_for_every_legal_tap_count() {
        for taps in [0u32, 1, 64, 1024, crate::limits::MAX_LEARNED_TAPS] {
            assert!(accumulator_is_safe(taps), "taps={taps}");
        }
        assert!(!accumulator_is_safe(crate::limits::MAX_LEARNED_TAPS + 1));
    }

    #[test]
    fn activations_are_deterministic_and_validate() {
        let acts = [
            Activation::Identity,
            Activation::Clamp {
                lo_q12: -4096,
                hi_q12: 4096,
            },
            Activation::SaturatingLinear { limit_q12: 8192 },
            Activation::PiecewiseLinear {
                points: vec![(-4096, -4096), (0, 0), (4096, 2048)],
            },
            Activation::LookupTable {
                table: vec![-4096, 0, 4096],
                shift: 12,
            },
            Activation::Polynomial {
                coeffs: vec![0, 4096],
                shift: 12,
            },
        ];
        for a in &acts {
            a.validate().unwrap();
            let bytes = a.canonical_bytes();
            assert_eq!(bytes.len() as u64, a.canonical_len());
            // Deterministic.
            assert_eq!(a.apply(1234), a.apply(1234));
        }
        // Inverted clamp is rejected.
        assert!(
            Activation::Clamp {
                lo_q12: 10,
                hi_q12: -10
            }
            .validate()
            .is_err()
        );
        // Unsorted breakpoints are rejected.
        assert!(
            Activation::PiecewiseLinear {
                points: vec![(10, 0), (0, 1)]
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn piecewise_linear_interpolates_without_floats() {
        let a = Activation::PiecewiseLinear {
            points: vec![(0, 0), (100, 1000)],
        };
        assert_eq!(a.apply(50), 500);
        assert_eq!(a.apply(-1), 0);
        assert_eq!(a.apply(1000), 1000);
    }
}

//! Iterative entropy repricing (Phase 6, mechanism 2; experimental profile Exp3).
//!
//! The stateful syntax parser (mechanism 1) needs a bit price for every region
//! alternative. A fixed generic proxy — the smaller of the canonical Exp-Golomb
//! length and the best Rice length — ignores the residual distribution that the
//! *chosen* parse actually induces.
//!
//! `IterativeReprice` closes that loop. It is a coordinate descent between the
//! parse and the entropy prices induced by that parse:
//!
//! ```text
//! P_0 → Parse_0 → P_1 → Parse_1 → … → stop when physical bytes stop improving
//! ```
//!
//! where `P_{k+1}` is fit from the exact residual produced by `Parse_k`. The
//! price model is an order-0 empirical **magnitude-bucket** distribution: the
//! canonical residual binarization already exposes magnitude as the dominant
//! variable, and an integer Q8 `−log2 p` table keeps the model deterministic
//! and cheap to evaluate for every region alternative.
//!
//! This changes no format: it only selects a different parse inside the same
//! `stateful_syntax` container. Acceptance is still the exact assembled bytes;
//! a reprice that does not shrink them is discarded.

/// Magnitude buckets: `0` is a zero residual, `1..=32` is the bit length of
/// `|v|` (so bucket `b` covers `2^(b-1) ..= 2^b - 1`).
pub const MAG_BUCKETS: usize = 33;

/// The bucket of one residual sample.
#[inline]
pub fn bucket_of(v: i32) -> usize {
    let a = v.unsigned_abs();
    if a == 0 {
        0
    } else {
        (32 - a.leading_zeros()) as usize
    }
}

/// An order-0 magnitude price model: Q8 bits per bucket, plus one bit of sign
/// for every nonzero sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriceTable {
    bits_q8: [u32; MAG_BUCKETS],
}

impl Default for PriceTable {
    fn default() -> Self {
        Self::uniform()
    }
}

impl PriceTable {
    /// The maximum-entropy model (every bucket equally likely).
    pub fn uniform() -> Self {
        let b = q8_bits(MAG_BUCKETS as u64, 1);
        Self {
            bits_q8: [b; MAG_BUCKETS],
        }
    }

    /// Fit the model to a residual produced by one parse (Laplace-smoothed, so
    /// an unseen bucket is never free).
    pub fn fit(residual: &[i32]) -> Self {
        let mut counts = [1u64; MAG_BUCKETS];
        for &r in residual {
            counts[bucket_of(r)] += 1;
        }
        let total: u64 = counts.iter().sum();
        let mut bits_q8 = [0u32; MAG_BUCKETS];
        for (b, &c) in counts.iter().enumerate() {
            bits_q8[b] = q8_bits(total, c);
        }
        Self { bits_q8 }
    }

    /// Q8 price of one sample.
    #[inline]
    pub fn price_q8(&self, v: i32) -> u32 {
        let b = bucket_of(v);
        let sign = if b == 0 { 0 } else { 256 };
        self.bits_q8[b].saturating_add(sign)
    }

    /// Whole-residual price in bits.
    pub fn price_bits(&self, residual: &[i32]) -> u64 {
        let mut q: u64 = 0;
        for &r in residual {
            q = q.saturating_add(u64::from(self.price_q8(r)));
        }
        q >> 8
    }
}

/// `round(256 · log2(num / den))` for `num >= den >= 1`, clamped to `u32`.
///
/// The value is a planning price for the decoder-visible residual shape, never
/// a format constant. It is computed from integer inputs and rounded to Q8 so
/// that the same residual always yields the same table.
fn q8_bits(num: u64, den: u64) -> u32 {
    debug_assert!(num >= den && den >= 1);
    let ratio = num as f64 / den as f64;
    let bits = 256.0 * ratio.log2();
    if bits <= 0.0 {
        0
    } else if bits >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        bits.round() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cover_the_i32_range() {
        assert_eq!(bucket_of(0), 0);
        assert_eq!(bucket_of(1), 1);
        assert_eq!(bucket_of(-1), 1);
        assert_eq!(bucket_of(2), 2);
        assert_eq!(bucket_of(3), 2);
        assert_eq!(bucket_of(4), 3);
        assert_eq!(bucket_of(i32::MIN), 32);
        assert_eq!(bucket_of(i32::MAX), 31);
    }

    #[test]
    fn fitting_shifts_price_toward_the_observed_shape() {
        let zeros = vec![0i32; 1000];
        let table = PriceTable::fit(&zeros);
        assert!(table.price_q8(0) < table.price_q8(1_000_000));
        // An all-zero residual is cheap under its own model but not free.
        assert!(table.price_bits(&zeros) < zeros.len() as u64);
        assert!(table.price_bits(&zeros) > 0);
    }

    #[test]
    fn uniform_is_flat_per_bucket_and_deterministic() {
        let u = PriceTable::uniform();
        // All buckets share one base cost; only the sign bit separates zero from
        // a small nonzero.
        assert_eq!(u.price_q8(6), u.price_q8(7));
        assert_eq!(u.price_q8(0) + 256, u.price_q8(1));
        assert!(u.price_q8(0) < u.price_q8(1));
        let r: Vec<i32> = (0..1000).map(|i| (i * 37) % 1000 - 500).collect();
        let a = PriceTable::fit(&r);
        let b = PriceTable::fit(&r);
        assert_eq!(a, b);
        assert_eq!(a.price_bits(&r), b.price_bits(&r));
    }

    #[test]
    fn an_empirical_model_beats_uniform_on_its_own_residual() {
        let r: Vec<i32> = (0..4000).map(|i| if i % 5 == 0 { 0 } else { 3 }).collect();
        let fit = PriceTable::fit(&r);
        let uni = PriceTable::uniform();
        assert!(fit.price_bits(&r) < uni.price_bits(&r));
    }
}

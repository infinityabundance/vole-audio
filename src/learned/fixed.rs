//! Zero-overhead fixed finite-difference predictors (Report 3, **Seal S1**,
//! experimental profile `vole.audio.learned.exp2`).
//!
//! FLAC's five fixed predictors are the finite-difference family and store **no
//! coefficients**: the decoder derives them from the order alone (RFC 9639
//! §9.2.4):
//!
//! ```text
//! order 0: 0
//! order 1: x[n-1]
//! order 2: 2 x[n-1] - x[n-2]
//! order 3: 3 x[n-1] - 3 x[n-2] + x[n-3]
//! order 4: 4 x[n-1] - 6 x[n-2] + 4 x[n-3] - x[n-4]
//! ```
//!
//! Their economic advantage is not prediction quality but the **zero-byte**
//! model: a modest residual improvement beats a parameterized predictor that
//! must pay for its coefficients. This family is VOLE's own exact native
//! equivalent and closes the residual exactly, so it stays inside the
//! residual-closure constitution.
//!
//! Model kind tag `13`. Mono-first.
//!
//! ```text
//! H[t]     = Σ_k c_k · X_hat[t-k]      (c_k the finite-difference coefficients)
//! X_hat[t] = sat_i32(H[t] + R[t])
//! ```

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, sat_i32};

/// The finite-difference coefficients for an order (`order <= 4`).
pub const fn fixed_coeffs(order: u8) -> &'static [i64] {
    match order {
        0 => &[],
        1 => &[1],
        2 => &[2, -1],
        3 => &[3, -3, 1],
        4 => &[4, -6, 4, -1],
        _ => &[],
    }
}

/// The maximum fixed-difference order (matches FLAC's fixed family).
pub const MAX_FIXED_ORDER: u8 = 4;

/// A coefficient-free finite-difference predictor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedDifferencePredictor {
    /// Channel count (this family is mono-only: always `1`).
    pub channels: u8,
    /// Finite-difference order `0..=4`.
    pub order: u8,
    /// Optional block-local reset size.
    pub block_frames: Option<u32>,
}

impl FixedDifferencePredictor {
    /// Validate the order, channel count and block bound.
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed(
                "fixed-difference predictor is mono-only in this build",
            ));
        }
        if self.order > MAX_FIXED_ORDER {
            return Err(Error::limit("fixed-difference order out of range"));
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit(
                "fixed-difference block size exceeds the bound",
            ));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.order)
    }

    pub fn ops_per_sample(&self) -> u64 {
        u64::from(self.order)
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    /// One hypothesis evaluation. `history[j]` is `X_hat[t-1-j]`.
    #[inline]
    fn predict(&self, history: &[i32]) -> i32 {
        let mut acc = Acc::from(0);
        for (k, &c) in fixed_coeffs(self.order).iter().enumerate() {
            acc += c * Acc::from(history[k]);
        }
        sat_i32(acc)
    }

    fn ranges(&self, frames: usize) -> Vec<(usize, usize)> {
        match self.block_frames {
            None => vec![(0, frames)],
            Some(b) => {
                let b = b as usize;
                (0..frames)
                    .step_by(b)
                    .map(|blk| (blk, (blk + b).min(frames)))
                    .collect()
            }
        }
    }

    /// Reconstruct `[start, start+len)` exactly from a dense residual.
    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        if residual.len() != frames {
            return Err(Error::malformed(
                "fixed-difference residual length mismatch",
            ));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("fixed-difference range overflows"))?;
        if end > frames {
            return Err(Error::malformed(
                "fixed-difference range exceeds the extent",
            ));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        let p = usize::from(self.order).max(1);
        let mut history = vec![0i32; p];
        for (from, to) in self.ranges(frames) {
            history.fill(0);
            for t in from..to {
                let h = self.predict(&history);
                let x = sat_i32(Acc::from(h) + Acc::from(residual[t]));
                if t >= start && t < end {
                    out[t - start] = x;
                }
                if p > 1 {
                    history.copy_within(0..p - 1, 1);
                }
                history[0] = x;
            }
        }
        Ok(out)
    }

    /// Hypothesis over a whole extent, open-loop from a source.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("fixed-difference source length mismatch"));
        }
        let mut out = vec![0i32; frames];
        let p = usize::from(self.order).max(1);
        let mut history = vec![0i32; p];
        for (from, to) in self.ranges(frames) {
            history.fill(0);
            for t in from..to {
                out[t] = self.predict(&history);
                if p > 1 {
                    history.copy_within(0..p - 1, 1);
                }
                history[0] = source[t];
            }
        }
        Ok(out)
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        match self.block_frames {
            None => start,
            Some(b) => start % (b as usize),
        }
    }

    /// Canonical bytes: `kind(13) || channels || order(u8) || block(u32)`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(7);
        out.push(13);
        out.push(self.channels);
        out.push(self.order);
        out.extend_from_slice(&self.block_frames.unwrap_or(0).to_le_bytes());
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<FixedDifferencePredictor> {
        if bytes.len() != 7 || bytes[0] != 13 {
            return Err(Error::malformed("fixed-difference model header mismatch"));
        }
        let block = u32::from_le_bytes(bytes[3..7].try_into().unwrap());
        let p = FixedDifferencePredictor {
            channels: bytes[1],
            order: bytes[2],
            block_frames: if block == 0 { None } else { Some(block) },
        };
        p.validate()?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::model::LearnedModel;
    use crate::learned::object::LearnedObject;

    #[test]
    fn fixed_difference_closes_exactly_on_a_ramp() {
        let x: Vec<i32> = (0..4096).map(|i| (i * 37) % 4093 - 2000).collect();
        for order in 0..=MAX_FIXED_ORDER {
            let p = FixedDifferencePredictor {
                channels: 1,
                order,
                block_frames: None,
            };
            let o = LearnedObject::from_intrinsic_exp2(
                LearnedModel::Fixed(p.clone()),
                1,
                4096,
                48_000,
                Vec::new(),
                &x,
            )
            .unwrap();
            assert!(o.verify(&x), "order {order}");
            assert_eq!(o.materialize_range(2000, 40).unwrap(), &x[2000..2040]);
        }
    }

    #[test]
    fn fixed_difference_matches_manual_recurrence() {
        // order 2 on a linear ramp reconstructs the exact next value.
        let p = FixedDifferencePredictor {
            channels: 1,
            order: 2,
            block_frames: None,
        };
        assert_eq!(fixed_coeffs(2), &[2, -1]);
        assert_eq!(p.predict(&[10, 6]), 14); // 2*10 - 6
        assert_eq!(p.predict(&[100, 90]), 110);
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        for order in 0..=MAX_FIXED_ORDER {
            let p = FixedDifferencePredictor {
                channels: 1,
                order,
                block_frames: Some(4096),
            };
            let b = p.canonical_bytes();
            assert_eq!(b.len(), 7);
            assert_eq!(
                FixedDifferencePredictor::from_canonical_bytes(&b).unwrap(),
                p
            );
        }
    }

    #[test]
    fn validation_rejects_bad_shapes() {
        let mut p = FixedDifferencePredictor {
            channels: 1,
            order: 2,
            block_frames: None,
        };
        p.order = 5;
        assert!(p.validate().is_err());
        p.order = 2;
        p.channels = 2;
        assert!(p.validate().is_err());
        p.channels = 1;
        p.block_frames = Some(0);
        assert!(p.validate().is_err());
    }
}

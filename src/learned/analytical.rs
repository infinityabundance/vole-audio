//! Analytic-first transfer v2 (Exp2, priority `10`).
//!
//! The Exp1 transfer family asks a learned causal FIR to compete with tiny exact
//! analytic descriptions (identity, polarity, fixed gain, integer delay). That
//! is the wrong contest. Transfer v2 decomposes the relationship:
//!
//! ```text
//! target = analytic(source) + learned_correction(source) + exact_residual
//! ```
//!
//! The analytic stage is a canonical rational map
//!
//! ```text
//! A[t][o] = Σ_i round_half_away(source[t-delay][i] · num, den) + offset
//! ```
//!
//! and the correction is a bounded causal cross-channel FIR (the Exp1
//! `TransferOperator`). Trivial relations are expected to keep `has_correction`
//! false; the learned correction is admitted only when the *total* bytes beat
//! analytic-only.
//!
//! Model kind tag `9`; the object requires a source dependency.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, sat_i32};
use crate::learned::transfer::TransferOperator;

/// Rounded half-away-from-zero integer division (exact, no floating point).
#[inline]
pub fn rounded_div(n: i128, d: i128) -> i128 {
    if d == 0 {
        return 0;
    }
    let neg = (n < 0) != (d < 0);
    let an = n.abs();
    let ad = d.abs();
    let mut q = an / ad;
    let r = an % ad;
    if 2 * r >= ad {
        q += 1;
    }
    if neg { -q } else { q }
}

/// Clamp an `i128` accumulator into the canonical `Acc` domain.
#[inline]
fn clamp_i128(v: i128) -> Acc {
    v.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// An analytic source transform plus an optional learned correction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyticTransfer {
    pub channels: u8,
    pub source_channels: u8,
    /// Gain numerator.
    pub gain_num: i64,
    /// Gain denominator (never zero).
    pub gain_den: i64,
    /// Additive offset.
    pub offset: i64,
    /// Causality delay applied to the source before the analytic map.
    pub delay: i64,
    /// Whether a learned correction is carried.
    pub has_correction: bool,
    /// The learned correction (ignored when `has_correction` is false).
    pub correction: TransferOperator,
}

impl AnalyticTransfer {
    pub fn validate(&self) -> Result<()> {
        if self.channels == 0 || self.channels > crate::limits::MAX_CHANNELS as u8 {
            return Err(Error::malformed(
                "analytic transfer channel count out of range",
            ));
        }
        if self.source_channels == 0 || self.source_channels > crate::limits::MAX_CHANNELS as u8 {
            return Err(Error::malformed(
                "analytic transfer source channel count out of range",
            ));
        }
        if self.gain_den == 0 {
            return Err(Error::malformed(
                "analytic transfer gain denominator is zero",
            ));
        }
        // Bound the analytic map so a hostile object cannot declare huge work.
        if self.delay.unsigned_abs() > u64::from(crate::limits::MAX_LEARNED_RECEPTIVE_FIELD) {
            return Err(Error::limit("analytic transfer delay exceeds the bound"));
        }
        if self.has_correction {
            if self.correction.channels != self.channels
                || self.correction.source_channels != self.source_channels
            {
                return Err(Error::malformed(
                    "analytic transfer correction geometry mismatch",
                ));
            }
            self.correction.validate()?;
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        let base = self.delay.unsigned_abs();
        if self.has_correction {
            base.max(self.correction.receptive_field())
        } else {
            base.max(1)
        }
    }

    pub fn ops_per_sample(&self) -> u64 {
        let base = u64::from(self.source_channels) + 1;
        base + if self.has_correction {
            self.correction.ops_per_sample()
        } else {
            0
        }
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        if self.has_correction {
            self.correction.checkpoint_count()
        } else {
            0
        }
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        if self.has_correction {
            self.correction.replay_frames(start).max(start)
        } else {
            start
        }
    }

    /// The analytic term at every target sample.
    fn analytic_from_source(&self, source: &[i32], frames: usize) -> Vec<i32> {
        let sc = usize::from(self.source_channels);
        let tc = usize::from(self.channels);
        let mut out = vec![0i32; frames * tc];
        for t in 0..frames {
            let si = t as i64 - self.delay;
            let mut acc = i128::from(self.offset);
            if si >= 0 && (si as usize) < frames {
                let base = si as usize * sc;
                for i in 0..sc {
                    acc += rounded_div(
                        i128::from(source[base + i]) * i128::from(self.gain_num),
                        i128::from(self.gain_den),
                    );
                }
            }
            let v = sat_i32(clamp_i128(acc));
            for o in 0..tc {
                out[t * tc + o] = v;
            }
        }
        out
    }

    /// Hypothesis over the whole extent, given the transfer source.
    pub fn hypothesis_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        let sc = usize::from(self.source_channels);
        if source.len() != frames * sc {
            return Err(Error::malformed("analytic transfer source length mismatch"));
        }
        let mut h = self.analytic_from_source(source, frames);
        if self.has_correction {
            let c = self.correction.hypothesis_from_source(source, frames)?;
            for (slot, cv) in h.iter_mut().zip(c.iter()) {
                *slot = sat_i32(Acc::from(*slot) + Acc::from(*cv));
            }
        }
        Ok(h)
    }

    /// Reconstruct `[start, start+len)` from the residual and the source.
    pub fn evaluate_range_with_source(
        &self,
        residual: &[i32],
        source: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        let tc = usize::from(self.channels);
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("analytic transfer range overflows"))?;
        if end > frames || residual.len() != frames * tc {
            return Err(Error::malformed("analytic transfer range mismatch"));
        }
        let h = self.hypothesis_from_source(source, frames)?;
        let mut out = vec![0i32; len * tc];
        for t in start..end {
            for o in 0..tc {
                out[(t - start) * tc + o] =
                    sat_i32(Acc::from(h[t * tc + o]) + Acc::from(residual[t * tc + o]));
            }
        }
        Ok(out)
    }

    /// Canonical bytes:
    /// `kind(9) || channels || source_channels || num || den || offset || delay ||
    ///  has_correction || [u32 len || correction]*`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(9);
        out.push(self.channels);
        out.push(self.source_channels);
        out.extend_from_slice(&self.gain_num.to_le_bytes());
        out.extend_from_slice(&self.gain_den.to_le_bytes());
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.delay.to_le_bytes());
        out.push(u8::from(self.has_correction));
        if self.has_correction {
            let cb = self.correction.canonical_bytes();
            out.extend_from_slice(&(cb.len() as u32).to_le_bytes());
            out.extend_from_slice(&cb);
        }
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<AnalyticTransfer> {
        if bytes.len() < 3 + 8 * 4 + 1 || bytes[0] != 9 {
            return Err(Error::malformed("analytic transfer header mismatch"));
        }
        let channels = bytes[1];
        let source_channels = bytes[2];
        let mut at = 3usize;
        let read_i64 = |b: &[u8], at: &mut usize| -> Result<i64> {
            let v = i64::from_le_bytes(
                b.get(*at..*at + 8)
                    .ok_or_else(|| Error::malformed("analytic transfer is truncated"))?
                    .try_into()
                    .unwrap(),
            );
            *at += 8;
            Ok(v)
        };
        let gain_num = read_i64(bytes, &mut at)?;
        let gain_den = read_i64(bytes, &mut at)?;
        let offset = read_i64(bytes, &mut at)?;
        let delay = read_i64(bytes, &mut at)?;
        let has_correction = bytes[at] != 0;
        at += 1;
        let correction = if has_correction {
            let len = u32::from_le_bytes(
                bytes
                    .get(at..at + 4)
                    .ok_or_else(|| Error::malformed("analytic transfer is truncated"))?
                    .try_into()
                    .unwrap(),
            ) as usize;
            at += 4;
            let cb = bytes
                .get(at..at + len)
                .ok_or_else(|| Error::malformed("analytic correction is truncated"))?;
            at += len;
            TransferOperator::from_canonical_bytes(cb)?
        } else {
            // A zero-weight correction is never used; build a minimal valid one.
            TransferOperator {
                channels,
                source_channels,
                taps: 1,
                delay: 0,
                weights: vec![0i16; usize::from(channels) * usize::from(source_channels)],
                bias: vec![0i32; usize::from(channels)],
            }
        };
        if at != bytes.len() {
            return Err(Error::malformed("analytic transfer has trailing bytes"));
        }
        let t = AnalyticTransfer {
            channels,
            source_channels,
            gain_num,
            gain_den,
            offset,
            delay,
            has_correction,
            correction,
        };
        t.validate()?;
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::model::LearnedModel;
    use crate::learned::object::LearnedObject;

    fn identity() -> AnalyticTransfer {
        AnalyticTransfer {
            channels: 1,
            source_channels: 1,
            gain_num: 1,
            gain_den: 1,
            offset: 0,
            delay: 0,
            has_correction: false,
            correction: TransferOperator {
                channels: 1,
                source_channels: 1,
                taps: 1,
                delay: 0,
                weights: vec![0],
                bias: vec![0],
            },
        }
    }

    #[test]
    fn rounded_div_is_half_away_from_zero() {
        assert_eq!(rounded_div(5, 2), 3);
        assert_eq!(rounded_div(-5, 2), -3);
        assert_eq!(rounded_div(4, 2), 2);
        assert_eq!(rounded_div(-4, 2), -2);
        assert_eq!(rounded_div(0, 7), 0);
    }

    #[test]
    fn analytic_identity_transfers_exactly() {
        let source: Vec<i32> = (0..500).map(|i| ((i * 313) % 9001) - 4500).collect();
        let t = identity();
        let o = LearnedObject::from_transfer_operator_exp2(
            LearnedModel::AnalyticTransfer(t.clone()),
            1,
            500,
            48_000,
            Vec::new(),
            &source,
            &source,
        )
        .unwrap();
        assert!(o.verify_with_source(&source, &source));
        assert_eq!(
            o.materialize_range_with_source(200, 30, Some(&source))
                .unwrap(),
            &source[200..230]
        );
    }

    #[test]
    fn analytic_gain_and_delay_close_exactly() {
        let source: Vec<i32> = (0..500).map(|i| ((i * 97) % 4001) - 2000).collect();
        // target = 3/4 * source delayed by 5
        let mut target = vec![0i32; 500];
        for t in 5..500 {
            target[t] = rounded_div(i128::from(source[t - 5]) * 3, 4) as i32;
        }
        let mut t = identity();
        t.gain_num = 3;
        t.gain_den = 4;
        t.delay = 5;
        let o = LearnedObject::from_transfer_operator_exp2(
            LearnedModel::AnalyticTransfer(t),
            1,
            500,
            48_000,
            Vec::new(),
            &source,
            &target,
        )
        .unwrap();
        assert!(o.verify_with_source(&source, &target));
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let mut t = identity();
        t.gain_num = 7;
        t.gain_den = 3;
        t.offset = -11;
        t.delay = 9;
        let b = t.canonical_bytes();
        let back = AnalyticTransfer::from_canonical_bytes(&b).unwrap();
        assert_eq!(back, t);
    }
}

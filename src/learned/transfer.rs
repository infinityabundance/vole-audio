//! Learned transfer operator (`O.4`, `O.5`, `O.7`).
//!
//! A transfer operator models a deterministic relationship between a source
//! object and a target object:
//!
//! ```text
//! H[t]     = F_Theta(S, C)[t]
//! X_hat[t] = sat_i32(H[t] + R[t])      with   X_hat == X
//! ```
//!
//! `S` is a **declared dependency** (a source `SampleObject` or deterministic
//! source state). The operator never treats that dependency as free: standalone
//! and marginal accounting are reported separately (`O.5`).
//!
//! The concrete operator here is a bounded causal cross-channel FIR over the
//! aligned source, which is the strongest simple transfer family and the fair
//! target for the analytic baselines in the transfer court.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, round_shift_half_away, sat_i32};

/// A learned causal cross-channel transfer operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferOperator {
    /// Target channels.
    pub channels: u8,
    /// Source channels.
    pub source_channels: u8,
    /// Causal taps over the source (`k = 1..=taps`).
    pub taps: u16,
    /// Alignment delay in frames: source index is `t - delay - (k-1)`.
    pub delay: i64,
    /// `taps · C · Cs` Q12 weights, indexed `((k-1)·C + out)·Cs + in`.
    pub weights: Vec<i16>,
    /// `C` Q12 biases.
    pub bias: Vec<i32>,
}

impl TransferOperator {
    pub fn validate(&self) -> Result<()> {
        let c = usize::from(self.channels);
        let cs = usize::from(self.source_channels);
        if c == 0 || c > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::malformed("transfer target channels out of range"));
        }
        if cs == 0 || cs > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::malformed("transfer source channels out of range"));
        }
        if self.taps == 0 || u32::from(self.taps) > crate::limits::MAX_LEARNED_TAPS {
            return Err(Error::limit("transfer tap count exceeds the bound"));
        }
        let expect = usize::from(self.taps)
            .checked_mul(c)
            .and_then(|n| n.checked_mul(cs))
            .ok_or_else(|| Error::limit("transfer weight count overflows"))?;
        if self.weights.len() != expect || self.bias.len() != c {
            return Err(Error::malformed("transfer operator shape mismatch"));
        }
        let mag = self.delay.unsigned_abs();
        if mag > u64::from(crate::limits::MAX_LEARNED_RECEPTIVE_FIELD) {
            return Err(Error::limit("transfer alignment delay exceeds the bound"));
        }
        let weight_bytes = (expect as u64) * 2 + (c as u64) * 4 + 24;
        if weight_bytes > crate::limits::MAX_LEARNED_WEIGHT_BYTES {
            return Err(Error::limit("transfer weight bytes exceed the bound"));
        }
        if self.ops_per_sample() > crate::limits::MAX_LEARNED_OPS_PER_SAMPLE {
            return Err(Error::limit(
                "transfer operations per sample exceed the bound",
            ));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.taps).saturating_add(self.delay.unsigned_abs())
    }

    pub fn ops_per_sample(&self) -> u64 {
        let c = u64::from(self.channels);
        let cs = u64::from(self.source_channels);
        u64::from(self.taps) * c * cs + c
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    /// Frames replayed to serve a seek: none (the source is a dependency).
    pub fn replay_frames(&self, _start: usize) -> usize {
        0
    }

    fn sample(&self, source: &[i32], src_frames: usize, si: i64, in_ch: usize) -> i64 {
        if si < 0 || si >= src_frames as i64 {
            0
        } else {
            let idx = si as usize * usize::from(self.source_channels) + in_ch;
            i64::from(source[idx])
        }
    }

    /// Compute the hypothesis over the whole extent from a source.
    pub fn hypothesis_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        let cs = usize::from(self.source_channels);
        if source.len() != frames * cs {
            return Err(Error::malformed("transfer source length mismatch"));
        }
        let c = usize::from(self.channels);
        let k = usize::from(self.taps);
        let mut h = vec![0i32; frames * c];
        for t in 0..frames {
            for out in 0..c {
                let mut acc: Acc = Acc::from(self.bias[out]);
                for kk in 1..=k {
                    let si = t as i64 - self.delay - (kk as i64 - 1);
                    for i in 0..cs {
                        let w = self.weights[((kk - 1) * c + out) * cs + i];
                        acc += Acc::from(w) * self.sample(source, frames, si, i);
                    }
                }
                h[t * c + out] =
                    sat_i32(round_shift_half_away(acc, crate::limits::LEARNED_WEIGHT_Q));
            }
        }
        Ok(h)
    }

    /// Reconstruct `[start, start+len)` exactly from the target residual and the
    /// source samples.
    pub fn evaluate_range_with_source(
        &self,
        residual: &[i32],
        source: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        let cs = usize::from(self.source_channels);
        if source.len() != frames * cs {
            return Err(Error::malformed("transfer source length mismatch"));
        }
        let c = usize::from(self.channels);
        if residual.len() != frames * c {
            return Err(Error::malformed("transfer residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("transfer range overflows"))?;
        if end > frames {
            return Err(Error::malformed("transfer range exceeds the extent"));
        }
        // Evaluate only the requested window (plus no target replay).
        let h_all = self.hypothesis_from_source(source, frames)?;
        let mut out = vec![0i32; len * c];
        for t in start..end {
            for i in 0..c {
                out[(t - start) * c + i] =
                    sat_i32(Acc::from(h_all[t * c + i]) + Acc::from(residual[t * c + i]));
            }
        }
        Ok(out)
    }

    /// Transfer objects cannot be evaluated without their source.
    pub fn evaluate_range(
        &self,
        _residual: &[i32],
        _frames: usize,
        _start: usize,
        _len: usize,
    ) -> Result<Vec<i32>> {
        Err(Error::dependency(
            "transfer object requires its source samples for evaluation",
        ))
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(3); // kind 3 = transfer operator
        out.push(self.channels);
        out.push(self.source_channels);
        out.extend_from_slice(&self.taps.to_le_bytes());
        out.extend_from_slice(&self.delay.to_le_bytes());
        for w in &self.weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for b in &self.bias {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<TransferOperator> {
        let mut r = crate::learned::serialization::Reader::new(bytes);
        let kind = r.u8()?;
        if kind != 3 {
            return Err(Error::malformed("transfer operator kind mismatch"));
        }
        let channels = r.u8()?;
        let source_channels = r.u8()?;
        let taps = r.u16()?;
        let delay = r.u64()? as i64;
        let c = usize::from(channels);
        let cs = usize::from(source_channels);
        let k = usize::from(taps);
        if c == 0 || cs == 0 || k == 0 {
            return Err(Error::malformed("transfer operator has a zero dimension"));
        }
        let w_count = k
            .checked_mul(c)
            .and_then(|n| n.checked_mul(cs))
            .ok_or_else(|| Error::limit("transfer weight count overflows"))?;
        let mut weights = Vec::with_capacity(w_count.min(1 << 20));
        for _ in 0..w_count {
            weights.push(r.i16()?);
        }
        let mut bias = Vec::with_capacity(c.min(1 << 16));
        for _ in 0..c {
            bias.push(r.i32()?);
        }
        r.finish()?;
        let t = TransferOperator {
            channels,
            source_channels,
            taps,
            delay,
            weights,
            bias,
        };
        t.validate()?;
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::{quantize_bias, quantize_weight};

    fn gain_operator() -> TransferOperator {
        TransferOperator {
            channels: 1,
            source_channels: 1,
            taps: 1,
            delay: 0,
            weights: vec![quantize_weight(0.5)],
            bias: vec![quantize_bias(0.0)],
        }
    }

    #[test]
    fn transfer_closes_exactly_and_is_window_exact() {
        let t = gain_operator();
        let source: Vec<i32> = (0..128).map(|i| (i * 1000) - 60_000).collect();
        let target: Vec<i32> = source.iter().map(|&s| s / 2).collect();
        let h = t.hypothesis_from_source(&source, 128).unwrap();
        let residual: Vec<i32> = (0..128).map(|i| target[i] - h[i]).collect();
        let recon = t
            .evaluate_range_with_source(&residual, &source, 128, 0, 128)
            .unwrap();
        assert_eq!(recon, target);
        // Windowed evaluation equals the slice.
        assert_eq!(
            t.evaluate_range_with_source(&residual, &source, 128, 40, 30)
                .unwrap(),
            &target[40..70]
        );
        // Without a source it refuses.
        assert!(t.evaluate_range(&residual, 128, 0, 128).is_err());
    }

    #[test]
    fn delay_and_multichannel_shapes_validate() {
        let mut t = gain_operator();
        t.delay = 3;
        t.validate().unwrap();
        t.taps = 2;
        assert!(t.validate().is_err()); // weights no longer match taps
        t.weights = vec![0; 2];
        t.source_channels = 2;
        assert!(t.validate().is_err()); // weights no longer match source channels
        t.weights = vec![0; 4];
        t.validate().unwrap();
    }

    #[test]
    fn canonical_round_trip_and_bounds() {
        let t = gain_operator();
        let bytes = t.canonical_bytes();
        assert_eq!(TransferOperator::from_canonical_bytes(&bytes).unwrap(), t);
        assert!(TransferOperator::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut bomb = bytes.clone();
        bomb[3..5].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(TransferOperator::from_canonical_bytes(&bomb).is_err());
    }
}

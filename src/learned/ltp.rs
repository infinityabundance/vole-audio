//! Long-term / pitch prediction (Exp2, priority `4`).
//!
//! A short consecutive FIR can need many taps to approximate a relationship
//! that is really one lag plus a few gains. MPEG-4 ALS applies long-term
//! prediction to the short-term residual for exactly this reason. This family
//! places a small long-term stage **over the short-term residual**:
//!
//! ```text
//! H_short[t] = short(X_hat history)[t]
//! E_short[t] = X_hat[t] - H_short[t]
//! H_long[t]  = Σ_j g[j] · E_short[t - (τ - j)]
//! H_total[t] = H_short[t] + H_long[t]
//! X_hat[t]   = sat_i32(H_total[t] + R[t])
//! ```
//!
//! Model kind tag `6`. Mono-first. Only already-reconstructed samples are ever
//! read (closed-loop exact predictive decoding).

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, round_shift_half_away, sat_i32};
use crate::learned::sparse::SparseLinearPredictor;

/// A short-term sparse predictor plus a long-term (pitch) stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongTermPredictor {
    /// Channel count (mono-only in this build).
    pub channels: u8,
    /// The short-term stage.
    pub short: SparseLinearPredictor,
    /// Pitch lag `τ` (`>= long-term taps`).
    pub lag: u16,
    /// Long-term gains (Q12), in order for lags `τ, τ-1, …, τ-G+1`.
    pub gains: Vec<i16>,
    /// Optional block-local reset.
    pub block_frames: Option<u32>,
}

/// Per-frame closed-loop state: short X_hat history and short-residual history.
struct Stage {
    x_hist: Vec<i32>,
    e_hist: Vec<i32>,
}

impl LongTermPredictor {
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed("long-term predictor is mono-only"));
        }
        self.short.validate()?;
        let g = self.gains.len();
        if g == 0 || g as u32 > crate::limits::MAX_LEARNED_LTP_TAPS {
            return Err(Error::limit("long-term tap count out of range"));
        }
        if self.lag == 0 || u32::from(self.lag) > crate::limits::MAX_LEARNED_TAPS {
            return Err(Error::limit("long-term lag exceeds the bound"));
        }
        if usize::from(self.lag) < g {
            return Err(Error::malformed(
                "long-term lag must be at least the long-term tap count",
            ));
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit("long-term block size exceeds the bound"));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.lag).max(self.short.receptive_field())
    }

    pub fn ops_per_sample(&self) -> u64 {
        self.short.ops_per_sample() + self.gains.len() as u64
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    /// Per-frame state: short X_hat history and short-residual history.
    fn new_stage(&self) -> Stage {
        Stage {
            x_hist: vec![0i32; self.short.max_lag()],
            e_hist: vec![0i32; usize::from(self.lag)],
        }
    }

    /// Hypothesis at the current frame given the running stage.
    #[inline]
    fn hypothesis(&self, stage: &Stage, total_out: &mut i32, e_short_out: &mut i32) {
        let mut h = [0i32; 1];
        self.short.hypothesis(&stage.x_hist, &mut h);
        let mut acc = Acc::from(0);
        let g = self.gains.len();
        for (j, &gain) in self.gains.iter().enumerate() {
            // E_short[t - (τ - j)] = e_hist[τ - j - 1].
            let idx = usize::from(self.lag) - j - 1;
            acc += Acc::from(gain) * Acc::from(stage.e_hist[idx]);
        }
        let h_long = round_shift_half_away(acc, crate::limits::LEARNED_WEIGHT_Q);
        *total_out = sat_i32(Acc::from(h[0]) + Acc::from(h_long as i64));
        *e_short_out = h[0];
        let _ = g;
    }

    fn push(&self, stage: &mut Stage, x_hat: i32, h_short: i32) {
        let xm = stage.x_hist.len();
        if xm > 1 {
            stage.x_hist.copy_within(0..xm - 1, 1);
        }
        if xm >= 1 {
            stage.x_hist[0] = x_hat;
        }
        let em = stage.e_hist.len();
        if em > 1 {
            stage.e_hist.copy_within(0..em - 1, 1);
        }
        if em >= 1 {
            stage.e_hist[0] = sat_i32(Acc::from(x_hat) - Acc::from(h_short));
        }
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
            return Err(Error::malformed("long-term residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("long-term range overflows"))?;
        if end > frames {
            return Err(Error::malformed("long-term range exceeds the extent"));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        for (from, to) in self.ranges(frames) {
            let mut stage = self.new_stage();
            for t in from..to {
                let (mut h_total, mut h_short) = (0i32, 0i32);
                self.hypothesis(&stage, &mut h_total, &mut h_short);
                let x = sat_i32(Acc::from(h_total) + Acc::from(residual[t]));
                if t >= start && t < end {
                    out[t - start] = x;
                }
                self.push(&mut stage, x, h_short);
            }
        }
        Ok(out)
    }

    /// Hypothesis over a whole extent, open-loop from a source.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("long-term source length mismatch"));
        }
        let mut out = vec![0i32; frames];
        for (from, to) in self.ranges(frames) {
            let mut stage = self.new_stage();
            for t in from..to {
                let (mut h_total, mut h_short) = (0i32, 0i32);
                self.hypothesis(&stage, &mut h_total, &mut h_short);
                out[t] = h_total;
                self.push(&mut stage, source[t], h_short);
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

    /// Canonical bytes:
    /// `kind(6) || channels || short_len(u32) || short || lag(u16) || G(u8) ||
    ///  gains || block(u32)`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let short = self.short.canonical_bytes();
        let mut out = Vec::with_capacity(short.len() + 16 + self.gains.len() * 2);
        out.push(6);
        out.push(self.channels);
        out.extend_from_slice(&(short.len() as u32).to_le_bytes());
        out.extend_from_slice(&short);
        out.extend_from_slice(&self.lag.to_le_bytes());
        out.push(self.gains.len() as u8);
        for &g in &self.gains {
            out.extend_from_slice(&g.to_le_bytes());
        }
        out.extend_from_slice(&self.block_frames.unwrap_or(0).to_le_bytes());
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<LongTermPredictor> {
        if bytes.len() < 9 || bytes[0] != 6 {
            return Err(Error::malformed("long-term model header mismatch"));
        }
        let channels = bytes[1];
        let short_len = u32::from_le_bytes(bytes[2..6].try_into().unwrap()) as usize;
        if short_len == 0 || bytes.len() < 6 + short_len + 7 {
            return Err(Error::malformed("long-term short model is truncated"));
        }
        let short = SparseLinearPredictor::from_canonical_bytes(&bytes[6..6 + short_len])?;
        let mut at = 6 + short_len;
        let lag = u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap());
        at += 2;
        let g = bytes[at] as usize;
        at += 1;
        if g == 0 || g > crate::limits::MAX_LEARNED_LTP_TAPS as usize {
            return Err(Error::limit("long-term tap count out of range"));
        }
        if bytes.len() < at + g * 2 + 4 {
            return Err(Error::malformed("long-term gains are truncated"));
        }
        let mut gains = Vec::with_capacity(g);
        for _ in 0..g {
            gains.push(i16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()));
            at += 2;
        }
        let block = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        at += 4;
        if at != bytes.len() {
            return Err(Error::malformed("long-term model has trailing bytes"));
        }
        let p = LongTermPredictor {
            channels,
            short,
            lag,
            gains,
            block_frames: if block == 0 { None } else { Some(block) },
        };
        p.validate()?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::quantize_weight;
    use crate::learned::model::LearnedModel;
    use crate::learned::object::LearnedObject;
    use crate::learned::sparse::COEFF_DELTA_VARINT;

    fn ltp() -> LongTermPredictor {
        LongTermPredictor {
            channels: 1,
            short: SparseLinearPredictor {
                channels: 1,
                lags: vec![1],
                weights: vec![quantize_weight(0.9)],
                bias: vec![0],
                block_frames: None,
                coeff_encoding: COEFF_DELTA_VARINT,
            },
            lag: 20,
            gains: vec![
                quantize_weight(0.7),
                quantize_weight(-0.2),
                quantize_weight(0.15),
            ],
            block_frames: None,
        }
    }

    #[test]
    fn long_term_predictor_closes_exactly() {
        // A quasi-periodic signal with a strong period-20 component.
        let mut x = vec![0i32; 64];
        for t in 64..3000 {
            let prev = x[t - 1];
            let lp = x[t - 20] as i64;
            let v = (prev as i64 * 3) / 5 + (lp * 7) / 10 + ((t % 13) as i64 - 6) * 3;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let p = ltp();
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::LongTerm(p.clone()),
            1,
            3000,
            48_000,
            Vec::new(),
            &x,
        )
        .unwrap();
        assert!(o.verify(&x));
        assert_eq!(o.materialize_range(1500, 60).unwrap(), &x[1500..1560]);
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let p = ltp();
        let b = p.canonical_bytes();
        let back = LongTermPredictor::from_canonical_bytes(&b).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn validation_rejects_short_lag() {
        let mut p = ltp();
        p.lag = 2;
        assert!(p.validate().is_err());
    }
}

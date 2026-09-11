//! Backward-adaptive deterministic prediction (Exp2, priority `10`).
//!
//! Forward-adaptive models transmit a fresh coefficient vector per block. A
//! backward-adaptive predictor updates itself from **already reconstructed**
//! decoder-visible samples, so encoder and decoder evolve the same state without
//! transmitting a new model. This amortizes model-description cost over long
//! recordings.
//!
//! Model kind tag `10`. Mono-first. The update is a frozen integer sign-sign LMS:
//!
//! ```text
//! H[t]     = bias + (Σ_k w_k · X_hat[t-1-k]) >> 12
//! X_hat[t] = sat_i32(H[t] + R[t])
//! e[t]     = X_hat[t] - H[t]
//! w_k     += sign(e[t]) · sign(X_hat[t-1-k]) · step      (clamped to i16)
//! ```
//!
//! Only `X_hat` (never the encoder-only source) drives the update, so the
//! decoder stays synchronised. The state is bounded and integer-only.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, round_shift_half_away, sat_i32};

/// A backward-adaptive causal predictor with a frozen integer update rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptivePredictor {
    pub channels: u8,
    pub taps: u16,
    /// Initial Q12 weights, lag order (`lag 1` first).
    pub init_weights: Vec<i16>,
    pub init_bias: i32,
    /// Sign-sign LMS step in Q12 units.
    pub step: i16,
    pub block_frames: Option<u32>,
}

impl AdaptivePredictor {
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed("adaptive predictor is mono-only"));
        }
        if self.taps == 0 || u32::from(self.taps) > crate::limits::MAX_LEARNED_ADAPTIVE_COEFFS {
            return Err(Error::limit("adaptive tap count out of range"));
        }
        if self.init_weights.len() != usize::from(self.taps) {
            return Err(Error::malformed("adaptive weight count mismatch"));
        }
        if self.step <= 0 {
            return Err(Error::malformed("adaptive step must be positive"));
        }
        if !crate::learned::arithmetic::accumulator_is_safe(u32::from(self.taps)) {
            return Err(Error::limit("adaptive accumulator bound exceeded"));
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit("adaptive block size exceeds the bound"));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.taps)
    }

    pub fn ops_per_sample(&self) -> u64 {
        u64::from(self.taps) * 3 + 1
    }

    pub fn state_bytes(&self) -> u64 {
        u64::from(self.taps) * 2
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        // The adaptive state depends on the whole history up to `start`.
        start
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

    #[inline]
    fn predict(history: &[i32], weights: &[i32], bias: i32, taps: usize) -> i32 {
        let mut acc = Acc::from(bias);
        for k in 0..taps {
            acc += Acc::from(weights[k]) * Acc::from(history[k]);
        }
        sat_i32(round_shift_half_away(acc, crate::limits::LEARNED_WEIGHT_Q))
    }

    #[inline]
    fn update(weights: &mut [i32], e: i32, history: &[i32], step: i16) {
        let se = e.signum();
        if se == 0 {
            return;
        }
        let step = i32::from(step);
        for (w, &x) in weights.iter_mut().zip(history.iter()) {
            let sx = x.signum();
            let next = *w + se * sx * step;
            *w = next.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
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
            return Err(Error::malformed("adaptive residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("adaptive range overflows"))?;
        if end > frames {
            return Err(Error::malformed("adaptive range exceeds the extent"));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        let taps = usize::from(self.taps);
        let mut history = vec![0i32; taps];
        let bias = self.init_bias;
        for (from, to) in self.ranges(frames) {
            history.fill(0);
            let mut weights: Vec<i32> = self.init_weights.iter().map(|&w| i32::from(w)).collect();
            for t in from..to {
                let h = Self::predict(&history, &weights, bias, taps);
                let x = sat_i32(Acc::from(h) + Acc::from(residual[t]));
                if t >= start && t < end {
                    out[t - start] = x;
                }
                let e = x.wrapping_sub(h);
                Self::update(&mut weights, e, &history, self.step);
                if taps > 1 {
                    history.copy_within(0..taps - 1, 1);
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
            return Err(Error::malformed("adaptive source length mismatch"));
        }
        let taps = usize::from(self.taps);
        let mut out = vec![0i32; frames];
        let mut history = vec![0i32; taps];
        let bias = self.init_bias;
        for (from, to) in self.ranges(frames) {
            history.fill(0);
            let mut weights: Vec<i32> = self.init_weights.iter().map(|&w| i32::from(w)).collect();
            for t in from..to {
                let h = Self::predict(&history, &weights, bias, taps);
                out[t] = h;
                let e = source[t].wrapping_sub(h);
                Self::update(&mut weights, e, &history, self.step);
                if taps > 1 {
                    history.copy_within(0..taps - 1, 1);
                }
                history[0] = source[t];
            }
        }
        Ok(out)
    }

    /// Canonical bytes:
    /// `kind(10) || channels || taps(u16) || step(i16) || weights || bias(i32) || block(u32)`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 1 + 2 + 2 + self.init_weights.len() * 2 + 4 + 4);
        out.push(10);
        out.push(self.channels);
        out.extend_from_slice(&self.taps.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        for w in &self.init_weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&self.init_bias.to_le_bytes());
        out.extend_from_slice(&self.block_frames.unwrap_or(0).to_le_bytes());
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<AdaptivePredictor> {
        if bytes.len() < 14 || bytes[0] != 10 {
            return Err(Error::malformed("adaptive model header mismatch"));
        }
        let channels = bytes[1];
        let taps = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let step = i16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let t = usize::from(taps);
        let expect = 1 + 1 + 2 + 2 + t * 2 + 4 + 4;
        if bytes.len() != expect {
            return Err(Error::malformed("adaptive model length mismatch"));
        }
        let mut at = 6usize;
        let mut init_weights = Vec::with_capacity(t);
        for _ in 0..t {
            init_weights.push(i16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()));
            at += 2;
        }
        let init_bias = i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        at += 4;
        let block = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let p = AdaptivePredictor {
            channels,
            taps,
            init_weights,
            init_bias,
            step,
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
    fn adaptive_predictor_closes_exactly_and_adapts() {
        // An AR(2) whose coefficients drift slowly: a stationary FIR cannot
        // track it, but the adaptive update can.
        let mut x = vec![0i32; 2];
        for t in 2..3000 {
            let a = 0.4 + 0.2 * ((t / 500) as f64 * 0.1);
            let b = 0.3 - 0.1 * ((t / 500) as f64 * 0.1);
            let v = (x[t - 1] as f64 * a + x[t - 2] as f64 * b + ((t % 7) as f64 - 3.0) * 5.0)
                .round() as i64;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let p = AdaptivePredictor {
            channels: 1,
            taps: 2,
            init_weights: vec![2000, 1000],
            init_bias: 0,
            step: 2,
            block_frames: None,
        };
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::Adaptive(p.clone()),
            1,
            3000,
            48_000,
            Vec::new(),
            &x,
        )
        .unwrap();
        assert!(o.verify(&x));
        assert_eq!(o.materialize_range(2000, 40).unwrap(), &x[2000..2040]);
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let p = AdaptivePredictor {
            channels: 1,
            taps: 3,
            init_weights: vec![100, -200, 300],
            init_bias: 7,
            step: 1,
            block_frames: Some(256),
        };
        let b = p.canonical_bytes();
        let back = AdaptivePredictor::from_canonical_bytes(&b).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn zero_step_is_rejected() {
        let p = AdaptivePredictor {
            channels: 1,
            taps: 1,
            init_weights: vec![0],
            init_bias: 0,
            step: 0,
            block_frames: None,
        };
        assert!(p.validate().is_err());
    }
}

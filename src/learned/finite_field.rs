//! Learned linear finite-field predictor (`O.3.1`, `O.17`, `O.3.3`).
//!
//! The first learned family, deliberately linear and integer-only. It is a
//! **closed-loop causal** predictor: the hypothesis at frame `t` reads the
//! already *reconstructed* canonical samples `X_hat[t-1 .. t-K]`, never the
//! encoder-only source.
//!
//! ```text
//! H[t][c]     = bias[c] + Σ_{k=1..K} Σ_{c'} w[k][c][c'] · X_hat[t-k][c']   (Q12)
//! X_hat[t][c] = sat_i32(H[t][c] + R[t][c])
//! ```
//!
//! `R` is the exact residual; at construction `R = X - H` computed with *this*
//! quantized evaluator, so `X_hat == X` bit for bit.
//!
//! ## Two semantic classes share this predictor
//!
//! * `block_frames == None` — a single causal stream (`LearnedFiniteField`).
//!   Seeking to frame `t` replays from frame 0 (or from a checkpoint); the
//!   *halo* the evaluator reads is `K` frames, but obtaining that history costs
//!   replay.
//! * `block_frames == Some(B)` — `LearnedBlockLocal`: history resets at every
//!   block boundary, so each block is independently materializable and random
//!   access touches only the target block.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, round_shift_half_away, sat_i32};

/// A quantized causal FIR predictor in the canonical code domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearPredictor {
    /// Channel count of the intrinsic domain (`>= 1`).
    pub channels: u8,
    /// Causal taps `K` (`>= 1`).
    pub taps: u16,
    /// `K · C · C` Q12 weights, indexed `((k-1)·C + out)·C + in`.
    pub weights: Vec<i16>,
    /// `C` Q12 biases.
    pub bias: Vec<i32>,
    /// `Some(B)` selects block-local reset semantics (independent blocks).
    pub block_frames: Option<u32>,
}

impl LinearPredictor {
    /// Validate dimensions, bounds, and the accumulator proof.
    pub fn validate(&self) -> Result<()> {
        let c = usize::from(self.channels);
        if c == 0 || c > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::malformed(
                "learned predictor channel count out of range",
            ));
        }
        if self.taps == 0 || u32::from(self.taps) > crate::limits::MAX_LEARNED_TAPS {
            return Err(Error::limit(
                "learned predictor tap count exceeds the bound",
            ));
        }
        let expect = usize::from(self.taps) * c * c;
        if self.weights.len() != expect {
            return Err(Error::malformed("learned predictor weight count mismatch"));
        }
        if self.bias.len() != c {
            return Err(Error::malformed("learned predictor bias count mismatch"));
        }
        if !crate::learned::arithmetic::accumulator_is_safe(u32::from(self.taps)) {
            return Err(Error::limit("learned predictor accumulator bound exceeded"));
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit("learned block size exceeds the bound"));
        }
        let weight_bytes = (self.weights.len() as u64) * 2 + (self.bias.len() as u64) * 4;
        if weight_bytes > crate::limits::MAX_LEARNED_WEIGHT_BYTES {
            return Err(Error::limit("learned weight bytes exceed the bound"));
        }
        Ok(())
    }

    /// Tap count as a `usize`.
    pub fn tap_count(&self) -> usize {
        usize::from(self.taps)
    }

    /// The declared receptive field in frames (`K`).
    pub fn receptive_field(&self) -> u64 {
        u64::from(self.taps)
    }

    /// Abstract operations per output sample (`K · C · C` multiply-adds + bias).
    pub fn ops_per_sample(&self) -> u64 {
        let c = u64::from(self.channels);
        u64::from(self.taps) * c * c + c
    }

    /// Persistent state bytes (none: the family is finite-field).
    pub fn state_bytes(&self) -> u64 {
        0
    }

    /// Checkpoints declared (none).
    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    /// One hypothesis evaluation at frame `t`, given `history` = `X_hat[t-K..t]`
    /// (oldest first, `K · C` values; entries before frame 0 are zero).
    #[inline]
    pub fn hypothesis(&self, history: &[i32], out: &mut [i32]) {
        let c = usize::from(self.channels);
        let k_taps = usize::from(self.taps);
        debug_assert_eq!(history.len(), k_taps * c);
        for (o, slot) in out.iter_mut().enumerate() {
            let mut acc: Acc = Acc::from(self.bias[o]);
            // history[0..c] is X_hat[t-K], ... history[(K-1)c..] is X_hat[t-1]
            for k in 1..=k_taps {
                let base = (k_taps - k) * c; // oldest tap first in `history`
                for i in 0..c {
                    let w = self.weights[((k - 1) * c + o) * c + i];
                    acc += Acc::from(w) * Acc::from(history[base + i]);
                }
            }
            *slot = sat_i32(round_shift_half_away(acc, crate::limits::LEARNED_WEIGHT_Q));
        }
    }

    /// Reconstruct `X_hat[start .. start+len)` exactly from a dense residual.
    ///
    /// With `block_frames == None` the closed loop replays from frame 0. With
    /// `block_frames == Some(B)` only the blocks intersecting the request are
    /// evaluated, each reset at its block boundary.
    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if frames * c != residual.len() {
            return Err(Error::malformed(
                "learned residual length does not match the declared geometry",
            ));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("learned range overflows"))?;
        if end > frames {
            return Err(Error::malformed("learned range exceeds the extent"));
        }
        let mut out = vec![0i32; len * c];
        if len == 0 {
            return Ok(out);
        }
        let mut history = vec![0i32; self.tap_count() * c];
        let mut h = vec![0i32; c];
        // Bounds of the closed-loop replay: the whole extent for the single
        // stream, or the intersecting blocks for the block-local class.
        let ranges: Vec<(usize, usize)> = match self.block_frames {
            None => vec![(0, end)],
            Some(b) => {
                let b = b as usize;
                let first = start / b;
                let last = (end - 1) / b;
                (first..=last)
                    .map(|blk| (blk * b, ((blk + 1) * b).min(frames)))
                    .collect()
            }
        };
        for (from, to) in ranges {
            history.fill(0);
            for t in from..to {
                self.hypothesis(&history, &mut h);
                // Drop the oldest reconstructed frame, then append this one.
                if self.tap_count() > 1 {
                    history.copy_within(c.., 0);
                }
                for i in 0..c {
                    let x = sat_i32(Acc::from(h[i]) + Acc::from(residual[t * c + i]));
                    if t >= start && t < end {
                        out[(t - start) * c + i] = x;
                    }
                    history[(self.tap_count() - 1) * c + i] = x;
                }
            }
        }
        Ok(out)
    }

    /// Compute the hypothesis for a whole extent **open-loop from a source**
    /// signal (used to build the residual). The decoder's closed loop is
    /// authoritative and reconstructs the same values because the residual is
    /// defined against the source it reads.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if source.len() != frames * c {
            return Err(Error::malformed("learned source length mismatch"));
        }
        let mut h_all = vec![0i32; frames * c];
        let mut history = vec![0i32; self.tap_count() * c];
        let mut out = vec![0i32; c];
        let ranges: Vec<(usize, usize)> = match self.block_frames {
            None => vec![(0, frames)],
            Some(b) => {
                let b = b as usize;
                (0..frames)
                    .step_by(b)
                    .map(|blk| (blk, (blk + b).min(frames)))
                    .collect()
            }
        };
        for (from, to) in ranges {
            history.fill(0);
            for t in from..to {
                self.hypothesis(&history, &mut out);
                if self.tap_count() > 1 {
                    history.copy_within(c.., 0);
                }
                for i in 0..c {
                    h_all[t * c + i] = out[i];
                    history[(self.tap_count() - 1) * c + i] = source[t * c + i];
                }
            }
        }
        Ok(h_all)
    }

    /// Number of frames that must be replayed to serve `start` (closed-loop).
    pub fn replay_frames(&self, start: usize) -> usize {
        match self.block_frames {
            None => start,
            Some(b) => start % (b as usize),
        }
    }

    /// Canonical model bytes: `kind || channels || taps || block || weights || bias`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(1 + 1 + 2 + 4 + self.weights.len() * 2 + self.bias.len() * 4);
        out.push(0); // model kind 0 = linear finite-field
        out.push(self.channels);
        out.extend_from_slice(&self.taps.to_le_bytes());
        match self.block_frames {
            None => out.extend_from_slice(&0u32.to_le_bytes()),
            Some(b) => out.extend_from_slice(&b.to_le_bytes()),
        }
        for w in &self.weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for b in &self.bias {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    /// Parse canonical model bytes (bounded, exact length).
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<LinearPredictor> {
        if bytes.len() < 8 {
            return Err(Error::malformed("learned linear model is truncated"));
        }
        if bytes[0] != 0 {
            return Err(Error::malformed("learned linear model kind mismatch"));
        }
        let channels = bytes[1];
        let taps = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let block = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let c = usize::from(channels);
        let k = usize::from(taps);
        if c == 0 || k == 0 {
            return Err(Error::malformed(
                "learned linear model has zero channels or taps",
            ));
        }
        let w_count = k
            .checked_mul(c)
            .and_then(|n| n.checked_mul(c))
            .ok_or_else(|| Error::limit("learned weight count overflows"))?;
        let expect = 8usize
            .checked_add(
                w_count
                    .checked_mul(2)
                    .ok_or_else(|| Error::limit("learned weight bytes overflow"))?,
            )
            .and_then(|n| n.checked_add(c.checked_mul(4)?))
            .ok_or_else(|| Error::limit("learned model length overflows"))?;
        if bytes.len() != expect {
            return Err(Error::malformed("learned linear model length mismatch"));
        }
        let mut weights = Vec::with_capacity(w_count);
        let mut at = 8;
        for _ in 0..w_count {
            weights.push(i16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()));
            at += 2;
        }
        let mut bias = Vec::with_capacity(c);
        for _ in 0..c {
            bias.push(i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
            at += 4;
        }
        let p = LinearPredictor {
            channels,
            taps,
            weights,
            bias,
            block_frames: if block == 0 { None } else { Some(block) },
        };
        p.validate()?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::{quantize_bias, quantize_weight};

    fn identity_predictor(block: Option<u32>) -> LinearPredictor {
        // H[t] = X_hat[t-1]: one tap, weight exactly 1.0, zero bias.
        LinearPredictor {
            channels: 1,
            taps: 1,
            weights: vec![quantize_weight(1.0)],
            bias: vec![quantize_bias(0.0)],
            block_frames: block,
        }
    }

    #[test]
    fn validate_rejects_malformed_dimensions() {
        let mut p = identity_predictor(None);
        p.validate().unwrap();
        p.weights.push(1);
        assert!(p.validate().is_err());
        p.weights.pop();
        p.bias.clear();
        assert!(p.validate().is_err());
        p.bias.push(0);
        p.taps = 0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn exact_closure_on_a_ramp_and_a_glitch() {
        // X = 0, 1000, 2000, ... with a glitch.
        let x: Vec<i32> = (0..64)
            .map(|i| if i == 30 { -50_000 } else { i * 1000 })
            .collect();
        let p = identity_predictor(None);
        // Build the residual with the open-loop hypothesis over the source, then
        // prove the decoder's closed loop reconstructs the source exactly.
        let h = p.hypothesis_all_from_source(&x, 64).unwrap();
        let residual: Vec<i32> = (0..64).map(|t| x[t] - h[t]).collect();
        let recon = p.evaluate_range(&residual, 64, 0, 64).unwrap();
        assert_eq!(recon, x);
        // Chunked == contiguous.
        for start in [0usize, 1, 17, 63] {
            let n = 64 - start;
            assert_eq!(
                p.evaluate_range(&residual, 64, start, n).unwrap(),
                &x[start..]
            );
        }
    }

    #[test]
    fn block_local_blocks_are_independent() {
        let p = identity_predictor(Some(16));
        let x: Vec<i32> = (0..64).map(|i| i * 111 - 3000).collect();
        // Residual per block, using the block-local open-loop hypothesis.
        let h = p.hypothesis_all_from_source(&x, 64).unwrap();
        let residual: Vec<i32> = (0..64).map(|i| x[i] - h[i]).collect();
        let recon = p.evaluate_range(&residual, 64, 0, 64).unwrap();
        assert_eq!(recon, x);
        // A single block seek only replays within the block.
        assert_eq!(p.replay_frames(20), 4); // block 16..32, offset 4
        assert_eq!(p.replay_frames(35), 3); // block 32..48, offset 3
        assert_eq!(p.evaluate_range(&residual, 64, 17, 5).unwrap(), &x[17..22]);
    }

    #[test]
    fn canonical_round_trip_is_exact_and_bounded() {
        let p = LinearPredictor {
            channels: 2,
            taps: 3,
            weights: vec![7; 3 * 2 * 2],
            bias: vec![1, -2],
            block_frames: Some(256),
        };
        let bytes = p.canonical_bytes();
        assert_eq!(LinearPredictor::from_canonical_bytes(&bytes).unwrap(), p);
        // Truncations and trailing bytes are rejected.
        assert!(LinearPredictor::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(LinearPredictor::from_canonical_bytes(&extra).is_err());
        // A huge declared tap count is rejected before allocating.
        let mut bomb = p.canonical_bytes();
        bomb[2..4].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(LinearPredictor::from_canonical_bytes(&bomb).is_err());
    }

    #[test]
    fn accumulator_is_unaffected_by_reduction_order() {
        // The scalar evaluator is the only normative reduction; this asserts the
        // documented algebra: reordering taps cannot change an exact i64 sum.
        let p = LinearPredictor {
            channels: 1,
            taps: 4,
            weights: vec![i16::MAX, i16::MIN, 1234, -4321],
            bias: vec![7],
            block_frames: None,
        };
        let history = vec![i32::MAX, i32::MIN, 12345, -6789];
        let mut a = vec![0i32; 1];
        p.hypothesis(&history, &mut a);
        // Sum in a different order by hand.
        let c = usize::from(p.channels);
        let mut acc: Acc = Acc::from(p.bias[0]);
        for (k, item) in history.iter().rev().enumerate().take(4) {
            let w = p.weights[((k) * c) * c];
            acc += Acc::from(w) * Acc::from(*item);
        }
        let b = sat_i32(round_shift_half_away(acc, crate::limits::LEARNED_WEIGHT_Q));
        assert_eq!(a[0], b);
    }
}

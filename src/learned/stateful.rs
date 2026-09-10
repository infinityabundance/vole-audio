//! Closed-loop stateful predictor (`O.3.2`, `O.13`, `O.32`).
//!
//! Canonical semantic order is normative:
//!
//! ```text
//! H[t]     = sat_i32(round_shift(out_b + W_out · S_t, 12))
//! X_hat[t] = sat_i32(H[t] + R[t])
//! S_{t+1}  = act(round_shift(rec_b + W_rec · S_t + W_in · X_hat[t], 12))
//! ```
//!
//! The state update consumes the **reconstructed** exact current sample, never
//! an encoder-only source sample the decoder does not yet possess. This is
//! closed-loop exact predictive decoding.
//!
//! Seeks use canonical checkpoints: the nearest checkpoint at or before the
//! request is loaded and the closed loop is replayed deterministically.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, Activation, round_shift_half_away, sat_i32};

/// One canonical state checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCheckpoint {
    pub frame: u64,
    pub state: Vec<i32>,
}

/// A bounded closed-loop recurrent predictor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatefulPredictor {
    pub channels: u8,
    pub state_dim: u16,
    /// Output weights `C × state_dim` (Q12).
    pub out_w: Vec<i16>,
    /// Output biases `C` (Q12).
    pub out_b: Vec<i32>,
    /// Recurrent weights `state_dim × state_dim` (Q12).
    pub rec_w: Vec<i16>,
    /// Recurrent biases `state_dim` (Q12).
    pub rec_b: Vec<i32>,
    /// Input weights `state_dim × C` (Q12).
    pub in_w: Vec<i16>,
    /// State activation.
    pub activation: Activation,
    /// Checkpoint spacing in frames (`0` = none).
    pub checkpoint_interval: u32,
    /// Canonical checkpoints, ascending by frame.
    pub checkpoints: Vec<StateCheckpoint>,
}

impl StatefulPredictor {
    pub fn validate(&self) -> Result<()> {
        let c = usize::from(self.channels);
        let s = usize::from(self.state_dim);
        if c == 0 || c > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::malformed(
                "stateful predictor channel count out of range",
            ));
        }
        if s == 0 || s > 4096 {
            return Err(Error::limit(
                "stateful predictor state dimension exceeds the bound",
            ));
        }
        let expect_out = c
            .checked_mul(s)
            .ok_or_else(|| Error::limit("stateful output weights overflow"))?;
        let expect_rec = s
            .checked_mul(s)
            .ok_or_else(|| Error::limit("stateful recurrent weights overflow"))?;
        let expect_in = s
            .checked_mul(c)
            .ok_or_else(|| Error::limit("stateful input weights overflow"))?;
        if self.out_w.len() != expect_out
            || self.out_b.len() != c
            || self.rec_w.len() != expect_rec
            || self.rec_b.len() != s
            || self.in_w.len() != expect_in
        {
            return Err(Error::malformed("stateful predictor shape mismatch"));
        }
        let weight_bytes = ((expect_out + expect_rec + expect_in) as u64) * 2
            + ((c + s) as u64) * 4
            + self
                .checkpoints
                .iter()
                .map(|cp| 8 + (cp.state.len() as u64) * 4)
                .sum::<u64>();
        if weight_bytes > crate::limits::MAX_LEARNED_WEIGHT_BYTES {
            return Err(Error::limit("stateful weight bytes exceed the bound"));
        }
        self.activation.validate()?;
        if self.checkpoints.len() as u32 > crate::limits::MAX_LEARNED_CHECKPOINTS {
            return Err(Error::limit("stateful checkpoint count exceeds the bound"));
        }
        for w in self.checkpoints.windows(2) {
            if w[0].frame >= w[1].frame {
                return Err(Error::malformed("stateful checkpoints must ascend"));
            }
        }
        for cp in &self.checkpoints {
            if cp.state.len() != s {
                return Err(Error::malformed(
                    "stateful checkpoint state dimension mismatch",
                ));
            }
            if self.checkpoint_interval != 0 && cp.frame % u64::from(self.checkpoint_interval) != 0
            {
                return Err(Error::malformed(
                    "stateful checkpoint is not on the declared interval",
                ));
            }
        }
        if self.ops_per_sample() > crate::limits::MAX_LEARNED_OPS_PER_SAMPLE {
            return Err(Error::limit(
                "stateful operations per sample exceed the bound",
            ));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        // The state summarises all prior history.
        u64::from(self.state_dim)
    }

    pub fn ops_per_sample(&self) -> u64 {
        let c = u64::from(self.channels);
        let s = u64::from(self.state_dim);
        s * s + s * c + c * s + s
    }

    pub fn state_bytes(&self) -> u64 {
        u64::from(self.state_dim) * 4
    }

    pub fn checkpoint_count(&self) -> u32 {
        self.checkpoints.len() as u32
    }

    pub fn checkpoint_bytes(&self) -> u64 {
        self.checkpoints
            .iter()
            .map(|cp| 8 + (cp.state.len() as u64) * 4)
            .sum()
    }

    fn initial_state(&self) -> Vec<i32> {
        vec![0i32; usize::from(self.state_dim)]
    }

    #[inline]
    fn emit(&self, state: &[i32], out: &mut [i32]) {
        let s = usize::from(self.state_dim);
        for (o, slot) in out.iter_mut().enumerate() {
            let mut acc: Acc = Acc::from(self.out_b[o]);
            for (j, &sj) in state.iter().enumerate() {
                acc += Acc::from(self.out_w[o * s + j]) * Acc::from(sj);
            }
            *slot = sat_i32(round_shift_half_away(acc, crate::limits::LEARNED_WEIGHT_Q));
        }
    }

    #[inline]
    fn step(&self, state: &mut [i32], x_hat: &[i32]) {
        let c = usize::from(self.channels);
        let s = usize::from(self.state_dim);
        let mut next = vec![0i32; s];
        for (j, slot) in next.iter_mut().enumerate() {
            let mut acc: Acc = Acc::from(self.rec_b[j]);
            for (k, &sk) in state.iter().enumerate() {
                acc += Acc::from(self.rec_w[j * s + k]) * Acc::from(sk);
            }
            for (i, &xi) in x_hat.iter().enumerate() {
                acc += Acc::from(self.in_w[j * c + i]) * Acc::from(xi);
            }
            let v = self.activation.apply(acc);
            *slot = sat_i32(round_shift_half_away(v, crate::limits::LEARNED_WEIGHT_Q));
        }
        state.copy_from_slice(&next);
    }

    /// Reconstruct `[start, start+len)` exactly from dense residual and source
    /// checkpoints are used to bound replay.
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
            return Err(Error::malformed("stateful residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("stateful range overflows"))?;
        if end > frames {
            return Err(Error::malformed("stateful range exceeds the extent"));
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        // Nearest checkpoint at or before `start`.
        let cp = self
            .checkpoints
            .iter()
            .rev()
            .find(|cp| cp.frame as usize <= start)
            .cloned();
        let (mut from, mut state) = match cp {
            Some(cp) => (cp.frame as usize, cp.state),
            None => (0usize, self.initial_state()),
        };
        if from > end {
            from = 0;
            state = self.initial_state();
        }
        let mut out = vec![0i32; len * c];
        let mut h = vec![0i32; c];
        let mut x_hat = vec![0i32; c];
        for t in from..end {
            self.emit(&state, &mut h);
            for i in 0..c {
                x_hat[i] = sat_i32(Acc::from(h[i]) + Acc::from(residual[t * c + i]));
                if t >= start {
                    out[(t - start) * c + i] = x_hat[i];
                }
            }
            self.step(&mut state, &x_hat);
        }
        Ok(out)
    }

    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if source.len() != frames * c {
            return Err(Error::malformed("stateful source length mismatch"));
        }
        let mut h_all = vec![0i32; frames * c];
        let mut state = self.initial_state();
        let mut h = vec![0i32; c];
        let mut x_hat = vec![0i32; c];
        for t in 0..frames {
            self.emit(&state, &mut h);
            for i in 0..c {
                h_all[t * c + i] = h[i];
                x_hat[i] = source[t * c + i];
            }
            self.step(&mut state, &x_hat);
        }
        Ok(h_all)
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        let cp = self
            .checkpoints
            .iter()
            .rev()
            .find(|cp| cp.frame as usize <= start)
            .map(|cp| cp.frame as usize)
            .unwrap_or(0);
        start.saturating_sub(cp)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(2); // kind 2 = stateful
        out.push(self.channels);
        out.extend_from_slice(&self.state_dim.to_le_bytes());
        out.extend_from_slice(&self.checkpoint_interval.to_le_bytes());
        out.extend_from_slice(&(self.checkpoints.len() as u32).to_le_bytes());
        for w in &self.out_w {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for b in &self.out_b {
            out.extend_from_slice(&b.to_le_bytes());
        }
        for w in &self.rec_w {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for b in &self.rec_b {
            out.extend_from_slice(&b.to_le_bytes());
        }
        for w in &self.in_w {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&self.activation.canonical_bytes());
        for cp in &self.checkpoints {
            out.extend_from_slice(&cp.frame.to_le_bytes());
            for v in &cp.state {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<StatefulPredictor> {
        let mut r = crate::learned::serialization::Reader::new(bytes);
        let kind = r.u8()?;
        if kind != 2 {
            return Err(Error::malformed("stateful predictor kind mismatch"));
        }
        let channels = r.u8()?;
        let state_dim = r.u16()?;
        let checkpoint_interval = r.u32()?;
        let cp_count = r.u32()? as usize;
        if cp_count > crate::limits::MAX_LEARNED_CHECKPOINTS as usize {
            return Err(Error::limit("stateful checkpoint count exceeds the bound"));
        }
        let c = usize::from(channels);
        let s = usize::from(state_dim);
        if c == 0 || s == 0 {
            return Err(Error::malformed("stateful predictor has a zero dimension"));
        }
        let mut out_w = Vec::with_capacity(c * s);
        for _ in 0..c * s {
            out_w.push(r.i16()?);
        }
        let mut out_b = Vec::with_capacity(c);
        for _ in 0..c {
            out_b.push(r.i32()?);
        }
        let mut rec_w = Vec::with_capacity(s * s);
        for _ in 0..s * s {
            rec_w.push(r.i16()?);
        }
        let mut rec_b = Vec::with_capacity(s);
        for _ in 0..s {
            rec_b.push(r.i32()?);
        }
        let mut in_w = Vec::with_capacity(s * c);
        for _ in 0..s * c {
            in_w.push(r.i16()?);
        }
        let activation = Activation::from_canonical_bytes(&mut r)?;
        let mut checkpoints = Vec::with_capacity(cp_count.min(4096));
        for _ in 0..cp_count {
            let frame = r.u64()?;
            let mut state = Vec::with_capacity(s);
            for _ in 0..s {
                state.push(r.i32()?);
            }
            checkpoints.push(StateCheckpoint { frame, state });
        }
        r.finish()?;
        let p = StatefulPredictor {
            channels,
            state_dim,
            out_w,
            out_b,
            rec_w,
            rec_b,
            in_w,
            activation,
            checkpoint_interval,
            checkpoints,
        };
        p.validate()?;
        Ok(p)
    }

    /// Derive canonical checkpoints by a deterministic closed-loop pass over a
    /// source (used by the trainer; the encoded checkpoints are then frozen).
    pub fn derive_checkpoints(
        &self,
        source: &[i32],
        frames: usize,
    ) -> Result<Vec<StateCheckpoint>> {
        if self.checkpoint_interval == 0 {
            return Ok(Vec::new());
        }
        let c = usize::from(self.channels);
        if source.len() != frames * c {
            return Err(Error::malformed("stateful source length mismatch"));
        }
        let h_all = self.hypothesis_all_from_source(source, frames)?;
        let _ = &h_all;
        let mut state = self.initial_state();
        let mut cps = Vec::new();
        let interval = self.checkpoint_interval as usize;
        let mut x_hat = vec![0i32; c];
        for t in 0..frames {
            if t > 0 && t % interval == 0 {
                cps.push(StateCheckpoint {
                    frame: t as u64,
                    state: state.clone(),
                });
            }
            // The decoder's state update consumes the *reconstructed* sample,
            // which equals the source exactly once the residual closes.
            x_hat.copy_from_slice(&source[t * c..t * c + c]);
            self.step(&mut state, &x_hat);
        }
        Ok(cps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::{quantize_bias, quantize_weight};

    /// A predictor whose output is the previous reconstructed sample: state[0]
    /// holds X_hat[t-1], and the recurrence stores the current input.
    fn delay_predictor(interval: u32) -> StatefulPredictor {
        StatefulPredictor {
            channels: 1,
            state_dim: 1,
            out_w: vec![quantize_weight(1.0)],
            out_b: vec![quantize_bias(0.0)],
            rec_w: vec![0],
            rec_b: vec![0],
            in_w: vec![quantize_weight(1.0)],
            activation: Activation::Identity,
            checkpoint_interval: interval,
            checkpoints: Vec::new(),
        }
    }

    #[test]
    fn closed_loop_reconstruction_is_exact_and_seek_matches() {
        let mut p = delay_predictor(16);
        let x: Vec<i32> = (0..200).map(|i| ((i * 661) % 5000) - 2000).collect();
        let h = p.hypothesis_all_from_source(&x, 200).unwrap();
        let residual: Vec<i32> = (0..200).map(|t| x[t] - h[t]).collect();
        assert_eq!(p.evaluate_range(&residual, 200, 0, 200).unwrap(), x);
        for start in [0usize, 1, 99, 199] {
            assert_eq!(
                p.evaluate_range(&residual, 200, start, 200 - start)
                    .unwrap(),
                &x[start..]
            );
        }
        // Checkpoints bound replay and preserve equality.
        let cps = p.derive_checkpoints(&x, 200).unwrap();
        assert!(!cps.is_empty());
        p.checkpoints = cps;
        p.validate().unwrap();
        assert_eq!(p.evaluate_range(&residual, 200, 0, 200).unwrap(), x);
        assert_eq!(
            p.evaluate_range(&residual, 200, 150, 50).unwrap(),
            &x[150..]
        );
        assert!(p.replay_frames(150) < 150);
    }

    #[test]
    fn canonical_round_trip_and_bounds() {
        let mut p = delay_predictor(16);
        let x: Vec<i32> = (0..100).map(|i| i * 3).collect();
        p.checkpoints = p.derive_checkpoints(&x, 100).unwrap();
        p.validate().unwrap();
        let bytes = p.canonical_bytes();
        assert_eq!(StatefulPredictor::from_canonical_bytes(&bytes).unwrap(), p);
        assert!(StatefulPredictor::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut bomb = bytes.clone();
        bomb[8..12].copy_from_slice(&u32::MAX.to_le_bytes()); // checkpoint count
        assert!(StatefulPredictor::from_canonical_bytes(&bomb).is_err());
    }
}

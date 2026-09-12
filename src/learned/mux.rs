//! Hard per-microgroup expert selection (Phase 6 mechanism 4,
//! `SampleExpertMux`; experimental profile Exp3).
//!
//! A stationary or single-adaptive predictor is punished when the *best
//! predictor* changes across a recording. `SampleExpertMux` maintains several
//! decoder-synchronized backward-adaptive experts and, for every microgroup of
//! `group_frames` samples, selects **one** of them:
//!
//! ```text
//! H[t]     = H_{j*(t)}[t]                     (hard selection, never a mixture)
//! X_hat[t] = sat_i32(H[t] + R[t])
//! ```
//!
//! The selector `j*(t)` is transmitted as a packed bit field (one entry per
//! microgroup), so the side stream is a couple of bits per group. Crucially,
//! **every** expert is stepped from the reconstructed value `X_hat[t]`, not only
//! the selected one, so all experts stay decoder-synchronized without any
//! transmitted state. Because each expert's update depends only on its own
//! prediction and the common `X_hat`, the experts' trajectories are independent
//! of the selector — which makes the per-group choice exactly optimal for the
//! fixed expert set (no path dependence, no search).
//!
//! Model kind tag `20`. Mono-first. The experts are the same frozen integer
//! sign-sign LMS predictors as the backward-adaptive family
//! ([`crate::learned::adaptive`]).

use crate::error::{Error, Result};
use crate::learned::adaptive::AdaptivePredictor;
use crate::learned::arithmetic::{Acc, sat_i32};

/// A hard-selection mux over backward-adaptive experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertMuxPredictor {
    /// Channel count (this family is mono-only in this build).
    pub channels: u8,
    /// Microgroup length in frames (`>= 1`).
    pub group_frames: u32,
    /// The expert set (`2..=MAX_EXPERTS`), each a whole-extent adaptive model.
    pub experts: Vec<AdaptivePredictor>,
    /// One selector per microgroup, each in `0..experts.len()`.
    pub selectors: Vec<u8>,
}

/// Runtime state of the mux.
struct MuxState {
    history: Vec<i32>,
    weights: Vec<Vec<i32>>,
    bias: Vec<i32>,
    preds: Vec<i32>,
}

impl ExpertMuxPredictor {
    /// Maximum experts in one mux.
    pub const MAX_EXPERTS: usize = 4;

    /// Bits per selector for this expert count.
    pub fn bits_per_selector(&self) -> u32 {
        let n = self.experts.len() as u32;
        if n <= 1 {
            0
        } else {
            32 - (n - 1).leading_zeros()
        }
    }

    /// Number of microgroups covering `frames`.
    pub fn group_count(&self, frames: usize) -> usize {
        frames.div_ceil(self.group_frames.max(1) as usize).max(1)
    }

    fn max_taps(&self) -> usize {
        self.experts
            .iter()
            .map(|e| usize::from(e.taps))
            .max()
            .unwrap_or(1)
    }

    /// Validate shape, experts, selectors and ceilings.
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed("expert mux is mono-only in this build"));
        }
        if self.experts.len() < 2 || self.experts.len() > Self::MAX_EXPERTS {
            return Err(Error::limit("expert mux expert count out of range"));
        }
        if self.group_frames == 0 || self.group_frames > crate::limits::MAX_LEARNED_BLOCK_FRAMES {
            return Err(Error::limit("expert mux group size exceeds the bound"));
        }
        for e in &self.experts {
            e.validate()?;
            if e.block_frames.is_some() {
                return Err(Error::malformed(
                    "expert mux experts must span the whole extent",
                ));
            }
        }
        if self.selectors.is_empty() {
            return Err(Error::malformed("expert mux has no selectors"));
        }
        let k = self.experts.len() as u8;
        if self.selectors.iter().any(|&s| s >= k) {
            return Err(Error::malformed("expert mux selector out of range"));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.max_taps() as u32)
    }

    pub fn ops_per_sample(&self) -> u64 {
        self.experts.iter().map(|e| u64::from(e.taps) * 3 + 1).sum()
    }

    pub fn state_bytes(&self) -> u64 {
        self.experts.iter().map(|e| u64::from(e.taps) * 2).sum()
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        // Every expert is adaptive from frame zero.
        start
    }

    fn init_state(&self) -> MuxState {
        let taps = self.max_taps();
        MuxState {
            history: vec![0i32; taps],
            weights: self
                .experts
                .iter()
                .map(|e| e.init_weights.iter().map(|&w| i32::from(w)).collect())
                .collect(),
            bias: self.experts.iter().map(|e| e.init_bias).collect(),
            preds: vec![0i32; self.experts.len()],
        }
    }

    /// Compute every expert's prediction, return the selected one.
    #[inline]
    fn predict_step(&self, st: &mut MuxState, t: usize) -> i32 {
        let group = (t / self.group_frames.max(1) as usize).min(self.selectors.len() - 1);
        let sel = self.selectors[group] as usize;
        let mut chosen = 0i32;
        for j in 0..self.experts.len() {
            let taps = usize::from(self.experts[j].taps);
            let h =
                AdaptivePredictor::predict(&st.history[..taps], &st.weights[j], st.bias[j], taps);
            st.preds[j] = h;
            if j == sel {
                chosen = h;
            }
        }
        chosen
    }

    /// Step **every** expert from the reconstructed value and advance history.
    #[inline]
    fn commit(&self, st: &mut MuxState, x_hat: i32) {
        for j in 0..self.experts.len() {
            let taps = usize::from(self.experts[j].taps);
            let e = x_hat.wrapping_sub(st.preds[j]);
            AdaptivePredictor::update(
                &mut st.weights[j],
                e,
                &st.history[..taps],
                self.experts[j].step,
            );
        }
        let taps = st.history.len();
        if taps > 1 {
            st.history.copy_within(0..taps - 1, 1);
        }
        st.history[0] = x_hat;
    }

    /// Hypothesis over a whole extent, closed-loop with the source as `X_hat`.
    pub fn hypothesis_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("expert mux source length mismatch"));
        }
        if self.selectors.len() != self.group_count(frames) {
            return Err(Error::malformed("expert mux selector count mismatch"));
        }
        let mut st = self.init_state();
        let mut out = vec![0i32; frames];
        for t in 0..frames {
            let h = self.predict_step(&mut st, t);
            out[t] = h;
            self.commit(&mut st, source[t]);
        }
        Ok(out)
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
            return Err(Error::malformed("expert mux residual length mismatch"));
        }
        if self.selectors.len() != self.group_count(frames) {
            return Err(Error::malformed("expert mux selector count mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("expert mux range overflows"))?;
        if end > frames {
            return Err(Error::malformed("expert mux range exceeds the extent"));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        let mut st = self.init_state();
        for (t, &r) in residual.iter().enumerate() {
            let h = self.predict_step(&mut st, t);
            let x = sat_i32(Acc::from(h) + Acc::from(r));
            if t >= start && t < end {
                out[t - start] = x;
            }
            self.commit(&mut st, x);
        }
        Ok(out)
    }

    /// Canonical bytes:
    /// `kind(20) || channels || group_frames(u32) || expert_count(u8) ||
    ///  [expert_len(u32) || expert_bytes] … || selector_count(u32) || packed`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(20);
        out.push(self.channels);
        out.extend_from_slice(&self.group_frames.to_le_bytes());
        out.push(self.experts.len() as u8);
        for e in &self.experts {
            let eb = e.canonical_bytes();
            out.extend_from_slice(&(eb.len() as u32).to_le_bytes());
            out.extend_from_slice(&eb);
        }
        out.extend_from_slice(&(self.selectors.len() as u32).to_le_bytes());
        let bits = self.bits_per_selector();
        let bytes = ((self.selectors.len() as u64 * u64::from(bits)).div_ceil(8)) as usize;
        let mut packed = vec![0u8; bytes];
        let mut bit = 0usize;
        for &s in &self.selectors {
            for b in (0..bits).rev() {
                if (s >> b) & 1 == 1 {
                    packed[bit / 8] |= 1 << (7 - (bit % 8));
                }
                bit += 1;
            }
        }
        out.extend_from_slice(&packed);
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<ExpertMuxPredictor> {
        if bytes.len() < 2 + 4 + 1 || bytes[0] != 20 {
            return Err(Error::malformed("expert mux model header mismatch"));
        }
        let channels = bytes[1];
        let group_frames = u32::from_le_bytes(bytes[2..6].try_into().unwrap());
        let count = bytes[6] as usize;
        if !(2..=ExpertMuxPredictor::MAX_EXPERTS).contains(&count) {
            return Err(Error::limit("expert mux expert count out of range"));
        }
        let mut at = 7usize;
        let mut experts = Vec::with_capacity(count);
        for _ in 0..count {
            let len = u32::from_le_bytes(
                bytes
                    .get(at..at + 4)
                    .ok_or_else(|| Error::malformed("expert mux model is truncated"))?
                    .try_into()
                    .unwrap(),
            ) as usize;
            at += 4;
            let eb = bytes
                .get(at..at + len)
                .ok_or_else(|| Error::malformed("expert mux model is truncated"))?;
            at += len;
            experts.push(AdaptivePredictor::from_canonical_bytes(eb)?);
        }
        let selector_count = u32::from_le_bytes(
            bytes
                .get(at..at + 4)
                .ok_or_else(|| Error::malformed("expert mux model is truncated"))?
                .try_into()
                .unwrap(),
        ) as usize;
        at += 4;
        if selector_count == 0 || selector_count as u64 > crate::limits::MAX_OBJECT_FRAMES {
            return Err(Error::limit("expert mux selector count exceeds the bound"));
        }
        let bits = if count <= 1 {
            0
        } else {
            32 - (count as u32 - 1).leading_zeros()
        };
        let need = ((selector_count as u64 * u64::from(bits)).div_ceil(8)) as usize;
        let packed = bytes
            .get(at..at + need)
            .ok_or_else(|| Error::malformed("expert mux model is truncated"))?;
        at += need;
        if at != bytes.len() {
            return Err(Error::malformed("expert mux model has trailing bytes"));
        }
        let mut selectors = Vec::with_capacity(selector_count);
        let mut bit = 0usize;
        for _ in 0..selector_count {
            let mut s = 0u8;
            for _ in 0..bits {
                let v = (packed[bit / 8] >> (7 - (bit % 8))) & 1;
                s = (s << 1) | v;
                bit += 1;
            }
            selectors.push(s);
        }
        let m = ExpertMuxPredictor {
            channels,
            group_frames,
            experts,
            selectors,
        };
        m.validate()?;
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::model::LearnedModel;
    use crate::learned::object::LearnedObject;

    fn expert(taps: u16, step: i16) -> AdaptivePredictor {
        AdaptivePredictor {
            channels: 1,
            taps,
            init_weights: vec![0; usize::from(taps)],
            init_bias: 0,
            step,
            block_frames: None,
        }
    }

    fn signal(n: usize, switches: &[(usize, f64)], seed: u64) -> Vec<i32> {
        let mut st = seed | 1;
        let mut x = 0i64;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = switches
                .iter()
                .rev()
                .find(|(at, _)| i >= *at)
                .map(|(_, a)| *a)
                .unwrap_or(0.9);
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (((st >> 33) & 0xffff) as i64 - 32768) >> 4;
            x = ((a * x as f64).round() as i64) + noise;
            x = x.clamp(-(1 << 22), 1 << 22);
            out.push(x as i32);
        }
        out
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let m = ExpertMuxPredictor {
            channels: 1,
            group_frames: 64,
            experts: vec![expert(4, 8), expert(8, 1)],
            selectors: (0..16).map(|i| (i % 2) as u8).collect(),
        };
        let b = m.canonical_bytes();
        let back = ExpertMuxPredictor::from_canonical_bytes(&b).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn mux_closes_exactly_and_selects_both_experts() {
        let n = 4096usize;
        let x = signal(n, &[(0, 0.95), (2048, -0.6)], 7);
        // Selectors chosen greedily by measured proxy.
        let experts = vec![expert(4, 8), expert(8, 1)];
        let selectors = greedy_selectors(&x, &experts, 128);
        assert!(selectors.contains(&0) && selectors.contains(&1));
        let m = ExpertMuxPredictor {
            channels: 1,
            group_frames: 128,
            experts,
            selectors,
        };
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::ExpertMux(m),
            1,
            n as u64,
            48_000,
            Vec::new(),
            &x,
        )
        .unwrap();
        assert!(o.verify(&x));
        assert_eq!(o.materialize_range(2000, 40).unwrap(), &x[2000..2040]);
    }

    #[test]
    fn decode_rejects_empty_selector_stream() {
        let m = ExpertMuxPredictor {
            channels: 1,
            group_frames: 64,
            experts: vec![expert(4, 8), expert(8, 1)],
            selectors: Vec::new(),
        };
        assert!(m.validate().is_err());
    }

    // A tiny local greedy selector chooser used only by the tests; the fitter in
    // `train::mux` uses the same rule with the shared proxy.
    fn greedy_selectors(source: &[i32], experts: &[AdaptivePredictor], group: usize) -> Vec<u8> {
        let n = source.len();
        let mut hs = Vec::new();
        for e in experts {
            hs.push(e.hypothesis_all_from_source(source, n).unwrap());
        }
        let groups = n.div_ceil(group.max(1));
        let mut out = Vec::with_capacity(groups);
        let mut g = 0usize;
        while g * group < n.max(1) {
            let a = g * group;
            let b = (a + group).min(n);
            let mut best = (0usize, u64::MAX);
            for (j, h) in hs.iter().enumerate() {
                let r: Vec<i32> = (a..b).map(|t| source[t] - h[t]).collect();
                let cost = crate::learned::carousel::proxy_residual_bits(&r);
                if cost < best.1 {
                    best = (j, cost);
                }
            }
            out.push(best.0 as u8);
            g += 1;
        }
        out
    }
}

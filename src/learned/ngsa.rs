//! Natural-gradient backward adaptation (fourth-pass **Seal A0**, NARU-inspired).
//!
//! The existing [`crate::learned::adaptive::AdaptivePredictor`] is a frozen
//! integer **sign-sign LMS** update. It is honest but it converged too slowly on
//! short windows, which was one of the family's recorded negatives. NARU
//! (Mineo & Shouno, 2022) attacks exactly that convergence problem with a
//! natural-gradient sign algorithm and an O(p) AR-based preconditioner.
//!
//! This is a clean-room fixed-point realisation of that idea. The update is
//! preconditioned by an O(p) approximation to the inverse input autocorrelation:
//! an AR(1) input model with lag-1 correlation `rho` (a decoder-visible EMA) has
//! an inverse autocorrelation that is tridiagonal, so the natural-gradient
//! direction is
//!
//! ```text
//! v_k = x_k - rho · (x_{k-1} + x_{k+1})              (only already-reconstructed x)
//! w_k += step · e · v_k / max(‖x‖², floor)           (clamped to i16)
//! ```
//!
//! which is O(p) per sample and needs only reconstructed history, exactly like
//! the sign-sign family. The predictor is a candidate family, never semantic
//! authority: it closes `X_hat = sat_i32(H + R)` sample for sample or the
//! object is refused. Model kind tag `18`. Mono-first.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, round_shift_half_away, sat_i32};

/// A natural-gradient backward-adaptive causal predictor (Seal A0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgsaPredictor {
    pub channels: u8,
    pub taps: u16,
    /// Initial Q12 weights, lag order (`lag 1` first).
    pub init_weights: Vec<i16>,
    pub init_bias: i32,
    /// Learning rate in Q12 units.
    pub step: i16,
    /// EMA shift of the lag-1 correlation estimate.
    pub rho_shift: u8,
    pub block_frames: Option<u32>,
}

#[inline]
fn precondition(history: &[i32], k: usize, taps: usize, rho: i64) -> i64 {
    // Tridiagonal inverse of the AR(1) correlation matrix (scale 1/(1-rho^2)
    // dropped, since the step is normalised): interior diagonal `1+rho^2`,
    // corner diagonal `1`, off-diagonals `-rho`.
    let q = 1i64 << 12;
    let rho2 = (rho * rho) >> 12;
    let xk = i64::from(history[k]);
    let xm = if k >= 1 { i64::from(history[k - 1]) } else { 0 };
    let xp = if k + 1 < taps {
        i64::from(history[k + 1])
    } else {
        0
    };
    let diag = if k == 0 || k + 1 == taps { q } else { q + rho2 };
    ((diag * xk) >> 12) - ((rho * (xm + xp)) >> 12)
}

/// One natural-gradient step (normalized by the preconditioned energy `x·v`).
///
/// `v = x − rho·(x₋₁+x₊₁)` is the O(p) AR(1) inverse-autocorrelation direction;
/// `v` is built from already-reconstructed history only, and the step is bounded
/// so a hostile or near-silent window can never blow the weights up.
#[inline]
#[allow(clippy::needless_range_loop)]
fn adapt(weights: &mut [i32], history: &[i32], e: i64, step: i16, rho: i64) {
    let taps = history.len();
    let mut dot: i128 = 0;
    for k in 0..taps {
        dot += i128::from(history[k]) * i128::from(precondition(history, k, taps, rho));
    }
    let denom = dot.abs().max(1 << 12);
    let s = i128::from(step) * i128::from(e);
    for k in 0..taps {
        let v = precondition(history, k, taps, rho);
        let dw = ((s * i128::from(v)) / denom).clamp(-(1 << 13), 1 << 13) as i64;
        let next = i64::from(weights[k]) + dw;
        weights[k] = next.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i32;
    }
}

impl NgsaPredictor {
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed("ngsa predictor is mono-only"));
        }
        if self.taps == 0 || u32::from(self.taps) > crate::limits::MAX_LEARNED_ADAPTIVE_COEFFS {
            return Err(Error::limit("ngsa tap count out of range"));
        }
        if self.init_weights.len() != usize::from(self.taps) {
            return Err(Error::malformed("ngsa weight count mismatch"));
        }
        if self.step <= 0 {
            return Err(Error::malformed("ngsa step must be positive"));
        }
        if self.rho_shift == 0 || self.rho_shift > 20 {
            return Err(Error::malformed("ngsa rho shift out of range"));
        }
        if !crate::learned::arithmetic::accumulator_is_safe(u32::from(self.taps)) {
            return Err(Error::limit("ngsa accumulator bound exceeded"));
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit("ngsa block size exceeds the bound"));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.taps)
    }

    pub fn ops_per_sample(&self) -> u64 {
        u64::from(self.taps) * 5 + 4
    }

    pub fn state_bytes(&self) -> u64 {
        u64::from(self.taps) * 2 + 24
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    pub fn replay_frames(&self, start: usize) -> usize {
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
            return Err(Error::malformed("ngsa residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("ngsa range overflows"))?;
        if end > frames {
            return Err(Error::malformed("ngsa range exceeds the extent"));
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
            let mut num: i64 = 0;
            let mut den: i64 = 0;
            let mut rho: i64 = 0;
            let mut weights: Vec<i32> = self.init_weights.iter().map(|&w| i32::from(w)).collect();
            for t in from..to {
                let h = Self::predict(&history, &weights, bias, taps);
                let x = sat_i32(Acc::from(h) + Acc::from(residual[t]));
                if t >= start && t < end {
                    out[t - start] = x;
                }
                let e = i64::from(x) - i64::from(h);
                adapt(&mut weights, &history, e, self.step, rho);
                Self::update_rho(&history, &mut num, &mut den, &mut rho, self.rho_shift);
                if taps > 1 {
                    history.copy_within(0..taps - 1, 1);
                }
                history[0] = x;
            }
        }
        Ok(out)
    }

    #[inline]
    fn update_rho(history: &[i32], num: &mut i64, den: &mut i64, rho: &mut i64, shift: u8) {
        if history.len() < 2 {
            return;
        }
        let a = i64::from(history[0]);
        let b = i64::from(history[1]);
        *num += ((a * b) - *num) >> shift;
        *den += ((a * a) - *den) >> shift;
        *rho = if *den > 0 {
            (((i128::from(*num)) << 12) / i128::from(*den)).clamp(-4096, 4096) as i64
        } else {
            0
        };
    }

    /// Hypothesis over a whole extent, open-loop from a source.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("ngsa source length mismatch"));
        }
        let taps = usize::from(self.taps);
        let mut out = vec![0i32; frames];
        let mut history = vec![0i32; taps];
        let bias = self.init_bias;
        for (from, to) in self.ranges(frames) {
            history.fill(0);
            let mut num: i64 = 0;
            let mut den: i64 = 0;
            let mut rho: i64 = 0;
            let mut weights: Vec<i32> = self.init_weights.iter().map(|&w| i32::from(w)).collect();
            for t in from..to {
                let h = Self::predict(&history, &weights, bias, taps);
                out[t] = h;
                let e = i64::from(source[t]) - i64::from(h);
                adapt(&mut weights, &history, e, self.step, rho);
                Self::update_rho(&history, &mut num, &mut den, &mut rho, self.rho_shift);
                if taps > 1 {
                    history.copy_within(0..taps - 1, 1);
                }
                history[0] = source[t];
            }
        }
        Ok(out)
    }

    /// Canonical bytes:
    /// `kind(18) || channels || taps(u16) || step(i16) || rho_shift(u8) ||
    ///  weights || bias(i32) || block(u32)`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 1 + 2 + 2 + 1 + self.init_weights.len() * 2 + 4 + 4);
        out.push(18);
        out.push(self.channels);
        out.extend_from_slice(&self.taps.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        out.push(self.rho_shift);
        for w in &self.init_weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&self.init_bias.to_le_bytes());
        out.extend_from_slice(&self.block_frames.unwrap_or(0).to_le_bytes());
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<NgsaPredictor> {
        if bytes.len() < 15 || bytes[0] != 18 {
            return Err(Error::malformed("ngsa model header mismatch"));
        }
        let channels = bytes[1];
        let taps = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let step = i16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let rho_shift = bytes[6];
        let t = usize::from(taps);
        let expect = 1 + 1 + 2 + 2 + 1 + t * 2 + 4 + 4;
        if bytes.len() != expect {
            return Err(Error::malformed("ngsa model length mismatch"));
        }
        let mut at = 7usize;
        let mut init_weights = Vec::with_capacity(t);
        for _ in 0..t {
            init_weights.push(i16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()));
            at += 2;
        }
        let init_bias = i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        at += 4;
        let block = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let p = NgsaPredictor {
            channels,
            taps,
            init_weights,
            init_bias,
            step,
            rho_shift,
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

    fn drifting_ar2(n: usize, seed: f64) -> Vec<i32> {
        let mut x = vec![0i32; 2];
        for t in 2..n {
            let a = 0.5 + 0.15 * ((t / 400) as f64 * 0.1) + 0.001 * seed;
            let v = (x[t - 1] as f64 * a + x[t - 2] as f64 * 0.2 + ((t % 9) as f64 - 4.0) * 4.0)
                .round() as i64;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        x
    }

    #[test]
    fn ngsa_closes_exactly_and_adapts() {
        let x = drifting_ar2(3000, 0.0);
        let p = NgsaPredictor {
            channels: 1,
            taps: 2,
            init_weights: vec![2000, 800],
            init_bias: 0,
            step: 2048,
            rho_shift: 6,
            block_frames: None,
        };
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::Ngsa(p.clone()),
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
        let p = NgsaPredictor {
            channels: 1,
            taps: 3,
            init_weights: vec![100, -200, 300],
            init_bias: 7,
            step: 1024,
            rho_shift: 5,
            block_frames: Some(256),
        };
        let b = p.canonical_bytes();
        let back = NgsaPredictor::from_canonical_bytes(&b).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let mut p = NgsaPredictor {
            channels: 1,
            taps: 1,
            init_weights: vec![0],
            init_bias: 0,
            step: 1024,
            rho_shift: 6,
            block_frames: None,
        };
        assert!(p.validate().is_ok());
        p.step = 0;
        assert!(p.validate().is_err());
        p.step = 1024;
        p.rho_shift = 0;
        assert!(p.validate().is_err());
    }
}

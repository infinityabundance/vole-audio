//! Wasted-bits / common-factor wrapper (Report 3 mechanism 14, experimental
//! profile `vole.audio.learned.exp3`).
//!
//! When every sample of an intrinsic shares a power-of-two factor — as the frozen
//! **U1 s16 ingest** produces (`i32 = i16 << 16`, so every sample has 16 low zero
//! bits) — a predictor should operate on the quotients, not the scaled samples.
//! This wrapper models `q[t] = X[t] >> shift`, uses an inner model unchanged, and
//! scales the prediction back:
//!
//! ```text
//! H[t]     = sat_i32( inner_hypothesis(X >> shift)[t] << shift )
//! X_hat[t] = sat_i32( inner_evaluation(R >> shift)[t] << shift )
//! ```
//!
//! The exact residual is then `X - H = 2^shift·(q - H_q)`, a multiple of
//! `2^shift`; the Exp3 `FactorShift` residual codec strips that factor. The
//! wrapper is a pure integer transform and adds no approximation.
//!
//! Model kind tag `16`. Mono-first; the inner model must not itself require a
//! source and must not be another `Wasted` (no nesting).

use crate::error::{Error, Result};
use crate::learned::arithmetic::sat_i32;
use crate::learned::model::LearnedModel;

/// A predictor that models the `shift`-scaled quotients of the intrinsic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WastedPredictor {
    pub channels: u8,
    /// Right shift applied to the intrinsic before the inner model (`1..=31`).
    pub shift: u8,
    /// The inner hypothesis over the quotient domain.
    pub inner: Box<LearnedModel>,
}

impl WastedPredictor {
    /// Validate the shift, the non-nesting rule and the inner model.
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed(
                "wasted wrapper is mono-only in this build",
            ));
        }
        if self.shift == 0 || self.shift > 31 {
            return Err(Error::malformed("wasted shift out of range"));
        }
        if matches!(&*self.inner, LearnedModel::Wasted(_)) {
            return Err(Error::malformed("wasted wrappers may not nest"));
        }
        if self.inner.requires_source() {
            return Err(Error::malformed(
                "wasted wrapper cannot supply an inner source dependency",
            ));
        }
        self.inner.validate()?;
        if self.inner.channels() != self.channels {
            return Err(Error::malformed(
                "wasted inner model channels disagree with the wrapper",
            ));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        self.inner.receptive_field()
    }

    pub fn ops_per_sample(&self) -> u64 {
        self.inner.ops_per_sample().saturating_add(2)
    }

    pub fn state_bytes(&self) -> u64 {
        self.inner.state_bytes()
    }

    pub fn checkpoint_count(&self) -> u32 {
        self.inner.checkpoint_count()
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        self.inner.replay_frames(start)
    }

    /// Hypothesis over a whole extent, in the original (scaled) domain.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("wasted source length mismatch"));
        }
        let q: Vec<i32> = source.iter().map(|&v| v >> self.shift).collect();
        let hq = self.inner.hypothesis_from_source(&q, frames)?;
        Ok(hq
            .into_iter()
            .map(|v| sat_i32((i64::from(v)) << self.shift))
            .collect())
    }

    /// Reconstruct `[start, start+len)` in the original domain.
    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        if residual.len() != frames {
            return Err(Error::malformed("wasted residual length mismatch"));
        }
        let qr: Vec<i32> = residual.iter().map(|&v| v >> self.shift).collect();
        let q = self.inner.evaluate_range(&qr, frames, start, len)?;
        Ok(q.into_iter()
            .map(|v| sat_i32((i64::from(v)) << self.shift))
            .collect())
    }

    /// Canonical bytes: `kind(16) || channels || shift || inner_len(u64) || inner`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let inner = self.inner.canonical_bytes();
        let mut out = Vec::with_capacity(1 + 1 + 1 + 8 + inner.len());
        out.push(16);
        out.push(self.channels);
        out.push(self.shift);
        out.extend_from_slice(&(inner.len() as u64).to_le_bytes());
        out.extend_from_slice(&inner);
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<WastedPredictor> {
        if bytes.len() < 11 || bytes[0] != 16 {
            return Err(Error::malformed("wasted model header mismatch"));
        }
        let channels = bytes[1];
        let shift = bytes[2];
        let inner_len = u64::from_le_bytes(bytes[3..11].try_into().unwrap()) as usize;
        if bytes.len() != 11 + inner_len {
            return Err(Error::malformed("wasted model length mismatch"));
        }
        let inner = LearnedModel::from_canonical_bytes(&bytes[11..])?;
        let w = WastedPredictor {
            channels,
            shift,
            inner: Box::new(inner),
        };
        w.validate()?;
        Ok(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::lpc::LpcPredictor;
    use crate::learned::object::LearnedObject;

    #[test]
    fn wasted_wrapper_closes_exactly_and_shrinks_the_residual() {
        // AR(2) in the U1 domain: every sample is a multiple of 2^16.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut q = vec![0i32, 0];
        for t in 2..4096 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let d = (s >> 40) as i64 % 2001 - 1000;
            let v = i64::from(q[t - 1]) / 2 + i64::from(q[t - 2]) / 4 + d;
            q.push(v.clamp(-20000, 20000) as i32);
        }
        let x: Vec<i32> = q.iter().map(|&v| v << 16).collect();
        let inner = LearnedModel::Lpc(LpcPredictor {
            channels: 1,
            order: 2,
            precision: 16,
            shift: 15,
            coeffs: vec![16384, 8192],
            block_frames: None,
        });
        let model = LearnedModel::Wasted(WastedPredictor {
            channels: 1,
            shift: 16,
            inner: Box::new(inner),
        });
        let o = LearnedObject::from_intrinsic_exp3(model, 1, 4096, 48_000, Vec::new(), &x).unwrap();
        assert!(o.verify(&x));
        assert_eq!(o.materialize_range(2000, 40).unwrap(), &x[2000..2040]);
        // The residual is a multiple of 2^16, so FactorShift should be selected.
        assert_eq!(o.residual_codec.name(), "factor_shift");
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let inner = LearnedModel::Lpc(LpcPredictor {
            channels: 1,
            order: 1,
            precision: 12,
            shift: 12,
            coeffs: vec![100],
            block_frames: None,
        });
        let w = WastedPredictor {
            channels: 1,
            shift: 16,
            inner: Box::new(inner),
        };
        let b = w.canonical_bytes();
        assert_eq!(WastedPredictor::from_canonical_bytes(&b).unwrap(), w);
    }
}

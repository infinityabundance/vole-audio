//! Dense short-term LPC predictor (Report 3, Seal S2), **experimental profile**
//! `vole.audio.learned.exp2`.
//!
//! Speech is the canonical dense all-pole (vocal-tract) problem, and FLAC-5's
//! advantage on the frozen real corpus is a *local, dense, apodized LPC*
//! pipeline. The Exp1/Exp2 dense family is an **order-4 ridge FIR over the whole
//! clip**, which is a materially different model class. This family closes that
//! mismatch while staying inside VOLE's exact residual-closure constitution:
//!
//! ```text
//! H[t]      = (Σ_{i=1..P} q_i · X_hat[t-i]) >> shift      (arithmetic)
//! X_hat[t]  = sat_i32(H[t] + R[t])
//! ```
//!
//! The stored predictor is an exact integer object (`order`, `precision`,
//! `shift`, integer QLP coefficients). Float LPC analysis is disposable
//! proposal machinery and never semantic authority.
//!
//! Model kind tag `12`. Mono-first.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, sat_i32};

/// A dense causal all-pole predictor with explicit precision and shift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LpcPredictor {
    pub channels: u8,
    /// Order `P` (`1..=MAX_LEARNED_TAPS`).
    pub order: u16,
    /// Declared coefficient precision in bits (informational; the coefficients
    /// are stored exactly as `i32`).
    pub precision: u8,
    /// Arithmetic right shift applied to the accumulator (0..=31).
    pub shift: u8,
    /// `P` integer QLP coefficients, lag order (`coeffs[0]` is lag 1).
    pub coeffs: Vec<i32>,
    /// Optional block-local reset (independent blocks).
    pub block_frames: Option<u32>,
}

impl LpcPredictor {
    /// Validate structure, ordering and the accumulator proof.
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::new(
                crate::error::Kind::Unsupported,
                "dense LPC is mono-first in this build",
            ));
        }
        if self.order == 0 || u32::from(self.order) > crate::limits::MAX_LEARNED_TAPS {
            return Err(Error::limit("LPC order out of range"));
        }
        if self.coeffs.len() != usize::from(self.order) {
            return Err(Error::malformed("LPC coefficient count mismatch"));
        }
        if self.precision == 0 || self.precision > 31 {
            return Err(Error::malformed("LPC precision out of range"));
        }
        if self.shift > 31 {
            return Err(Error::malformed("LPC shift out of range"));
        }
        // Exact accumulator proof from the *actual* coefficients: each product
        // is `|q|·|x| <= max_abs · 2^31` (`|x| <= 2^31` because history is a
        // canonical `i32`), and the `order`-term sum must stay well inside
        // `i64`. This is sound even though `precision` is only informational.
        let mut max_abs = 0u64;
        for &q in &self.coeffs {
            max_abs = max_abs.max(i64::from(q).unsigned_abs());
        }
        let bound = max_abs
            .saturating_mul(1u64 << 31)
            .saturating_mul(u64::from(self.order));
        if bound >= (1u64 << 62) {
            return Err(Error::limit("LPC accumulator bound exceeded"));
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit("LPC block size exceeds the bound"));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.order)
    }

    pub fn ops_per_sample(&self) -> u64 {
        u64::from(self.order) + 1
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    #[inline]
    fn predict(&self, history: &[i32]) -> i32 {
        let mut acc = Acc::from(0);
        for (i, &q) in self.coeffs.iter().enumerate() {
            acc += Acc::from(q) * Acc::from(history[i]);
        }
        sat_i32(acc >> self.shift)
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
            return Err(Error::malformed("LPC residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("LPC range overflows"))?;
        if end > frames {
            return Err(Error::malformed("LPC range exceeds the extent"));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        let p = usize::from(self.order);
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
            return Err(Error::malformed("LPC source length mismatch"));
        }
        let p = usize::from(self.order);
        let mut out = vec![0i32; frames];
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

    /// Canonical bytes:
    /// `kind(12) || channels || order(u16) || precision(u8) || shift(u8) ||
    ///  coeffs(i32*) || block(u32)`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 1 + 2 + 1 + 1 + self.coeffs.len() * 4 + 4);
        out.push(12);
        out.push(self.channels);
        out.extend_from_slice(&self.order.to_le_bytes());
        out.push(self.precision);
        out.push(self.shift);
        for &q in &self.coeffs {
            out.extend_from_slice(&q.to_le_bytes());
        }
        out.extend_from_slice(&self.block_frames.unwrap_or(0).to_le_bytes());
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<LpcPredictor> {
        if bytes.len() < 13 || bytes[0] != 12 {
            return Err(Error::malformed("LPC model header mismatch"));
        }
        let channels = bytes[1];
        let order = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let precision = bytes[4];
        let shift = bytes[5];
        let p = usize::from(order);
        let expect = 1 + 1 + 2 + 1 + 1 + p * 4 + 4;
        if bytes.len() != expect {
            return Err(Error::malformed("LPC model length mismatch"));
        }
        let mut at = 6usize;
        let mut coeffs = Vec::with_capacity(p);
        for _ in 0..p {
            coeffs.push(i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
            at += 4;
        }
        let block = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let lpc = LpcPredictor {
            channels,
            order,
            precision,
            shift,
            coeffs,
            block_frames: if block == 0 { None } else { Some(block) },
        };
        lpc.validate()?;
        Ok(lpc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::model::LearnedModel;
    use crate::learned::object::LearnedObject;

    #[test]
    fn lpc_closes_exactly_on_a_synthetic_ar2() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut x = vec![0i32, 0];
        for t in 2..4096 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let drive = (s >> 40) as i64 % 2001 - 1000;
            let v = i64::from(x[t - 1]) / 2 + i64::from(x[t - 2]) / 4 + drive;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let p = LpcPredictor {
            channels: 1,
            order: 2,
            precision: 15,
            shift: 15,
            coeffs: vec![16384, 8192],
            block_frames: None,
        };
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::Lpc(p.clone()),
            1,
            4096,
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
        let p = LpcPredictor {
            channels: 1,
            order: 3,
            precision: 13,
            shift: 13,
            coeffs: vec![100, -200, 300],
            block_frames: Some(4096),
        };
        let b = p.canonical_bytes();
        let back = LpcPredictor::from_canonical_bytes(&b).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn validation_rejects_bad_shapes() {
        let mut p = LpcPredictor {
            channels: 1,
            order: 2,
            precision: 15,
            shift: 15,
            coeffs: vec![1, 2],
            block_frames: None,
        };
        p.coeffs = vec![1];
        assert!(p.validate().is_err());
        p.coeffs = vec![1, 2];
        p.shift = 40;
        assert!(p.validate().is_err());
        p.shift = 15;
        p.channels = 2;
        assert!(p.validate().is_err());
    }
}

//! Pole-zero (ARMA) predictor with exact residual closure (Report 3, **Seal S8**,
//! experimental profile `vole.audio.learned.exp3`).
//!
//! FLAC's LPC is all-pole. Real speech — especially nasal sounds — also has
//! spectral zeros an all-pole model represents inefficiently. A lossless codec
//! can define an exact causal pole-zero predictor using **only decoder-visible
//! history**: the already reconstructed samples and the already decoded exact
//! residuals.
//!
//! ```text
//! H[t]     = ( Σ_i a_i · X_hat[t-i] + Σ_j b_j · R[t-j] ) >> shift
//! X_hat[t] = sat_i32(H[t] + R[t])
//! ```
//!
//! Both histories are available to the decoder, so no hidden state is required
//! and the closure stays exact. Model kind tag `15`. Mono-first.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, sat_i32};

/// The Q shift shared by the AR and MA coefficient vectors.
pub const PZ_SHIFT: u8 = 12;

/// The bounded `(p, q)` ladder this family searches.
pub const PZ_LADDER: [(u16, u16); 5] = [(4, 1), (6, 1), (8, 1), (6, 2), (8, 2)];

/// A causal pole-zero predictor with exact residual closure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoleZeroPredictor {
    /// Channel count (this family is mono-only: always `1`).
    pub channels: u8,
    /// AR order `p`.
    pub order_ar: u16,
    /// MA (residual-feedback) order `q`.
    pub order_ma: u16,
    /// Q shift of both coefficient vectors.
    pub shift: u8,
    /// `p` AR coefficients in Q`shift` (lag 1 first).
    pub ar: Vec<i32>,
    /// `q` MA coefficients in Q`shift` (lag 1 first).
    pub ma: Vec<i32>,
    /// Optional block-local reset.
    pub block_frames: Option<u32>,
}

impl PoleZeroPredictor {
    /// Validate structure and the accumulator proof.
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed(
                "pole-zero predictor is mono-only in this build",
            ));
        }
        if self.order_ar == 0 && self.order_ma == 0 {
            return Err(Error::limit("pole-zero predictor has no terms"));
        }
        if self.ar.len() != usize::from(self.order_ar)
            || self.ma.len() != usize::from(self.order_ma)
        {
            return Err(Error::malformed("pole-zero coefficient count mismatch"));
        }
        if u32::from(self.order_ar) + u32::from(self.order_ma) > crate::limits::MAX_LEARNED_TAPS {
            return Err(Error::limit("pole-zero order exceeds the bound"));
        }
        if self.shift > 31 {
            return Err(Error::malformed("pole-zero shift out of range"));
        }
        let mut max_abs = 0u64;
        for &c in self.ar.iter().chain(self.ma.iter()) {
            if !(i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&c) {
                return Err(Error::malformed("pole-zero coefficient out of i16 domain"));
            }
            max_abs = max_abs.max(i64::from(c).unsigned_abs());
        }
        let terms = u64::from(self.order_ar) + u64::from(self.order_ma);
        let bound = max_abs
            .saturating_mul(1u64 << 31)
            .saturating_mul(terms.max(1));
        if bound >= (1u64 << 62) {
            return Err(Error::limit("pole-zero accumulator bound exceeded"));
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit("pole-zero block size exceeds the bound"));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.order_ar).max(u64::from(self.order_ma))
    }

    pub fn ops_per_sample(&self) -> u64 {
        u64::from(self.order_ar) + u64::from(self.order_ma) + 1
    }

    pub fn state_bytes(&self) -> u64 {
        (self.order_ma as u64) * 4
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    #[inline]
    fn predict(&self, xh: &[i32], rr: &[i32]) -> i32 {
        let mut acc = Acc::from(0);
        for (i, &a) in self.ar.iter().enumerate() {
            acc += Acc::from(a) * Acc::from(xh[i]);
        }
        for (j, &b) in self.ma.iter().enumerate() {
            acc += Acc::from(b) * Acc::from(rr[j]);
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
            return Err(Error::malformed("pole-zero residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("pole-zero range overflows"))?;
        if end > frames {
            return Err(Error::malformed("pole-zero range exceeds the extent"));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        let p = usize::from(self.order_ar).max(1);
        let q = usize::from(self.order_ma).max(1);
        let mut xh = vec![0i32; p];
        let mut rr = vec![0i32; q];
        for (from, to) in self.ranges(frames) {
            xh.fill(0);
            rr.fill(0);
            for t in from..to {
                let h = self.predict(&xh, &rr);
                let x = sat_i32(Acc::from(h) + Acc::from(residual[t]));
                if t >= start && t < end {
                    out[t - start] = x;
                }
                if p > 1 {
                    xh.copy_within(0..p - 1, 1);
                }
                xh[0] = x;
                if q > 1 {
                    rr.copy_within(0..q - 1, 1);
                }
                rr[0] = residual[t];
            }
        }
        Ok(out)
    }

    /// Hypothesis over a whole extent from a source, computing the residual
    /// feedback progressively (this is the open-loop residual construction).
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("pole-zero source length mismatch"));
        }
        let mut out = vec![0i32; frames];
        let p = usize::from(self.order_ar).max(1);
        let q = usize::from(self.order_ma).max(1);
        let mut xh = vec![0i32; p];
        let mut rr = vec![0i32; q];
        for (from, to) in self.ranges(frames) {
            xh.fill(0);
            rr.fill(0);
            for t in from..to {
                let h = self.predict(&xh, &rr);
                out[t] = h;
                let r = (i64::from(source[t]) - i64::from(h)) as i32;
                if p > 1 {
                    xh.copy_within(0..p - 1, 1);
                }
                xh[0] = source[t];
                if q > 1 {
                    rr.copy_within(0..q - 1, 1);
                }
                rr[0] = r;
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
    /// `kind(15) || channels || order_ar(u16) || order_ma(u16) || shift(u8) ||
    ///  ar(i16*) || ma(i16*) || block(u32)`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(1 + 1 + 2 + 2 + 1 + (self.ar.len() + self.ma.len()) * 2 + 4);
        out.push(15);
        out.push(self.channels);
        out.extend_from_slice(&self.order_ar.to_le_bytes());
        out.extend_from_slice(&self.order_ma.to_le_bytes());
        out.push(self.shift);
        for &c in self.ar.iter().chain(self.ma.iter()) {
            out.extend_from_slice(&(c as i16).to_le_bytes());
        }
        out.extend_from_slice(&self.block_frames.unwrap_or(0).to_le_bytes());
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<PoleZeroPredictor> {
        if bytes.len() < 11 || bytes[0] != 15 {
            return Err(Error::malformed("pole-zero model header mismatch"));
        }
        let channels = bytes[1];
        let order_ar = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let order_ma = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let shift = bytes[6];
        let p = usize::from(order_ar);
        let q = usize::from(order_ma);
        let expect = 1 + 1 + 2 + 2 + 1 + (p + q) * 2 + 4;
        if bytes.len() != expect {
            return Err(Error::malformed("pole-zero model length mismatch"));
        }
        let mut at = 7usize;
        let mut ar = Vec::with_capacity(p);
        let mut ma = Vec::with_capacity(q);
        for _ in 0..p {
            ar.push(i32::from(i16::from_le_bytes(
                bytes[at..at + 2].try_into().unwrap(),
            )));
            at += 2;
        }
        for _ in 0..q {
            ma.push(i32::from(i16::from_le_bytes(
                bytes[at..at + 2].try_into().unwrap(),
            )));
            at += 2;
        }
        let block = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let m = PoleZeroPredictor {
            channels,
            order_ar,
            order_ma,
            shift,
            ar,
            ma,
            block_frames: if block == 0 { None } else { Some(block) },
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

    #[test]
    fn pole_zero_closes_exactly() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut x = vec![0i32, 0, 0];
        for t in 3..4096 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let d = (s >> 40) as i64 % 2001 - 1000;
            let v = i64::from(x[t - 1]) / 2 + i64::from(x[t - 2]) / 4 + d;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let p = PoleZeroPredictor {
            channels: 1,
            order_ar: 2,
            order_ma: 1,
            shift: PZ_SHIFT,
            ar: vec![(0.5 * f64::from(1u32 << PZ_SHIFT)) as i32, 0],
            ma: vec![(0.25 * f64::from(1u32 << PZ_SHIFT)) as i32],
            block_frames: None,
        };
        let o = LearnedObject::from_intrinsic_exp3(
            LearnedModel::PoleZero(p.clone()),
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
        let p = PoleZeroPredictor {
            channels: 1,
            order_ar: 3,
            order_ma: 2,
            shift: 12,
            ar: vec![100, -200, 300],
            ma: vec![-50, 60],
            block_frames: Some(4096),
        };
        let b = p.canonical_bytes();
        assert_eq!(PoleZeroPredictor::from_canonical_bytes(&b).unwrap(), p);
    }
}

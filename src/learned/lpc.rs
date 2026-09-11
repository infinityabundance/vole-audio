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
        if self.precision == 0 || self.precision > 16 {
            return Err(Error::malformed("LPC precision out of range"));
        }
        if self.shift > 31 {
            return Err(Error::malformed("LPC shift out of range"));
        }
        // Coefficients are stored at the declared precision (signed, including
        // sign bit), so every coefficient's magnitude must be < 2^(p-1).
        let lo = -(1i64 << (self.precision - 1));
        let hi = (1i64 << (self.precision - 1)) - 1;
        for &q in &self.coeffs {
            if i64::from(q) < lo || i64::from(q) > hi {
                return Err(Error::malformed(
                    "LPC coefficient does not fit the declared precision",
                ));
            }
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
    ///  packed_coeffs(ceil(order·precision/8)) || block(u32)`.
    ///
    /// Coefficients are packed at the declared `precision` bits each (signed,
    /// two's complement, MSB-first), so a low-precision predictor pays only for
    /// the bits it uses.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let packed = pack_signed(&self.coeffs, self.precision);
        let mut out = Vec::with_capacity(6 + packed.len() + 4);
        out.push(12);
        out.push(self.channels);
        out.extend_from_slice(&self.order.to_le_bytes());
        out.push(self.precision);
        out.push(self.shift);
        out.extend_from_slice(&packed);
        out.extend_from_slice(&self.block_frames.unwrap_or(0).to_le_bytes());
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<LpcPredictor> {
        if bytes.len() < 10 || bytes[0] != 12 {
            return Err(Error::malformed("LPC model header mismatch"));
        }
        let channels = bytes[1];
        let order = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let precision = bytes[4];
        let shift = bytes[5];
        let p = usize::from(order);
        if precision == 0 || precision > 16 {
            return Err(Error::malformed("LPC precision out of range"));
        }
        let packed_len = (p * usize::from(precision)).div_ceil(8);
        let expect = 6 + packed_len + 4;
        if bytes.len() != expect {
            return Err(Error::malformed("LPC model length mismatch"));
        }
        let coeffs = unpack_signed(&bytes[6..6 + packed_len], precision, p)?;
        let at = 6 + packed_len;
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

/// Pack signed values at `bits` each, MSB-first, into whole bytes.
pub fn pack_signed(values: &[i32], bits: u8) -> Vec<u8> {
    let bits = usize::from(bits);
    let total = values.len() * bits;
    let mut out = vec![0u8; total.div_ceil(8)];
    let mut pos = 0usize;
    for &v in values {
        let u = (v as u32) & ((1u32 << bits) - 1);
        for b in (0..bits).rev() {
            let bit = (u >> b) & 1;
            out[pos >> 3] |= (bit as u8) << (7 - (pos & 7));
            pos += 1;
        }
    }
    out
}

/// Unpack `count` signed values of `bits` each (MSB-first).
pub fn unpack_signed(bytes: &[u8], bits: u8, count: usize) -> Result<Vec<i32>> {
    let bits_usize = usize::from(bits);
    if bits == 0 || bits > 16 {
        return Err(Error::malformed("packed coefficient width out of range"));
    }
    if bytes.len() * 8 < count * bits_usize {
        return Err(Error::malformed("packed coefficient payload truncated"));
    }
    let mask = (1u32 << bits_usize) - 1;
    let sign = 1u32 << (bits_usize - 1);
    let mut out = Vec::with_capacity(count);
    let mut pos = 0usize;
    for _ in 0..count {
        let mut u = 0u32;
        for _ in 0..bits_usize {
            let bit = (bytes[pos >> 3] >> (7 - (pos & 7))) & 1;
            u = (u << 1) | u32::from(bit);
            pos += 1;
        }
        u &= mask;
        let v = if u & sign != 0 {
            (u | !mask) as i32
        } else {
            u as i32
        };
        out.push(v);
    }
    Ok(out)
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
            precision: 16,
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
    fn packed_coefficients_round_trip() {
        for precision in [5u8, 8, 12, 15, 16] {
            let limit = 1i64 << (precision - 1);
            let values: Vec<i32> = vec![
                0,
                (limit - 1) as i32,
                -(limit) as i32,
                ((limit / 2) - 1) as i32,
            ];
            let packed = pack_signed(&values, precision);
            let back = unpack_signed(&packed, precision, values.len()).unwrap();
            assert_eq!(back, values, "precision {precision}");
        }
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

//! Lattice / PARCOR predictor realisation (Report 3, **Seal S7**, experimental
//! profile `vole.audio.learned.exp3`).
//!
//! Direct-form LPC coefficients are not the only realisation of an all-pole
//! predictor. Speech coding has long used **reflection (PARCOR) coefficients**
//! because they are bounded (`|k| < 1`), represent stability directly, and often
//! quantise better than direct-form coefficients. This family stores the
//! quantised reflection coefficients and reconstructs the direct predictor with
//! a deterministic fixed-point Levinson-from-reflection recurrence, then predicts
//! in the usual way:
//!
//! ```text
//! a_0 = [2^shift]
//! a_m[j]   = a_{m-1}[j] - ((k_m · a_{m-1}[m-j]) >> shift),  j = 1..m-1
//! a_m[m]   = -k_m
//! H[t]     = (Σ_j (-a_m[j]) · X_hat[t-j]) >> shift
//! X_hat[t] = sat_i32(H[t] + R[t])
//! ```
//!
//! The recurrence and the prediction are integer-exact, so the family closes the
//! residual exactly. Model kind tag `14`. Mono-first.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, sat_i32};

/// The Q shift of the stored reflection coefficients.
pub const LATTICE_SHIFT: u8 = 14;

/// The maximum lattice order (bounded so the reconstructed direct accumulator
/// stays well inside `i64`).
pub const MAX_LATTICE_ORDER: u16 = 12;

/// A causal all-pole predictor stored as quantised reflection coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticePredictor {
    /// Channel count (this family is mono-only: always `1`).
    pub channels: u8,
    /// Order `P` (`1..=MAX_LATTICE_ORDER`).
    pub order: u16,
    /// Q shift of `reflection` (`1..=15`).
    pub shift: u8,
    /// `P` reflection coefficients in Q`shift`, each with `|k| < 2^shift`.
    pub reflection: Vec<i32>,
    /// Optional block-local reset.
    pub block_frames: Option<u32>,
}

impl LatticePredictor {
    /// Reconstruct the direct-form coefficients `a[1..=P]` in Q`shift`.
    pub fn direct_coeffs(&self) -> Vec<i64> {
        let shift = u32::from(self.shift);
        let one = 1i64 << shift;
        let mut a: Vec<i64> = vec![one];
        for (m, &k) in self.reflection.iter().enumerate() {
            let m = m + 1;
            let k = i64::from(k);
            let mut next = vec![0i64; m + 1];
            next[0] = one;
            for j in 1..m {
                next[j] = a[j] - ((k * a[m - j]) >> shift);
            }
            next[m] = -k;
            a = next;
        }
        a[1..].to_vec()
    }

    /// Validate structure, stability and the accumulator proof.
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed(
                "lattice predictor is mono-only in this build",
            ));
        }
        if self.order == 0 || self.order > MAX_LATTICE_ORDER {
            return Err(Error::limit("lattice order out of range"));
        }
        if self.reflection.len() != usize::from(self.order) {
            return Err(Error::malformed("lattice reflection count mismatch"));
        }
        if self.shift == 0 || self.shift > 15 {
            return Err(Error::malformed("lattice shift out of range"));
        }
        let limit = 1i64 << self.shift;
        for &k in &self.reflection {
            if i64::from(k).unsigned_abs() >= limit as u64 {
                return Err(Error::malformed(
                    "lattice reflection coefficient is not stable (|k| >= 1)",
                ));
            }
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit("lattice block size exceeds the bound"));
        }
        // Exact accumulator proof from the reconstructed direct coefficients.
        let a = self.direct_coeffs();
        let mut max_abs = 0u64;
        for &v in &a {
            max_abs = max_abs.max(v.unsigned_abs());
        }
        let bound = max_abs
            .saturating_mul(1u64 << 31)
            .saturating_mul(u64::from(self.order).max(1));
        if bound >= (1u64 << 62) {
            return Err(Error::limit("lattice accumulator bound exceeded"));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.order)
    }

    pub fn ops_per_sample(&self) -> u64 {
        // The direct recurrence is recomputed once, then a P-term dot product.
        u64::from(self.order) * u64::from(self.order) + u64::from(self.order)
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    #[inline]
    fn predict(&self, direct: &[i64], history: &[i32]) -> i32 {
        let mut acc = Acc::from(0);
        for (j, &a) in direct.iter().enumerate() {
            acc -= a * Acc::from(history[j]);
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
            return Err(Error::malformed("lattice residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("lattice range overflows"))?;
        if end > frames {
            return Err(Error::malformed("lattice range exceeds the extent"));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        let direct = self.direct_coeffs();
        let p = usize::from(self.order).max(1);
        let mut history = vec![0i32; p];
        for (from, to) in self.ranges(frames) {
            history.fill(0);
            for t in from..to {
                let h = self.predict(&direct, &history);
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
            return Err(Error::malformed("lattice source length mismatch"));
        }
        let direct = self.direct_coeffs();
        let mut out = vec![0i32; frames];
        let p = usize::from(self.order).max(1);
        let mut history = vec![0i32; p];
        for (from, to) in self.ranges(frames) {
            history.fill(0);
            for t in from..to {
                out[t] = self.predict(&direct, &history);
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
    /// `kind(14) || channels || order(u16) || shift(u8) || reflection(i16*) || block(u32)`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 1 + 2 + 1 + self.reflection.len() * 2 + 4);
        out.push(14);
        out.push(self.channels);
        out.extend_from_slice(&self.order.to_le_bytes());
        out.push(self.shift);
        for &k in &self.reflection {
            out.extend_from_slice(&(k as i16).to_le_bytes());
        }
        out.extend_from_slice(&self.block_frames.unwrap_or(0).to_le_bytes());
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<LatticePredictor> {
        if bytes.len() < 9 || bytes[0] != 14 {
            return Err(Error::malformed("lattice model header mismatch"));
        }
        let channels = bytes[1];
        let order = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let shift = bytes[4];
        let p = usize::from(order);
        let expect = 1 + 1 + 2 + 1 + p * 2 + 4;
        if bytes.len() != expect {
            return Err(Error::malformed("lattice model length mismatch"));
        }
        let mut reflection = Vec::with_capacity(p);
        let mut at = 5usize;
        for _ in 0..p {
            reflection.push(i32::from(i16::from_le_bytes(
                bytes[at..at + 2].try_into().unwrap(),
            )));
            at += 2;
        }
        let block = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let p = LatticePredictor {
            channels,
            order,
            shift,
            reflection,
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
    fn lattice_closes_exactly_on_a_synthetic_ar2() {
        // AR(2) with c = [0.5, 0.25] -> reflection k = [0.5, 0.25] (order-2 lattice).
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut x = vec![0i32, 0];
        for t in 2..4096 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let d = (s >> 40) as i64 % 2001 - 1000;
            let v = i64::from(x[t - 1]) / 2 + i64::from(x[t - 2]) / 4 + d;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let p = LatticePredictor {
            channels: 1,
            order: 2,
            shift: LATTICE_SHIFT,
            // For c = [0.5, 0.25] the reflection coefficients are k1 = 2/3, k2 = 0.25.
            reflection: vec![10923, 1 << 12],
            block_frames: None,
        };
        let o = LearnedObject::from_intrinsic_exp3(
            LearnedModel::Lattice(p.clone()),
            1,
            4096,
            48_000,
            Vec::new(),
            &x,
        )
        .unwrap();
        assert!(o.verify(&x));
        assert_eq!(o.materialize_range(2000, 40).unwrap(), &x[2000..2040]);
        // The reconstructed order-2 direct predictor is ~[0.5, 0.25].
        let a = p.direct_coeffs();
        let scale = f64::from(1u32 << LATTICE_SHIFT);
        let c1 = -(a[0] as f64) / scale;
        let c2 = -(a[1] as f64) / scale;
        assert!((c1 - 0.5).abs() < 0.01, "c1 = {c1}");
        assert!((c2 - 0.25).abs() < 0.01, "c2 = {c2}");
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let p = LatticePredictor {
            channels: 1,
            order: 3,
            shift: 14,
            reflection: vec![100, -200, 300],
            block_frames: Some(4096),
        };
        let b = p.canonical_bytes();
        assert_eq!(b.len(), 1 + 1 + 2 + 1 + 6 + 4);
        assert_eq!(LatticePredictor::from_canonical_bytes(&b).unwrap(), p);
    }

    #[test]
    fn validation_rejects_unstable_reflection() {
        let mut p = LatticePredictor {
            channels: 1,
            order: 2,
            shift: 14,
            reflection: vec![1 << 13, 1 << 12],
            block_frames: None,
        };
        p.reflection[0] = 1 << 14; // |k| = 1 -> unstable
        assert!(p.validate().is_err());
        p.reflection[0] = 1 << 13;
        p.order = 13; // above the bound
        p.reflection = vec![0; 13];
        assert!(p.validate().is_err());
    }
}

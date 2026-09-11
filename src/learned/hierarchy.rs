//! Hierarchical residual prediction (Exp2, priority `9`).
//!
//! A single model is asked to explain the whole signal. Hierarchical residual
//! prediction lets several cheap explanations cooperate (Mineo & Shouno,
//! APSIPA ASC 2022):
//!
//! ```text
//! H1 = P1(X history)          R1 = X - H1
//! H2 = P2(R1 history)         R2 = R1 - H2
//! …
//! Hk = Pk(R_{k-1} history)    Rk = R_{k-1} - Hk
//! X  = H1 + H2 + … + Hk + Rk
//! ```
//!
//! Only the **last** residual `Rk` is stored. Each stage is optional and is
//! retained only when its marginal complete bytes are positive. Model kind tag
//! `8`. Mono-first.
//!
//! The decoder reconstructs closed-loop: stage 0 reads already reconstructed
//! `X_hat` history, stage `k>=1` reads already reconstructed `R_{k-1}_hat`
//! history. Because `R_{k}_hat = H_{k+1} + … + H_k + R_k`, the reconstruction is
//! exact by induction.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, sat_i32};
use crate::learned::sparse::SparseLinearPredictor;

/// A cascade of mono stage predictors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HierarchicalPredictor {
    /// Channel count (mono-only in this build).
    pub channels: u8,
    /// Stage predictors: stage 0 predicts `X`, stage `k` predicts `R_{k-1}`.
    pub stages: Vec<SparseLinearPredictor>,
}

impl HierarchicalPredictor {
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed("hierarchical predictor is mono-only"));
        }
        if self.stages.is_empty()
            || self.stages.len() as u32 > crate::limits::MAX_LEARNED_HIERARCHY_STAGES
        {
            return Err(Error::limit("hierarchical stage count out of range"));
        }
        for p in &self.stages {
            if p.channels != 1 {
                return Err(Error::malformed("hierarchical stages must be mono"));
            }
            p.validate()?;
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        self.stages
            .iter()
            .map(|p| p.receptive_field())
            .max()
            .unwrap_or(0)
            .max(1)
    }

    pub fn ops_per_sample(&self) -> u64 {
        self.stages.iter().map(|p| p.ops_per_sample()).sum()
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        start
    }

    fn ring_lens(&self) -> Vec<usize> {
        self.stages.iter().map(|p| p.max_lag()).collect()
    }

    /// Predict each stage from its own history ring.
    fn predict_stages(&self, rings: &[Vec<i32>], h: &mut [i32]) {
        let mut hist = vec![0i32; 1];
        for (k, p) in self.stages.iter().enumerate() {
            let m = p.max_lag();
            if hist.len() < m {
                hist.resize(m, 0);
            }
            hist[..m].copy_from_slice(&rings[k][..m]);
            let mut o = [0i32; 1];
            p.hypothesis(&hist[..m], &mut o);
            h[k] = o[0];
        }
    }

    /// Suffix sums: `suff[k] = Σ_{j>=k} H_j + last`. `suff[0]` is the total
    /// hypothesis plus the stored residual, i.e. the reconstructed `X`.
    fn suffix_sums(h: &[i32], last: i32) -> Vec<i32> {
        let mut suff = vec![0i32; h.len()];
        let mut acc = Acc::from(last);
        for k in (0..h.len()).rev() {
            acc += Acc::from(h[k]);
            suff[k] = sat_i32(acc);
        }
        suff
    }

    fn push(rings: &mut [Vec<i32>], values: &[i32]) {
        for (ring, &v) in rings.iter_mut().zip(values.iter()) {
            ring.rotate_right(1);
            ring[0] = v;
        }
    }

    /// Hypothesis over a whole extent, open-loop from a source. Returns
    /// `Σ H_k`, so the canonical residual is the last stage residual.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("hierarchical source length mismatch"));
        }
        let mut out = vec![0i32; frames];
        let mut rings: Vec<Vec<i32>> = self.ring_lens().iter().map(|&m| vec![0i32; m]).collect();
        let mut h = vec![0i32; self.stages.len()];
        let mut values = vec![0i32; self.stages.len()];
        for t in 0..frames {
            self.predict_stages(&rings, &mut h);
            // Ring `k` holds `R_k = X - Σ_{j<k} H_j` (ring 0 holds `X`).
            let mut prefix = Acc::from(0);
            let mut total = Acc::from(0);
            for &hv in &h {
                total += Acc::from(hv);
            }
            values[0] = source[t];
            for k in 1..self.stages.len() {
                prefix += Acc::from(h[k - 1]);
                values[k] = sat_i32(Acc::from(source[t]) - prefix);
            }
            Self::push(&mut rings, &values);
            // The returned hypothesis is `Σ H_k`, so the canonical residual is
            // the last stage residual `R_k`.
            out[t] = sat_i32(total);
        }
        Ok(out)
    }

    /// Reconstruct `[start, start+len)` exactly from the stored residual.
    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        if residual.len() != frames {
            return Err(Error::malformed("hierarchical residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("hierarchical range overflows"))?;
        if end > frames {
            return Err(Error::malformed("hierarchical range exceeds the extent"));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        let mut rings: Vec<Vec<i32>> = self.ring_lens().iter().map(|&m| vec![0i32; m]).collect();
        let mut h = vec![0i32; self.stages.len()];
        for t in 0..frames {
            self.predict_stages(&rings, &mut h);
            let suff = Self::suffix_sums(&h, residual[t]);
            if t >= start && t < end {
                out[t - start] = suff[0];
            }
            Self::push(&mut rings, &suff);
        }
        Ok(out)
    }

    /// Canonical bytes: `kind(8) || channels || stage_count(u8) || [u32 len || pred]*`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(8);
        out.push(self.channels);
        out.push(self.stages.len() as u8);
        for p in &self.stages {
            let pb = p.canonical_bytes();
            out.extend_from_slice(&(pb.len() as u32).to_le_bytes());
            out.extend_from_slice(&pb);
        }
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<HierarchicalPredictor> {
        if bytes.len() < 3 || bytes[0] != 8 {
            return Err(Error::malformed("hierarchical model header mismatch"));
        }
        let channels = bytes[1];
        let count = bytes[2] as usize;
        if count == 0 || count as u32 > crate::limits::MAX_LEARNED_HIERARCHY_STAGES {
            return Err(Error::limit("hierarchical stage count out of range"));
        }
        let mut at = 3usize;
        let mut stages = Vec::with_capacity(count);
        for _ in 0..count {
            let len = u32::from_le_bytes(
                bytes
                    .get(at..at + 4)
                    .ok_or_else(|| Error::malformed("hierarchical model is truncated"))?
                    .try_into()
                    .unwrap(),
            ) as usize;
            at += 4;
            let pb = bytes
                .get(at..at + len)
                .ok_or_else(|| Error::malformed("hierarchical stage is truncated"))?;
            at += len;
            stages.push(SparseLinearPredictor::from_canonical_bytes(pb)?);
        }
        if at != bytes.len() {
            return Err(Error::malformed("hierarchical model has trailing bytes"));
        }
        let p = HierarchicalPredictor { channels, stages };
        p.validate()?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::quantize_weight;
    use crate::learned::model::LearnedModel;
    use crate::learned::object::LearnedObject;
    use crate::learned::sparse::COEFF_DELTA_VARINT;

    fn pred(lag: u16, w: f64) -> SparseLinearPredictor {
        SparseLinearPredictor {
            channels: 1,
            lags: vec![lag],
            weights: vec![quantize_weight(w)],
            bias: vec![0],
            block_frames: None,
            coeff_encoding: COEFF_DELTA_VARINT,
        }
    }

    #[test]
    fn hierarchical_cascade_closes_exactly() {
        let mut x = vec![0i32; 32];
        for t in 32..3000 {
            let v = (x[t - 1] as i64 * 7) / 10
                + (x[t - 17] as i64 * 25) / 100
                + ((t % 13) as i64 - 6) * 2;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let m = HierarchicalPredictor {
            channels: 1,
            stages: vec![pred(1, 0.6), pred(16, 0.3), pred(1, 0.2)],
        };
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::Hierarchical(m.clone()),
            1,
            3000,
            48_000,
            Vec::new(),
            &x,
        )
        .unwrap();
        assert!(o.verify(&x));
        assert_eq!(o.materialize_range(1500, 50).unwrap(), &x[1500..1550]);
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let m = HierarchicalPredictor {
            channels: 1,
            stages: vec![pred(1, 0.5), pred(4, 0.25)],
        };
        let b = m.canonical_bytes();
        let back = HierarchicalPredictor::from_canonical_bytes(&b).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn a_single_stage_is_exact() {
        let x: Vec<i32> = (0..500).map(|i| ((i * 71) % 3001) - 1500).collect();
        let m = HierarchicalPredictor {
            channels: 1,
            stages: vec![pred(1, 0.95)],
        };
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::Hierarchical(m),
            1,
            500,
            48_000,
            Vec::new(),
            &x,
        )
        .unwrap();
        assert!(o.verify(&x));
    }
}

//! Context-mixture prediction (Exp2, priority `9`).
//!
//! Non-stationary audio often needs different predictors in different local
//! regimes. A small deterministic mixture-of-experts has a much better
//! byte/operation ratio than a large all-purpose model. The gate is causal and
//! decoder-visible: it is the magnitude bucket of the **previous stored
//! residual**, which is known before the current sample is decoded and needs no
//! side information.
//!
//! Model kind tag `11`. Mono-first.
//!
//! ```text
//! c[t]     = bucket(|R[t-1]|)                // R[-1] = 0
//! H[t]     = experts[c[t]].hypothesis(X_hat history)[t]
//! X_hat[t] = sat_i32(H[t] + R[t])
//! ```

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, sat_i32};
use crate::learned::sparse::SparseLinearPredictor;

/// A context-gated mixture of mono sparse experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextMixturePredictor {
    pub channels: u8,
    /// Ascending magnitude thresholds; `experts.len() == edges.len() + 1`.
    pub edges: Vec<u32>,
    /// One expert per bucket.
    pub experts: Vec<SparseLinearPredictor>,
}

impl ContextMixturePredictor {
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed("context mixture is mono-only"));
        }
        if self.experts.is_empty()
            || self.experts.len() as u32 > crate::limits::MAX_LEARNED_CONTEXT_EXPERTS
        {
            return Err(Error::limit("context mixture expert count out of range"));
        }
        if self.experts.len() != self.edges.len() + 1 {
            return Err(Error::malformed("context mixture edges/experts mismatch"));
        }
        if self.edges.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::malformed(
                "context mixture edges must strictly ascend",
            ));
        }
        for p in &self.experts {
            if p.channels != 1 {
                return Err(Error::malformed("context mixture experts must be mono"));
            }
            p.validate()?;
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        self.experts
            .iter()
            .map(|p| p.receptive_field())
            .max()
            .unwrap_or(0)
            .max(1)
    }

    pub fn ops_per_sample(&self) -> u64 {
        self.experts
            .iter()
            .map(|p| p.ops_per_sample())
            .max()
            .unwrap_or(0)
            + 1
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

    #[inline]
    fn bucket(&self, magnitude: u32) -> usize {
        self.edges.iter().filter(|&&e| magnitude > e).count()
    }

    fn max_lag(&self) -> usize {
        self.experts
            .iter()
            .map(|p| p.max_lag())
            .max()
            .unwrap_or(1)
            .max(1)
    }

    fn predict(&self, ctx: usize, history: &[i32], out: &mut [i32]) {
        let p = &self.experts[ctx];
        let m = p.max_lag();
        p.hypothesis(&history[..m], out);
    }

    fn push(history: &mut [i32], x: i32) {
        if history.len() > 1 {
            history.copy_within(0..history.len() - 1, 1);
        }
        history[0] = x;
    }

    /// Hypothesis over a whole extent, open-loop from a source.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("context mixture source length mismatch"));
        }
        let mut out = vec![0i32; frames];
        let mut history = vec![0i32; self.max_lag()];
        let mut h = [0i32; 1];
        let mut prev_residual: u32 = 0;
        for t in 0..frames {
            let ctx = self.bucket(prev_residual);
            self.predict(ctx, &history, &mut h);
            out[t] = h[0];
            let residual = source[t].wrapping_sub(h[0]);
            prev_residual = residual.unsigned_abs();
            Self::push(&mut history, source[t]);
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
            return Err(Error::malformed("context mixture residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("context mixture range overflows"))?;
        if end > frames {
            return Err(Error::malformed("context mixture range exceeds the extent"));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        let mut history = vec![0i32; self.max_lag()];
        let mut h = [0i32; 1];
        let mut prev_residual: u32 = 0;
        for t in 0..frames {
            let ctx = self.bucket(prev_residual);
            self.predict(ctx, &history, &mut h);
            let x = sat_i32(Acc::from(h[0]) + Acc::from(residual[t]));
            if t >= start && t < end {
                out[t - start] = x;
            }
            prev_residual = residual[t].unsigned_abs();
            Self::push(&mut history, x);
        }
        Ok(out)
    }

    /// Canonical bytes:
    /// `kind(11) || channels || edge_count(u8) || edges(u32*) || [u32 len || expert]*`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(11);
        out.push(self.channels);
        out.push(self.edges.len() as u8);
        for &e in &self.edges {
            out.extend_from_slice(&e.to_le_bytes());
        }
        for p in &self.experts {
            let pb = p.canonical_bytes();
            out.extend_from_slice(&(pb.len() as u32).to_le_bytes());
            out.extend_from_slice(&pb);
        }
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<ContextMixturePredictor> {
        if bytes.len() < 3 || bytes[0] != 11 {
            return Err(Error::malformed("context mixture header mismatch"));
        }
        let channels = bytes[1];
        let edge_count = bytes[2] as usize;
        if edge_count + 1 > crate::limits::MAX_LEARNED_CONTEXT_EXPERTS as usize {
            return Err(Error::limit("context mixture edge count out of range"));
        }
        let mut at = 3usize;
        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            let e = u32::from_le_bytes(
                bytes
                    .get(at..at + 4)
                    .ok_or_else(|| Error::malformed("context mixture is truncated"))?
                    .try_into()
                    .unwrap(),
            );
            at += 4;
            edges.push(e);
        }
        let mut experts = Vec::with_capacity(edge_count + 1);
        for _ in 0..=edge_count {
            let len = u32::from_le_bytes(
                bytes
                    .get(at..at + 4)
                    .ok_or_else(|| Error::malformed("context mixture is truncated"))?
                    .try_into()
                    .unwrap(),
            ) as usize;
            at += 4;
            let pb = bytes
                .get(at..at + len)
                .ok_or_else(|| Error::malformed("context mixture expert is truncated"))?;
            at += len;
            experts.push(SparseLinearPredictor::from_canonical_bytes(pb)?);
        }
        if at != bytes.len() {
            return Err(Error::malformed("context mixture has trailing bytes"));
        }
        let p = ContextMixturePredictor {
            channels,
            edges,
            experts,
        };
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

    fn expert(lag: u16, w: f64) -> SparseLinearPredictor {
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
    fn context_mixture_closes_exactly() {
        let mut x = vec![0i32; 8];
        for t in 8..2500 {
            let v = (x[t - 1] as i64 * 5) / 10
                + (x[t - 4] as i64 * 3) / 10
                + ((t % 17) as i64 - 8) * 11;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let m = ContextMixturePredictor {
            channels: 1,
            edges: vec![500, 5000],
            experts: vec![expert(1, 0.9), expert(1, 0.5), expert(4, 0.4)],
        };
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::ContextMixture(m.clone()),
            1,
            2500,
            48_000,
            Vec::new(),
            &x,
        )
        .unwrap();
        assert!(o.verify(&x));
        assert_eq!(o.materialize_range(1200, 30).unwrap(), &x[1200..1230]);
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let m = ContextMixturePredictor {
            channels: 1,
            edges: vec![100],
            experts: vec![expert(1, 0.5), expert(2, 0.25)],
        };
        let b = m.canonical_bytes();
        let back = ContextMixturePredictor::from_canonical_bytes(&b).unwrap();
        assert_eq!(back, m);
    }
}

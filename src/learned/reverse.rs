//! Reverse-direction block prediction (Report 3 remaining speech mechanism,
//! experimental profile `vole.audio.learned.exp3`).
//!
//! A finite block does not have to be predicted in chronological order. Speech
//! attacks and decays are temporally asymmetric, so for some blocks predicting
//! right-to-left from the block's terminal state produces a smaller exact
//! residual than predicting left-to-right. This wrapper runs any inner model on
//! the **reversed** block and reverses the result back:
//!
//! ```text
//! H = reverse( inner_hypothesis( reverse(X) ) )
//! X_hat = reverse( inner_evaluation( reverse(R) ) )
//! ```
//!
//! The transform is an involution and the inner model is exact, so closure stays
//! exact. The decoder reconstructs the whole block in reverse order (zero initial
//! history at the original block end) and slices. Model kind tag `17`.
//! Mono-first; the inner model must not itself be reverse, segmented or
//! source-dependent.

use crate::error::{Error, Result};
use crate::learned::model::LearnedModel;

/// A reverse-direction realisation of an inner model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReversePredictor {
    pub channels: u8,
    /// The inner hypothesis, evaluated on the reversed block.
    pub inner: Box<LearnedModel>,
}

impl ReversePredictor {
    /// Validate the inner nesting rules.
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed(
                "reverse predictor is mono-only in this build",
            ));
        }
        if matches!(&*self.inner, LearnedModel::Reverse(_)) {
            return Err(Error::malformed("reverse predictors may not nest"));
        }
        if matches!(&*self.inner, LearnedModel::Segmented(_)) {
            return Err(Error::malformed(
                "reverse predictor must wrap a single-block model, not a segmentation",
            ));
        }
        if self.inner.requires_source() {
            return Err(Error::malformed(
                "reverse predictor cannot supply an inner source dependency",
            ));
        }
        self.inner.validate()?;
        if self.inner.channels() != self.channels {
            return Err(Error::malformed(
                "reverse inner model channels disagree with the wrapper",
            ));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        self.inner.receptive_field()
    }

    pub fn ops_per_sample(&self) -> u64 {
        self.inner.ops_per_sample().saturating_add(1)
    }

    pub fn state_bytes(&self) -> u64 {
        self.inner.state_bytes()
    }

    pub fn checkpoint_count(&self) -> u32 {
        self.inner.checkpoint_count()
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        // Serving `start` in a reverse block requires replaying from the block
        // end down to `start`; the caller (a segmentation) bounds the block.
        self.inner.replay_frames(start)
    }

    /// Hypothesis over a whole extent.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("reverse source length mismatch"));
        }
        let rev: Vec<i32> = source.iter().rev().copied().collect();
        let h = self.inner.hypothesis_from_source(&rev, frames)?;
        Ok(h.into_iter().rev().collect())
    }

    /// Reconstruct `[start, start+len)` by reconstructing the whole block in
    /// reverse and slicing.
    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        if residual.len() != frames {
            return Err(Error::malformed("reverse residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("reverse range overflows"))?;
        if end > frames {
            return Err(Error::malformed("reverse range exceeds the extent"));
        }
        let rev: Vec<i32> = residual.iter().rev().copied().collect();
        let recon_rev = self.inner.evaluate_range(&rev, frames, 0, frames)?;
        let recon: Vec<i32> = recon_rev.into_iter().rev().collect();
        Ok(recon[start..end].to_vec())
    }

    /// Canonical bytes: `kind(17) || channels || inner_len(u64) || inner`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let inner = self.inner.canonical_bytes();
        let mut out = Vec::with_capacity(1 + 1 + 8 + inner.len());
        out.push(17);
        out.push(self.channels);
        out.extend_from_slice(&(inner.len() as u64).to_le_bytes());
        out.extend_from_slice(&inner);
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<ReversePredictor> {
        if bytes.len() < 10 || bytes[0] != 17 {
            return Err(Error::malformed("reverse model header mismatch"));
        }
        let channels = bytes[1];
        let inner_len = u64::from_le_bytes(bytes[2..10].try_into().unwrap()) as usize;
        if bytes.len() != 10 + inner_len {
            return Err(Error::malformed("reverse model length mismatch"));
        }
        let inner = LearnedModel::from_canonical_bytes(&bytes[10..])?;
        let r = ReversePredictor {
            channels,
            inner: Box::new(inner),
        };
        r.validate()?;
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::lpc::LpcPredictor;
    use crate::learned::object::LearnedObject;

    #[test]
    fn reverse_closes_exactly() {
        let x: Vec<i32> = (0..4096).map(|i| ((i * 37) % 2003) - 1000).collect();
        let inner = LearnedModel::Lpc(LpcPredictor {
            channels: 1,
            order: 4,
            precision: 16,
            shift: 12,
            coeffs: vec![100, -50, 25, -12],
            block_frames: None,
        });
        let model = LearnedModel::Reverse(ReversePredictor {
            channels: 1,
            inner: Box::new(inner),
        });
        let o = LearnedObject::from_intrinsic_exp3(model, 1, 4096, 48_000, Vec::new(), &x).unwrap();
        assert!(o.verify(&x));
        assert_eq!(o.materialize_range(2000, 40).unwrap(), &x[2000..2040]);
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
        let r = ReversePredictor {
            channels: 1,
            inner: Box::new(inner),
        };
        let b = r.canonical_bytes();
        assert_eq!(ReversePredictor::from_canonical_bytes(&b).unwrap(), r);
    }
}

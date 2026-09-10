//! The learned hypothesis vocabulary (`O.3`, `O.19`, `O.28`).
//!
//! A [`LearnedModel`] is the deterministic executable description proposed by
//! fitting/search. It is never the "true cause" of a signal and never has
//! semantic authority. The canonical evaluator is VOLE-native: no ONNX, no
//! PyTorch/TensorFlow runtime, no generic tensor format.
//!
//! Frozen model-kind tags:
//!
//! ```text
//! 0  linear finite-field / block-local (one causal FIR + bias)
//! 1  nonlinear finite-field (bounded integer graph)
//! 2  stateful (closed-loop recurrent with checkpoints)
//! 3  transfer operator (source dependency + learned operator)
//! ```

use crate::error::{Error, Result};
use crate::learned::finite_field::LinearPredictor;

/// The learned hypothesis families implemented by this build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LearnedModel {
    /// A quantized causal FIR (finite-field, optionally block-local).
    Linear(LinearPredictor),
    /// A bounded integer graph with frozen nonlinear primitives.
    Nonlinear(crate::learned::graph::NonlinearGraph),
    /// A closed-loop recurrent predictor with canonical checkpoints.
    Stateful(crate::learned::stateful::StatefulPredictor),
    /// A learned operator relating a source object to a target object.
    Transfer(crate::learned::transfer::TransferOperator),
}

impl LearnedModel {
    /// Frozen canonical kind tag.
    pub const fn kind_tag(&self) -> u8 {
        match self {
            LearnedModel::Linear(_) => 0,
            LearnedModel::Nonlinear(_) => 1,
            LearnedModel::Stateful(_) => 2,
            LearnedModel::Transfer(_) => 3,
        }
    }

    /// Stable evidence label.
    pub const fn kind_name(&self) -> &'static str {
        match self {
            LearnedModel::Linear(_) => "linear_finite_field",
            LearnedModel::Nonlinear(_) => "nonlinear_finite_field",
            LearnedModel::Stateful(_) => "stateful",
            LearnedModel::Transfer(_) => "transfer_operator",
        }
    }

    /// Validate dimensions, bounds and the complexity ceiling (`O.30`).
    pub fn validate(&self) -> Result<()> {
        match self {
            LearnedModel::Linear(p) => p.validate(),
            LearnedModel::Nonlinear(g) => g.validate(),
            LearnedModel::Stateful(s) => s.validate(),
            LearnedModel::Transfer(t) => t.validate(),
        }
    }

    /// Channels of the intrinsic domain.
    pub fn channels(&self) -> u8 {
        match self {
            LearnedModel::Linear(p) => p.channels,
            LearnedModel::Nonlinear(g) => g.channels,
            LearnedModel::Stateful(s) => s.channels,
            LearnedModel::Transfer(t) => t.channels,
        }
    }

    /// Declared receptive field in frames.
    pub fn receptive_field(&self) -> u64 {
        match self {
            LearnedModel::Linear(p) => p.receptive_field(),
            LearnedModel::Nonlinear(g) => g.receptive_field(),
            LearnedModel::Stateful(s) => s.receptive_field(),
            LearnedModel::Transfer(t) => t.receptive_field(),
        }
    }

    /// Abstract operations per output sample.
    pub fn ops_per_sample(&self) -> u64 {
        match self {
            LearnedModel::Linear(p) => p.ops_per_sample(),
            LearnedModel::Nonlinear(g) => g.ops_per_sample(),
            LearnedModel::Stateful(s) => s.ops_per_sample(),
            LearnedModel::Transfer(t) => t.ops_per_sample(),
        }
    }

    /// Persistent learned state bytes.
    pub fn state_bytes(&self) -> u64 {
        match self {
            LearnedModel::Linear(p) => p.state_bytes(),
            LearnedModel::Nonlinear(g) => g.state_bytes(),
            LearnedModel::Stateful(s) => s.state_bytes(),
            LearnedModel::Transfer(t) => t.state_bytes(),
        }
    }

    /// Declared checkpoints.
    pub fn checkpoint_count(&self) -> u32 {
        match self {
            LearnedModel::Linear(p) => p.checkpoint_count(),
            LearnedModel::Nonlinear(g) => g.checkpoint_count(),
            LearnedModel::Stateful(s) => s.checkpoint_count(),
            LearnedModel::Transfer(t) => t.checkpoint_count(),
        }
    }

    /// Canonical model bytes (kind tag first).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        match self {
            LearnedModel::Linear(p) => p.canonical_bytes(),
            LearnedModel::Nonlinear(g) => g.canonical_bytes(),
            LearnedModel::Stateful(s) => s.canonical_bytes(),
            LearnedModel::Transfer(t) => t.canonical_bytes(),
        }
    }

    /// Parse canonical model bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<LearnedModel> {
        let (&kind, rest) = bytes
            .split_first()
            .ok_or_else(|| Error::malformed("empty learned model"))?;
        let m = match kind {
            0 => LearnedModel::Linear(LinearPredictor::from_canonical_bytes(bytes)?),
            1 => LearnedModel::Nonlinear(
                crate::learned::graph::NonlinearGraph::from_canonical_bytes(bytes)?,
            ),
            2 => LearnedModel::Stateful(
                crate::learned::stateful::StatefulPredictor::from_canonical_bytes(bytes)?,
            ),
            3 => LearnedModel::Transfer(
                crate::learned::transfer::TransferOperator::from_canonical_bytes(bytes)?,
            ),
            other => {
                return Err(Error::new(
                    crate::error::Kind::Unsupported,
                    format!("unknown learned model kind {other}"),
                ));
            }
        };
        let _ = rest;
        m.validate()?;
        Ok(m)
    }

    /// Compute the hypothesis over a whole extent, open-loop from a source
    /// signal (used only to build the residual).
    pub fn hypothesis_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        match self {
            LearnedModel::Linear(p) => p.hypothesis_all_from_source(source, frames),
            LearnedModel::Nonlinear(g) => g.hypothesis_all_from_source(source, frames),
            LearnedModel::Stateful(s) => s.hypothesis_all_from_source(source, frames),
            LearnedModel::Transfer(t) => t.hypothesis_from_source(source, frames),
        }
    }

    /// Reconstruct the exact canonical samples for `[start, start+len)`.
    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        match self {
            LearnedModel::Linear(p) => p.evaluate_range(residual, frames, start, len),
            LearnedModel::Nonlinear(g) => g.evaluate_range(residual, frames, start, len),
            LearnedModel::Stateful(s) => s.evaluate_range(residual, frames, start, len),
            LearnedModel::Transfer(t) => t.evaluate_range(residual, frames, start, len),
        }
    }

    /// Frames that must be replayed to serve `start`.
    pub fn replay_frames(&self, start: usize) -> usize {
        match self {
            LearnedModel::Linear(p) => p.replay_frames(start),
            LearnedModel::Nonlinear(g) => g.replay_frames(start),
            LearnedModel::Stateful(s) => s.replay_frames(start),
            LearnedModel::Transfer(t) => t.replay_frames(start),
        }
    }

    /// True when the family reads a source object (a declared dependency).
    pub const fn requires_source(&self) -> bool {
        matches!(self, LearnedModel::Transfer(_))
    }

    /// Reconstruct `[start, start+len)` exactly, supplying a transfer source
    /// where the family needs one. Non-transfer families ignore `source`.
    pub fn evaluate_range_with_source(
        &self,
        residual: &[i32],
        source: Option<&[i32]>,
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        match self {
            LearnedModel::Transfer(t) => {
                let s = source.ok_or_else(|| {
                    Error::dependency("transfer object requires its source samples")
                })?;
                t.evaluate_range_with_source(residual, s, frames, start, len)
            }
            _ => self.evaluate_range(residual, frames, start, len),
        }
    }
}

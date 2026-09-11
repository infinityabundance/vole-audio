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
    /// An independently-materializable sequence of segments (Exp2).
    Segmented(crate::learned::segmented::SegmentedModel),
    /// A sparse selected-lag linear predictor (Exp2).
    SparseLinear(crate::learned::sparse::SparseLinearPredictor),
    /// A short-term predictor plus a long-term (pitch) stage (Exp2).
    LongTerm(crate::learned::ltp::LongTermPredictor),
    /// A reversible channel transform plus per-component predictors (Exp2).
    Multichannel(crate::learned::multichannel::MultichannelPredictor),
    /// A cascade of stage predictors over successive residuals (Exp2).
    Hierarchical(crate::learned::hierarchy::HierarchicalPredictor),
    /// An analytic source transform plus an optional learned correction (Exp2).
    AnalyticTransfer(crate::learned::analytical::AnalyticTransfer),
    /// A backward-adaptive predictor updated from reconstructed samples (Exp2).
    Adaptive(crate::learned::adaptive::AdaptivePredictor),
    /// A context-gated mixture of mono sparse experts (Exp2).
    ContextMixture(crate::learned::context_mixture::ContextMixturePredictor),
    /// A dense short-term all-pole LPC predictor (Exp2, Seal S2).
    Lpc(crate::learned::lpc::LpcPredictor),
}

impl LearnedModel {
    /// Frozen canonical kind tag.
    pub const fn kind_tag(&self) -> u8 {
        match self {
            LearnedModel::Linear(_) => 0,
            LearnedModel::Nonlinear(_) => 1,
            LearnedModel::Stateful(_) => 2,
            LearnedModel::Transfer(_) => 3,
            LearnedModel::Segmented(_) => 4,
            LearnedModel::SparseLinear(_) => 5,
            LearnedModel::LongTerm(_) => 6,
            LearnedModel::Multichannel(_) => 7,
            LearnedModel::Hierarchical(_) => 8,
            LearnedModel::AnalyticTransfer(_) => 9,
            LearnedModel::Adaptive(_) => 10,
            LearnedModel::ContextMixture(_) => 11,
            LearnedModel::Lpc(_) => 12,
        }
    }

    /// Stable evidence label.
    pub const fn kind_name(&self) -> &'static str {
        match self {
            LearnedModel::Linear(_) => "linear_finite_field",
            LearnedModel::Nonlinear(_) => "nonlinear_finite_field",
            LearnedModel::Stateful(_) => "stateful",
            LearnedModel::Transfer(_) => "transfer_operator",
            LearnedModel::Segmented(_) => "segmented",
            LearnedModel::SparseLinear(_) => "sparse_linear",
            LearnedModel::LongTerm(_) => "long_term",
            LearnedModel::Multichannel(_) => "multichannel",
            LearnedModel::Hierarchical(_) => "hierarchical",
            LearnedModel::AnalyticTransfer(_) => "analytic_transfer",
            LearnedModel::Adaptive(_) => "backward_adaptive",
            LearnedModel::ContextMixture(_) => "context_mixture",
            LearnedModel::Lpc(_) => "lpc",
        }
    }

    /// Validate dimensions, bounds and the complexity ceiling (`O.30`).
    pub fn validate(&self) -> Result<()> {
        match self {
            LearnedModel::Linear(p) => p.validate(),
            LearnedModel::Nonlinear(g) => g.validate(),
            LearnedModel::Stateful(s) => s.validate(),
            LearnedModel::Transfer(t) => t.validate(),
            LearnedModel::Segmented(s) => s.validate(),
            LearnedModel::SparseLinear(p) => p.validate(),
            LearnedModel::LongTerm(p) => p.validate(),
            LearnedModel::Multichannel(p) => p.validate(),
            LearnedModel::Hierarchical(p) => p.validate(),
            LearnedModel::AnalyticTransfer(p) => p.validate(),
            LearnedModel::Adaptive(p) => p.validate(),
            LearnedModel::ContextMixture(p) => p.validate(),
            LearnedModel::Lpc(p) => p.validate(),
        }
    }

    /// Channels of the intrinsic domain.
    pub fn channels(&self) -> u8 {
        match self {
            LearnedModel::Linear(p) => p.channels,
            LearnedModel::Nonlinear(g) => g.channels,
            LearnedModel::Stateful(s) => s.channels,
            LearnedModel::Transfer(t) => t.channels,
            LearnedModel::Segmented(s) => s.channels,
            LearnedModel::SparseLinear(p) => p.channels,
            LearnedModel::LongTerm(p) => p.channels,
            LearnedModel::Multichannel(p) => p.channels,
            LearnedModel::Hierarchical(p) => p.channels,
            LearnedModel::AnalyticTransfer(p) => p.channels,
            LearnedModel::Adaptive(p) => p.channels,
            LearnedModel::ContextMixture(p) => p.channels,
            LearnedModel::Lpc(p) => p.channels,
        }
    }

    /// Declared receptive field in frames.
    pub fn receptive_field(&self) -> u64 {
        match self {
            LearnedModel::Linear(p) => p.receptive_field(),
            LearnedModel::Nonlinear(g) => g.receptive_field(),
            LearnedModel::Stateful(s) => s.receptive_field(),
            LearnedModel::Transfer(t) => t.receptive_field(),
            LearnedModel::Segmented(s) => s.receptive_field(),
            LearnedModel::SparseLinear(p) => p.receptive_field(),
            LearnedModel::LongTerm(p) => p.receptive_field(),
            LearnedModel::Multichannel(p) => p.receptive_field(),
            LearnedModel::Hierarchical(p) => p.receptive_field(),
            LearnedModel::AnalyticTransfer(p) => p.receptive_field(),
            LearnedModel::Adaptive(p) => p.receptive_field(),
            LearnedModel::ContextMixture(p) => p.receptive_field(),
            LearnedModel::Lpc(p) => p.receptive_field(),
        }
    }

    /// Abstract operations per output sample.
    pub fn ops_per_sample(&self) -> u64 {
        match self {
            LearnedModel::Linear(p) => p.ops_per_sample(),
            LearnedModel::Nonlinear(g) => g.ops_per_sample(),
            LearnedModel::Stateful(s) => s.ops_per_sample(),
            LearnedModel::Transfer(t) => t.ops_per_sample(),
            LearnedModel::Segmented(s) => s.ops_per_sample(),
            LearnedModel::SparseLinear(p) => p.ops_per_sample(),
            LearnedModel::LongTerm(p) => p.ops_per_sample(),
            LearnedModel::Multichannel(p) => p.ops_per_sample(),
            LearnedModel::Hierarchical(p) => p.ops_per_sample(),
            LearnedModel::AnalyticTransfer(p) => p.ops_per_sample(),
            LearnedModel::Adaptive(p) => p.ops_per_sample(),
            LearnedModel::ContextMixture(p) => p.ops_per_sample(),
            LearnedModel::Lpc(p) => p.ops_per_sample(),
        }
    }

    /// Persistent learned state bytes.
    pub fn state_bytes(&self) -> u64 {
        match self {
            LearnedModel::Linear(p) => p.state_bytes(),
            LearnedModel::Nonlinear(g) => g.state_bytes(),
            LearnedModel::Stateful(s) => s.state_bytes(),
            LearnedModel::Transfer(t) => t.state_bytes(),
            LearnedModel::Segmented(s) => s.state_bytes(),
            LearnedModel::SparseLinear(p) => p.state_bytes(),
            LearnedModel::LongTerm(p) => p.state_bytes(),
            LearnedModel::Multichannel(p) => p.state_bytes(),
            LearnedModel::Hierarchical(p) => p.state_bytes(),
            LearnedModel::AnalyticTransfer(p) => p.state_bytes(),
            LearnedModel::Adaptive(p) => p.state_bytes(),
            LearnedModel::ContextMixture(p) => p.state_bytes(),
            LearnedModel::Lpc(p) => p.state_bytes(),
        }
    }

    /// Declared checkpoints.
    pub fn checkpoint_count(&self) -> u32 {
        match self {
            LearnedModel::Linear(p) => p.checkpoint_count(),
            LearnedModel::Nonlinear(g) => g.checkpoint_count(),
            LearnedModel::Stateful(s) => s.checkpoint_count(),
            LearnedModel::Transfer(t) => t.checkpoint_count(),
            LearnedModel::Segmented(s) => s.checkpoint_count(),
            LearnedModel::SparseLinear(p) => p.checkpoint_count(),
            LearnedModel::LongTerm(p) => p.checkpoint_count(),
            LearnedModel::Multichannel(p) => p.checkpoint_count(),
            LearnedModel::Hierarchical(p) => p.checkpoint_count(),
            LearnedModel::AnalyticTransfer(p) => p.checkpoint_count(),
            LearnedModel::Adaptive(p) => p.checkpoint_count(),
            LearnedModel::ContextMixture(p) => p.checkpoint_count(),
            LearnedModel::Lpc(p) => p.checkpoint_count(),
        }
    }

    /// Canonical model bytes (kind tag first).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        match self {
            LearnedModel::Linear(p) => p.canonical_bytes(),
            LearnedModel::Nonlinear(g) => g.canonical_bytes(),
            LearnedModel::Stateful(s) => s.canonical_bytes(),
            LearnedModel::Transfer(t) => t.canonical_bytes(),
            LearnedModel::Segmented(s) => s.canonical_bytes(),
            LearnedModel::SparseLinear(p) => p.canonical_bytes(),
            LearnedModel::LongTerm(p) => p.canonical_bytes(),
            LearnedModel::Multichannel(p) => p.canonical_bytes(),
            LearnedModel::Hierarchical(p) => p.canonical_bytes(),
            LearnedModel::AnalyticTransfer(p) => p.canonical_bytes(),
            LearnedModel::Adaptive(p) => p.canonical_bytes(),
            LearnedModel::ContextMixture(p) => p.canonical_bytes(),
            LearnedModel::Lpc(p) => p.canonical_bytes(),
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
            4 => LearnedModel::Segmented(
                crate::learned::segmented::SegmentedModel::from_canonical_bytes(bytes)?,
            ),
            5 => LearnedModel::SparseLinear(
                crate::learned::sparse::SparseLinearPredictor::from_canonical_bytes(bytes)?,
            ),
            6 => LearnedModel::LongTerm(
                crate::learned::ltp::LongTermPredictor::from_canonical_bytes(bytes)?,
            ),
            7 => LearnedModel::Multichannel(
                crate::learned::multichannel::MultichannelPredictor::from_canonical_bytes(bytes)?,
            ),
            8 => LearnedModel::Hierarchical(
                crate::learned::hierarchy::HierarchicalPredictor::from_canonical_bytes(bytes)?,
            ),
            9 => LearnedModel::AnalyticTransfer(
                crate::learned::analytical::AnalyticTransfer::from_canonical_bytes(bytes)?,
            ),
            10 => LearnedModel::Adaptive(
                crate::learned::adaptive::AdaptivePredictor::from_canonical_bytes(bytes)?,
            ),
            11 => LearnedModel::ContextMixture(
                crate::learned::context_mixture::ContextMixturePredictor::from_canonical_bytes(
                    bytes,
                )?,
            ),
            12 => LearnedModel::Lpc(crate::learned::lpc::LpcPredictor::from_canonical_bytes(
                bytes,
            )?),
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
            LearnedModel::Segmented(s) => s.hypothesis_from_source(source, frames),
            LearnedModel::SparseLinear(p) => p.hypothesis_all_from_source(source, frames),
            LearnedModel::LongTerm(p) => p.hypothesis_all_from_source(source, frames),
            LearnedModel::Multichannel(p) => p.hypothesis_all_from_source(source, frames),
            LearnedModel::Hierarchical(p) => p.hypothesis_all_from_source(source, frames),
            LearnedModel::AnalyticTransfer(p) => p.hypothesis_from_source(source, frames),
            LearnedModel::Adaptive(p) => p.hypothesis_all_from_source(source, frames),
            LearnedModel::ContextMixture(p) => p.hypothesis_all_from_source(source, frames),
            LearnedModel::Lpc(p) => p.hypothesis_all_from_source(source, frames),
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
            LearnedModel::Segmented(s) => s.evaluate_range(residual, frames, start, len),
            LearnedModel::SparseLinear(p) => p.evaluate_range(residual, frames, start, len),
            LearnedModel::LongTerm(p) => p.evaluate_range(residual, frames, start, len),
            LearnedModel::Multichannel(p) => p.evaluate_range(residual, frames, start, len),
            LearnedModel::Hierarchical(p) => p.evaluate_range(residual, frames, start, len),
            LearnedModel::AnalyticTransfer(_) => Err(Error::dependency(
                "analytic transfer requires its source samples",
            )),
            LearnedModel::Adaptive(p) => p.evaluate_range(residual, frames, start, len),
            LearnedModel::ContextMixture(p) => p.evaluate_range(residual, frames, start, len),
            LearnedModel::Lpc(p) => p.evaluate_range(residual, frames, start, len),
        }
    }

    /// Frames that must be replayed to serve `start`.
    pub fn replay_frames(&self, start: usize) -> usize {
        match self {
            LearnedModel::Linear(p) => p.replay_frames(start),
            LearnedModel::Nonlinear(g) => g.replay_frames(start),
            LearnedModel::Stateful(s) => s.replay_frames(start),
            LearnedModel::Transfer(t) => t.replay_frames(start),
            LearnedModel::Segmented(s) => s.replay_frames(start),
            LearnedModel::SparseLinear(p) => p.replay_frames(start),
            LearnedModel::LongTerm(p) => p.replay_frames(start),
            LearnedModel::Multichannel(p) => p.replay_frames(start),
            LearnedModel::Hierarchical(p) => p.replay_frames(start),
            LearnedModel::AnalyticTransfer(p) => p.replay_frames(start),
            LearnedModel::Adaptive(p) => p.replay_frames(start),
            LearnedModel::ContextMixture(p) => p.replay_frames(start),
            LearnedModel::Lpc(p) => p.replay_frames(start),
        }
    }

    /// True when the family reads a source object (a declared dependency).
    pub const fn requires_source(&self) -> bool {
        matches!(
            self,
            LearnedModel::Transfer(_) | LearnedModel::AnalyticTransfer(_)
        )
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
            LearnedModel::AnalyticTransfer(t) => {
                let s = source.ok_or_else(|| {
                    Error::dependency("analytic transfer object requires its source samples")
                })?;
                t.evaluate_range_with_source(residual, s, frames, start, len)
            }
            _ => self.evaluate_range(residual, frames, start, len),
        }
    }
}

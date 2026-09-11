//! Learned representation accounting (`O.5`, `O.12`, `O.13`, `O.15`, `O.40`).
//!
//! The fundamental quantity is the **complete representation cost**, never the
//! prediction error. For a learned object:
//!
//! ```text
//! complete_bytes = metadata
//!                + model (weights + biases + scales + activations + graph
//!                         + tensor dimensions + state/checkpoint definitions)
//!                + residual (codec id + payload)
//!                + dependencies
//!                + integrity
//! ```
//!
//! and the decomposition is required to sum **exactly** to the canonical
//! serialized length: "accounted bytes" and "actual bytes" are the same number
//! or the court fails. Transfer costs are reported in a standalone and a
//! marginal regime; the source dependency is never silently free.

use crate::error::Result;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;

/// Named model sub-components `(weights, biases, activations, checkpoints,
/// tensor_dims, state_def)`. Every family's components sum to at most its
/// canonical model length; the remainder is graph/header metadata.
fn model_components(m: &LearnedModel) -> (u64, u64, u64, u64, u64, u64) {
    match m {
        LearnedModel::Linear(p) => (
            (p.weights.len() as u64) * 2,
            (p.bias.len() as u64) * 4,
            0,
            0,
            0,
            0,
        ),
        LearnedModel::Nonlinear(g) => {
            let w: u64 = g.layers.iter().map(|l| (l.weights.len() as u64) * 2).sum();
            let b: u64 = g.layers.iter().map(|l| (l.bias.len() as u64) * 4).sum();
            let a: u64 = g.layers.iter().map(|l| l.activation.canonical_len()).sum();
            let t: u64 = (g.layers.len() as u64) * 8 + 4;
            (w, b, a, 0, t, 0)
        }
        LearnedModel::Stateful(s) => {
            let w = ((s.out_w.len() + s.rec_w.len() + s.in_w.len()) as u64) * 2;
            let b = ((s.out_b.len() + s.rec_b.len()) as u64) * 4;
            let a = s.activation.canonical_len();
            let cp = s.checkpoint_bytes();
            let t = 6;
            (w, b, a, cp, t, 0)
        }
        LearnedModel::Transfer(t) => (
            (t.weights.len() as u64) * 2,
            (t.bias.len() as u64) * 4,
            0,
            0,
            0,
            0,
        ),
        LearnedModel::SparseLinear(p) => {
            // Canonical coefficient/bias payloads, capped so the named
            // components never exceed the model length; the remainder is graph
            // metadata (lag deltas and framing).
            let len = p.canonical_bytes().len() as u64;
            let weights = ((p.weights.len() as u64) * 2).min(len);
            let bias = 4u64.min(len - weights);
            (weights, bias, 0, 0, 0, 0)
        }
        LearnedModel::LongTerm(p) => {
            let len = p.canonical_bytes().len() as u64;
            let (sw, sb, _, _, _, _) =
                model_components(&LearnedModel::SparseLinear(p.short.clone()));
            let weights = (sw + (p.gains.len() as u64) * 2).min(len);
            let bias = sb.min(len - weights);
            (weights, bias, 0, 0, 0, 0)
        }
        LearnedModel::Multichannel(p) => {
            let len = p.canonical_bytes().len() as u64;
            let mut w = 0u64;
            let mut b = 0u64;
            for pred in &p.predictors {
                let (sw, sb, _, _, _, _) =
                    model_components(&LearnedModel::SparseLinear(pred.clone()));
                w += sw;
                b += sb;
            }
            let weights = w.min(len);
            let bias = b.min(len - weights);
            (weights, bias, 0, 0, 0, 0)
        }
        LearnedModel::Hierarchical(p) => {
            let len = p.canonical_bytes().len() as u64;
            let mut w = 0u64;
            let mut b = 0u64;
            for pred in &p.stages {
                let (sw, sb, _, _, _, _) =
                    model_components(&LearnedModel::SparseLinear(pred.clone()));
                w += sw;
                b += sb;
            }
            let weights = w.min(len);
            let bias = b.min(len - weights);
            (weights, bias, 0, 0, 0, 0)
        }
        LearnedModel::AnalyticTransfer(t) => {
            let len = t.canonical_bytes().len() as u64;
            let (w, b) = if t.has_correction {
                (
                    (t.correction.weights.len() as u64) * 2,
                    (t.correction.bias.len() as u64) * 4,
                )
            } else {
                (0, 0)
            };
            let weights = w.min(len);
            let bias = b.min(len - weights);
            (weights, bias, 0, 0, 0, 0)
        }
        LearnedModel::Adaptive(p) => {
            let len = p.canonical_bytes().len() as u64;
            let weights = ((p.init_weights.len() as u64) * 2).min(len);
            let bias = 4u64.min(len - weights);
            (weights, bias, 0, 0, 0, 0)
        }
        LearnedModel::Segmented(s) => {
            let mut w = 0u64;
            let mut b = 0u64;
            let mut a = 0u64;
            let mut cp = 0u64;
            let mut td = 0u64;
            let mut st = 0u64;
            for seg in &s.segments {
                let (sw, sb, sa, scp, std, sst) = model_components(&seg.model);
                w += sw;
                b += sb;
                a += sa;
                cp += scp;
                td += std;
                st += sst;
            }
            // Segment framing: seg_count(4) + per-segment frames(4) + model_len(8).
            td += 4 + (s.segments.len() as u64) * 12;
            (w, b, a, cp, td, st)
        }
    }
}

/// The complete storage cost of one learned representation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LearnedCost {
    // --- model components (sum to `model_bytes`) ---
    pub canonical_weight_bytes: u64,
    pub raw_weight_bytes: u64,
    pub bias_bytes: u64,
    pub scale_bytes: u64,
    pub activation_table_bytes: u64,
    pub graph_metadata_bytes: u64,
    pub tensor_dimension_bytes: u64,
    pub state_definition_bytes: u64,
    pub checkpoint_definition_bytes: u64,
    /// Sum of the model components.
    pub model_bytes: u64,

    // --- the rest of the object ---
    /// Container framing (magic, version, profile, geometry, count fields).
    pub metadata_bytes: u64,
    /// Residual codec id byte plus payload.
    pub residual_bytes: u64,
    /// `32 × dependency_count`.
    pub dependency_bytes: u64,
    /// Trailing identity digest.
    pub integrity_bytes: u64,

    /// The eight-way sum; equals the canonical serialized length.
    pub complete_bytes: u64,

    // --- explicitly not part of the sum ---
    /// Raw canonical sample bytes of the unencoded intrinsic (`4 · N`).
    pub raw_sample_bytes: u64,
    /// Canonical U1 literal bytes of the same extent.
    pub canonical_literal_bytes: u64,
    /// Persistent sample-domain bytes the representation bakes (0 for learned).
    pub persistent_sample_domain_bytes: u64,
    /// Persistent learned state bytes.
    pub persistent_state_bytes: u64,
    /// Working state materialized to evaluate one window.
    pub decoded_window_state_bytes: u64,
    /// Abstract operations per output sample.
    pub ops_per_sample: u64,
    /// Frames that must be replayed to serve a mid-extent seek.
    pub worst_case_replay_frames: u64,
}

impl LearnedCost {
    /// Compute the exact decomposition of one learned object.
    #[allow(clippy::field_reassign_with_default)]
    pub fn of(o: &LearnedObject) -> Result<LearnedCost> {
        o.validate()?;
        let model_bytes = o.model.canonical_bytes();
        let model_len = model_bytes.len() as u64;

        let mut c = LearnedCost::default();
        // Framing: fixed fields plus the profile tag. The residual codec id
        // byte is counted in `residual_bytes`, not here.
        c.metadata_bytes = (12 + 1 + 1 + o.profile.tag().len() + 1 + 8 + 4 + 8 + 8 + 4) as u64;
        c.residual_bytes = 1 + o.residual_bytes.len() as u64;
        c.dependency_bytes = (o.dependencies.len() as u64) * 32;
        c.integrity_bytes = 32;

        // Model components. Named components are computed directly; the
        // remainder is graph/dimension metadata, so the sum is exact.
        let (weights, biases, activations, checkpoints, tensor_dims, state_def) =
            model_components(&o.model);
        let named = weights + biases + activations + checkpoints + tensor_dims + state_def;
        if named > model_len {
            return Err(crate::error::Error::internal(
                "learned model accounting exceeds the canonical model length",
            ));
        }
        c.raw_weight_bytes = weights;
        c.canonical_weight_bytes = weights;
        c.bias_bytes = biases;
        c.activation_table_bytes = activations;
        c.checkpoint_definition_bytes = checkpoints;
        c.tensor_dimension_bytes = tensor_dims;
        c.state_definition_bytes = state_def;
        // The remainder is graph/header metadata.
        c.graph_metadata_bytes = model_len - named;
        c.model_bytes = model_len;

        let frames = o.frames;
        let ch = u64::from(o.channels);
        c.raw_sample_bytes = frames.saturating_mul(ch).saturating_mul(4);
        c.canonical_literal_bytes =
            crate::inverse::cost::canonical_u1_literal_bytes(frames, o.channels);
        c.persistent_sample_domain_bytes = 0;
        c.persistent_state_bytes = o.model.state_bytes();
        c.decoded_window_state_bytes = frames.saturating_mul(ch).saturating_mul(4);
        c.ops_per_sample = o.model.ops_per_sample();
        c.worst_case_replay_frames =
            o.model.replay_frames((frames as usize).saturating_sub(1)) as u64;

        c.complete_bytes = c
            .metadata_bytes
            .saturating_add(c.model_bytes)
            .saturating_add(c.residual_bytes)
            .saturating_add(c.dependency_bytes)
            .saturating_add(c.integrity_bytes);
        Ok(c)
    }

    /// The eight-way decomposition must sum to `complete_bytes`.
    pub fn decomposition_is_consistent(&self) -> bool {
        self.metadata_bytes
            .saturating_add(self.model_bytes)
            .saturating_add(self.residual_bytes)
            .saturating_add(self.dependency_bytes)
            .saturating_add(self.integrity_bytes)
            == self.complete_bytes
    }

    /// The model sub-decomposition must sum to `model_bytes`.
    pub fn model_decomposition_is_consistent(&self) -> bool {
        self.canonical_weight_bytes
            .saturating_add(self.bias_bytes)
            .saturating_add(self.scale_bytes)
            .saturating_add(self.activation_table_bytes)
            .saturating_add(self.graph_metadata_bytes)
            .saturating_add(self.tensor_dimension_bytes)
            .saturating_add(self.state_definition_bytes)
            .saturating_add(self.checkpoint_definition_bytes)
            == self.model_bytes
    }
}

/// A shared-model amortization regime (`O.15`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedModelCost {
    /// Bytes of the shared learned model, counted once.
    pub shared_model_bytes: u64,
    /// Bytes of the per-object residual + framing + integrity.
    pub per_object_incremental_bytes: u64,
}

impl SharedModelCost {
    /// Whole-corpus cost for `n` objects under one shared model.
    pub fn corpus_bytes(&self, n: u64) -> u64 {
        self.shared_model_bytes
            .saturating_add(self.per_object_incremental_bytes.saturating_mul(n))
    }

    /// Amortization crossover `N*`: the smallest `N` such that the shared model
    /// plus `N` incremental objects is strictly cheaper than `N` independent
    /// baselines. `None` means no crossover was observed within the search.
    pub fn crossover(&self, independent_bytes_each: u64, max_n: u64) -> Option<u64> {
        if self.per_object_incremental_bytes >= independent_bytes_each {
            return None;
        }
        let mut n = 1u64;
        while n <= max_n {
            if self.corpus_bytes(n) < independent_bytes_each.saturating_mul(n) {
                return Some(n);
            }
            n += 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::finite_field::LinearPredictor;
    use crate::learned::model::LearnedModel;

    #[test]
    fn decomposition_is_exact_and_sums_to_the_canonical_length() {
        let model = LearnedModel::Linear(LinearPredictor {
            channels: 1,
            taps: 4,
            weights: vec![1, 2, 3, 4],
            bias: vec![5],
            block_frames: None,
        });
        let source: Vec<i32> = (0..256).map(|i| ((i * 97) % 401) - 200).collect();
        let o = LearnedObject::from_intrinsic(model, 1, 256, 48_000, Vec::new(), &source).unwrap();
        let cost = LearnedCost::of(&o).unwrap();
        assert!(cost.decomposition_is_consistent());
        assert!(cost.model_decomposition_is_consistent());
        assert_eq!(cost.complete_bytes, o.canonical_bytes().len() as u64);
    }
}

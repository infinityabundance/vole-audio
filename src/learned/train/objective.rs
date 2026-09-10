//! Training objective and Pareto selection vocabulary (`O.25`, `O.26`).
//!
//! Prediction error alone is **not** the objective. Training/search *may* use a
//! weighted estimate, but a weighted score has **proposal authority only**:
//! final selection is by Pareto dominance over measured complete bytes, decode
//! work, seek cost and state, with no single arbitrary score deciding a winner.

use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::object::LearnedObject;

/// Measured objective components for one learned candidate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Objective {
    pub model_bytes: u64,
    pub residual_bytes: u64,
    pub dependency_bytes: u64,
    pub complete_bytes: u64,
    pub eval_ops_per_sample: u64,
    pub seek_replay_frames: u64,
    pub state_bytes: u64,
    pub checkpoint_bytes: u64,
}

impl Objective {
    /// Measure the objective from the canonical object's own accounting.
    pub fn of(o: &LearnedObject) -> Result<Objective> {
        let c = LearnedCost::of(o)?;
        Ok(Objective {
            model_bytes: c.model_bytes,
            residual_bytes: c.residual_bytes,
            dependency_bytes: c.dependency_bytes,
            complete_bytes: c.complete_bytes,
            eval_ops_per_sample: c.ops_per_sample,
            seek_replay_frames: c.worst_case_replay_frames,
            state_bytes: c.persistent_state_bytes,
            checkpoint_bytes: c.checkpoint_definition_bytes,
        })
    }

    /// Weighted proposal score. **Never** a final-selection criterion.
    pub fn weighted(&self, w: &ObjectiveWeights) -> f64 {
        self.complete_bytes as f64
            + w.lambda_eval * self.eval_ops_per_sample as f64
            + w.lambda_seek * self.seek_replay_frames as f64
            + w.lambda_state * self.state_bytes as f64
            + w.lambda_checkpoint * self.checkpoint_bytes as f64
    }

    /// Pareto dominance: `self` dominates `other` when it is no worse in every
    /// reported dimension and strictly better in at least one.
    pub fn dominates(&self, other: &Objective) -> bool {
        let no_worse = self.complete_bytes <= other.complete_bytes
            && self.eval_ops_per_sample <= other.eval_ops_per_sample
            && self.seek_replay_frames <= other.seek_replay_frames
            && self.state_bytes <= other.state_bytes;
        let strictly_better = self.complete_bytes < other.complete_bytes
            || self.eval_ops_per_sample < other.eval_ops_per_sample
            || self.seek_replay_frames < other.seek_replay_frames
            || self.state_bytes < other.state_bytes;
        no_worse && strictly_better
    }

    /// Incomparable when neither dominates the other.
    pub fn incomparable(&self, other: &Objective) -> bool {
        !self.dominates(other) && !other.dominates(self)
    }
}

/// Proposal-only weights (training). Not normative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectiveWeights {
    pub lambda_eval: f64,
    pub lambda_seek: f64,
    pub lambda_state: f64,
    pub lambda_checkpoint: f64,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        ObjectiveWeights {
            lambda_eval: 0.0,
            lambda_seek: 0.0,
            lambda_state: 0.0,
            lambda_checkpoint: 0.0,
        }
    }
}

/// Residual-shape statistics used by residual-cost-aware training (`O.11`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualShape {
    pub values: u64,
    pub zeros: u64,
    pub zero_fraction: f64,
    pub nonzero_density: f64,
    pub mean_abs_nonzero: f64,
    pub max_abs: u64,
    pub longest_zero_run: u64,
    /// A first-order entropy estimate over zigzag varint bytes (bits/value).
    pub varint_entropy_bits: f64,
}

impl ResidualShape {
    /// Measure the shape of a dense residual.
    pub fn of(residual: &[i32]) -> ResidualShape {
        let n = residual.len() as u64;
        let mut zeros = 0u64;
        let mut sum_abs = 0f64;
        let mut max_abs = 0u64;
        let mut run = 0u64;
        let mut longest = 0u64;
        let mut hist = [0u64; 256];
        for &v in residual {
            if v == 0 {
                zeros += 1;
                run += 1;
                longest = longest.max(run);
            } else {
                run = 0;
                let a = i64::from(v).unsigned_abs();
                sum_abs += a as f64;
                max_abs = max_abs.max(a);
            }
            // Zigzag varint first byte (the dominant cost driver).
            let zz = (((i64::from(v)) << 1) ^ ((i64::from(v)) >> 63)) as u64;
            hist[(zz & 0x7F) as usize] += 1;
        }
        let nonzero = n.saturating_sub(zeros);
        let mut entropy = 0f64;
        if n > 0 {
            for &h in &hist {
                if h > 0 {
                    let p = h as f64 / n as f64;
                    entropy -= p * p.log2();
                }
            }
        }
        ResidualShape {
            values: n,
            zeros,
            zero_fraction: if n == 0 { 0.0 } else { zeros as f64 / n as f64 },
            nonzero_density: if n == 0 {
                0.0
            } else {
                nonzero as f64 / n as f64
            },
            mean_abs_nonzero: if nonzero == 0 {
                0.0
            } else {
                sum_abs / nonzero as f64
            },
            max_abs,
            longest_zero_run: longest,
            varint_entropy_bits: entropy,
        }
    }

    /// A cheap differentiable-free proxy for residual coding cost (bits).
    pub fn estimated_bits(&self) -> f64 {
        // Entropy of the first varint byte times the value count, plus a byte
        // per nonzero for the remaining varint bytes.
        self.varint_entropy_bits * self.values as f64
            + self.nonzero_density * self.values as f64 * 8.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::finite_field::LinearPredictor;
    use crate::learned::model::LearnedModel;

    #[test]
    fn pareto_dominance_and_incomparability() {
        let a = Objective {
            complete_bytes: 100,
            eval_ops_per_sample: 10,
            seek_replay_frames: 0,
            state_bytes: 0,
            ..Default::default()
        };
        let b = Objective {
            complete_bytes: 120,
            eval_ops_per_sample: 10,
            ..Default::default()
        };
        assert!(a.dominates(&b));
        assert!(!b.incomparable(&a));
        let c = Objective {
            complete_bytes: 90,
            eval_ops_per_sample: 50,
            ..Default::default()
        };
        assert!(a.incomparable(&c));
    }

    #[test]
    fn residual_shape_measures_sparsity() {
        let mut r = vec![0i32; 100];
        r[3] = 1000;
        r[4] = -1000;
        let s = ResidualShape::of(&r);
        assert_eq!(s.zeros, 98);
        assert!((s.zero_fraction - 0.98).abs() < 1e-9);
        assert_eq!(s.nonzero_density, 0.02);
        assert!(s.estimated_bits() > 0.0);

        let dense: Vec<i32> = (0..100).map(|i| i - 50).collect();
        let d = ResidualShape::of(&dense);
        assert_eq!(d.zeros, 1);
        assert!(d.nonzero_density >= 0.98);
    }

    #[test]
    fn objective_is_measured_from_the_object() {
        let model = LearnedModel::Linear(LinearPredictor {
            channels: 1,
            taps: 2,
            weights: vec![2048, 1024],
            bias: vec![0],
            block_frames: None,
        });
        let source: Vec<i32> = (0..256).map(|i| (i * 13) % 997).collect();
        let o = LearnedObject::from_intrinsic(model, 1, 256, 48_000, Vec::new(), &source).unwrap();
        let obj = Objective::of(&o).unwrap();
        assert_eq!(obj.complete_bytes, o.canonical_bytes().len() as u64);
    }
}

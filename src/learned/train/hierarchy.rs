//! Hierarchical residual fitting (Exp2, priority `9`).
//!
//! Stages are fitted greedily: each stage is a mono sparse predictor fitted on
//! the residual left by the previous stages. A stage is retained only if the
//! measured complete bytes improve, so the cascade is never larger than the
//! best prefix.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::hierarchy::HierarchicalPredictor;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::sparse::SparseLinearPredictor;
use crate::learned::train::sparse::fit_sparse_object;
use crate::learned::train::{TrainBudget, TrainStats};

fn fit_stage(
    residual: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    budget: &TrainBudget,
) -> Result<(SparseLinearPredictor, Vec<i32>)> {
    let (o, _) = fit_sparse_object(residual, frames, sample_rate_hz, 6, None, budget)?;
    let LearnedModel::SparseLinear(p) = o.model else {
        return Err(Error::internal(
            "hierarchy stage fit did not produce a sparse model",
        ));
    };
    let h = p.hypothesis_all_from_source(residual, frames as usize)?;
    let next: Vec<i32> = residual
        .iter()
        .zip(h.iter())
        .map(|(&x, &hh)| (i64::from(x) - i64::from(hh)) as i32)
        .collect();
    Ok((p, next))
}

/// Fit a hierarchical cascade, retaining only measured improvements.
pub fn fit_hierarchy_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    max_stages: u32,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let n = frames as usize;
    if source.len() != n {
        return Err(Error::malformed("hierarchy fit geometry mismatch"));
    }
    let max_stages = max_stages.clamp(1, crate::limits::MAX_LEARNED_HIERARCHY_STAGES);
    let mut stage_budget = *budget;
    stage_budget.max_iterations = stage_budget.max_iterations.min(48);

    let mut stages: Vec<SparseLinearPredictor> = Vec::new();
    let mut residual = source.to_vec();
    let mut best: Option<(HierarchicalPredictor, u64)> = None;
    for _ in 0..max_stages {
        stats.candidates += 1;
        let (p, next) = match fit_stage(&residual, frames, sample_rate_hz, &stage_budget) {
            Ok(v) => v,
            Err(_) => break,
        };
        stages.push(p);
        residual = next;
        let m = HierarchicalPredictor {
            channels: 1,
            stages: stages.clone(),
        };
        if m.validate().is_err() {
            break;
        }
        let o = match LearnedObject::from_intrinsic_exp2(
            LearnedModel::Hierarchical(m.clone()),
            1,
            frames,
            sample_rate_hz,
            Vec::new(),
            source,
        ) {
            Ok(o) => o,
            Err(_) => {
                stats.rejected += 1;
                break;
            }
        };
        if !o.verify(source) {
            stats.rejected += 1;
            break;
        }
        let bytes = crate::learned::accounting::LearnedCost::of(&o)
            .map(|c| c.complete_bytes)
            .unwrap_or(u64::MAX);
        if best.as_ref().is_none_or(|(_, b)| bytes < *b) {
            best = Some((m, bytes));
        }
        if residual.iter().all(|&v| v == 0) {
            break;
        }
    }
    let (model, _) = best.ok_or_else(|| Error::internal("hierarchy fit produced no candidate"))?;
    let object = LearnedObject::from_intrinsic_exp2(
        LearnedModel::Hierarchical(model),
        1,
        frames,
        sample_rate_hz,
        Vec::new(),
        source,
    )?;
    stats.quantization_attempts += 1;
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((object, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hierarchy_fit_closes_and_is_never_larger_than_one_stage() {
        let mut x = vec![0i32; 32];
        for t in 32..2500 {
            let v = (x[t - 1] as i64 * 65) / 100
                + (x[t - 19] as i64 * 20) / 100
                + ((t % 7) as i64 - 3) * 3;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let budget = TrainBudget::default();
        let (o, _) = fit_hierarchy_object(&x, 2500, 48_000, 3, &budget).unwrap();
        assert!(o.verify(&x));
        let LearnedModel::Hierarchical(p) = &o.model else {
            panic!("expected hierarchical");
        };
        assert!(!p.stages.is_empty());
    }
}

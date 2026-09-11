//! Multichannel fitting (Exp2, priority `5`).
//!
//! Tries each reversible channel transform and fits one mono sparse predictor
//! per transformed component. Only the measured canonical winner is returned.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::model::LearnedModel;
use crate::learned::multichannel::{MultichannelPredictor, TRANSFORM_MID_SIDE, TRANSFORM_NONE};
use crate::learned::object::LearnedObject;
use crate::learned::sparse::SparseLinearPredictor;
use crate::learned::train::sparse::fit_sparse_object;
use crate::learned::train::{TrainBudget, TrainStats};

/// A mono sparse fit of one component stream (returns the predictor only).
fn fit_component(
    component: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    budget: &TrainBudget,
) -> Result<(SparseLinearPredictor, TrainStats)> {
    let (o, stats) = fit_sparse_object(component, frames, sample_rate_hz, 8, None, budget)?;
    let LearnedModel::SparseLinear(p) = o.model else {
        return Err(Error::internal(
            "component fit did not produce a sparse model",
        ));
    };
    Ok((p, stats))
}

/// Split an interleaved multichannel signal into components under `transform`.
fn split(interleaved: &[i32], channels: u8, frames: usize, transform: u8) -> Vec<Vec<i32>> {
    let c = usize::from(channels);
    let mut comps = vec![vec![0i32; frames]; c];
    for t in 0..frames {
        match transform {
            TRANSFORM_MID_SIDE => {
                let l = interleaved[t * c];
                let r = interleaved[t * c + 1];
                let s = r.wrapping_sub(l);
                comps[0][t] = l.wrapping_add(s >> 1);
                comps[1][t] = s;
            }
            _ => {
                for cc in 0..c {
                    comps[cc][t] = interleaved[t * c + cc];
                }
            }
        }
    }
    comps
}

/// Fit a multichannel object, choosing the cheapest reversible transform.
pub fn fit_multichannel_object(
    interleaved: &[i32],
    channels: u8,
    frames: u64,
    sample_rate_hz: u32,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let c = usize::from(channels);
    if c == 0 || interleaved.len() != frames as usize * c {
        return Err(Error::malformed("multichannel fit geometry mismatch"));
    }
    let mut transforms = vec![TRANSFORM_NONE];
    if c == 2 {
        transforms.push(TRANSFORM_MID_SIDE);
    }
    // Component fits use a smaller refinement budget: the encoder's global
    // search is bounded, and per-component optimizer passes are the dominant
    // cost.
    let mut component_budget = *budget;
    component_budget.max_iterations = component_budget.max_iterations.min(32);
    let mut best: Option<(MultichannelPredictor, u64)> = None;
    for transform in transforms {
        stats.candidates += 1;
        let comps = split(interleaved, channels, frames as usize, transform);
        let mut predictors = Vec::with_capacity(c);
        let mut ok = true;
        for comp in &comps {
            match fit_component(comp, frames, sample_rate_hz, &component_budget) {
                Ok((p, st)) => {
                    stats.merge(&st);
                    predictors.push(p);
                }
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let m = MultichannelPredictor {
            channels,
            transform,
            predictors,
        };
        if m.validate().is_err() {
            continue;
        }
        let o = match LearnedObject::from_intrinsic_exp2(
            LearnedModel::Multichannel(m.clone()),
            channels,
            frames,
            sample_rate_hz,
            Vec::new(),
            interleaved,
        ) {
            Ok(o) => o,
            Err(_) => {
                stats.rejected += 1;
                continue;
            }
        };
        if !o.verify(interleaved) {
            stats.rejected += 1;
            continue;
        }
        let bytes = crate::learned::accounting::LearnedCost::of(&o)
            .map(|c| c.complete_bytes)
            .unwrap_or(u64::MAX);
        if best.as_ref().is_none_or(|(_, bb)| bytes < *bb) {
            best = Some((m, bytes));
        }
    }
    let (model, _) =
        best.ok_or_else(|| Error::internal("multichannel fit produced no candidate"))?;
    let object = LearnedObject::from_intrinsic_exp2(
        LearnedModel::Multichannel(model),
        channels,
        frames,
        sample_rate_hz,
        Vec::new(),
        interleaved,
    )?;
    stats.quantization_attempts += 1;
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((object, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multichannel_fit_closes_for_correlated_stereo() {
        let mut v = Vec::new();
        for t in 0..2000i64 {
            let l = ((t * 89) % 15013) as i32 - 7000;
            let r = l / 3 + ((t * 7) % 233) as i32 - 100;
            v.push(l);
            v.push(r);
        }
        let budget = TrainBudget::default();
        let (o, _) = fit_multichannel_object(&v, 2, 2000, 48_000, &budget).unwrap();
        assert!(o.verify(&v));
    }
}

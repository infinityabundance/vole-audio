//! Multichannel fitting (Exp2, priority `5`).
//!
//! Tries each reversible channel transform and fits one mono sparse predictor
//! per transformed component. Only the measured canonical winner is returned.
//!
//! Phase 6's `RleAwareChannelTransform` orders the stereo ladder by the
//! downstream run/zero topology of the transformed components (zero density,
//! zero runs, longest zero run, equal neighbours) rather than by residual
//! variance. Topology only proposes and orders; the exact canonical bytes of
//! each fully fitted candidate still decide the winner.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::model::LearnedModel;
use crate::learned::multichannel::{
    MultichannelPredictor, STEREO_TRANSFORMS, TRANSFORM_NONE, channel_topology, transform_frame,
};
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
        if c == 2 {
            let mut frame = [0i32; 2];
            let mut out = [0i32; 2];
            frame[0] = interleaved[t * c];
            frame[1] = interleaved[t * c + 1];
            transform_frame(transform, &frame, &mut out);
            comps[0][t] = out[0];
            comps[1][t] = out[1];
        } else {
            for cc in 0..c {
                comps[cc][t] = interleaved[t * c + cc];
            }
        }
    }
    comps
}

/// The transform ladder for a channel count, in canonical tie order.
fn transform_ladder(channels: u8) -> Vec<u8> {
    if channels == 2 {
        STEREO_TRANSFORMS.to_vec()
    } else {
        vec![TRANSFORM_NONE]
    }
}

/// Order the transform ladder by downstream run/zero degeneracy (descending
/// score, ties by ascending id). Falls back to canonical order when topology
/// cannot be measured.
fn topology_ranked_ladder(interleaved: &[i32], channels: u8, frames: usize) -> Vec<u8> {
    let mut ranked: Vec<(u64, u8)> = transform_ladder(channels)
        .into_iter()
        .map(|tf| {
            let score = channel_topology(interleaved, channels, frames, tf)
                .map(|t| t.rle_score())
                .unwrap_or(0);
            (score, tf)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked.into_iter().map(|(_, tf)| tf).collect()
}

/// Fit, close and measure one reversible transform. `None` when the transform
/// cannot produce a verified exact object.
pub fn measure_multichannel_transform(
    interleaved: &[i32],
    channels: u8,
    frames: u64,
    sample_rate_hz: u32,
    transform: u8,
    budget: &TrainBudget,
) -> Result<Option<(MultichannelPredictor, u64)>> {
    let c = usize::from(channels);
    if c == 0 || interleaved.len() != frames as usize * c {
        return Err(Error::malformed("multichannel fit geometry mismatch"));
    }
    // Component fits use a smaller refinement budget: the encoder's global
    // search is bounded, and per-component optimizer passes are the dominant
    // cost.
    let mut component_budget = *budget;
    component_budget.max_iterations = component_budget.max_iterations.min(32);
    let comps = split(interleaved, channels, frames as usize, transform);
    let mut predictors = Vec::with_capacity(c);
    for comp in &comps {
        match fit_component(comp, frames, sample_rate_hz, &component_budget) {
            Ok((p, _)) => predictors.push(p),
            Err(_) => return Ok(None),
        }
    }
    let m = MultichannelPredictor {
        channels,
        transform,
        predictors,
    };
    if m.validate().is_err() {
        return Ok(None);
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
        Err(_) => return Ok(None),
    };
    if !o.verify(interleaved) {
        return Ok(None);
    }
    let bytes = crate::learned::accounting::LearnedCost::of(&o)
        .map(|c| c.complete_bytes)
        .unwrap_or(u64::MAX);
    Ok(Some((m, bytes)))
}

/// Fit a multichannel object, choosing the cheapest reversible transform from
/// the topology-ordered ladder.
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
    let ladder = topology_ranked_ladder(interleaved, channels, frames as usize);
    let mut best: Option<(MultichannelPredictor, u64)> = None;
    for transform in ladder {
        stats.candidates += 1;
        match measure_multichannel_transform(
            interleaved,
            channels,
            frames,
            sample_rate_hz,
            transform,
            budget,
        ) {
            Ok(Some((m, bytes))) => {
                if best.as_ref().is_none_or(|(_, bb)| bytes < *bb) {
                    best = Some((m, bytes));
                }
            }
            Ok(None) => {
                stats.rejected += 1;
            }
            Err(_) => {
                stats.rejected += 1;
            }
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

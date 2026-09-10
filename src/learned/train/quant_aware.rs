//! Quantization-aware training (`O.23`).
//!
//! Post-training quantization can destroy a family that floated well. This
//! module steers the fit by the behaviour of the **canonical quantized**
//! evaluator: the cost optimized is the actual encoded residual size of the
//! compiled integer object, so what improves during training is what is stored.
//!
//! The float model is never the archived semantic object; only the compiled
//! integer hypothesis is.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::accounting::LearnedCost;
use crate::learned::finite_field::LinearPredictor;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::quantize::compile_linear;
use crate::learned::train::linear::fit_ridge;
use crate::learned::train::optimizer::coordinate_descent_weights;
use crate::learned::train::{TrainBudget, TrainStats};

fn build(
    weights: &[i16],
    base: &LinearPredictor,
    source: &[i32],
    channels: u8,
    frames: u64,
    sample_rate_hz: u32,
) -> Result<LearnedObject> {
    let mut p = base.clone();
    p.weights = weights.to_vec();
    LearnedObject::from_intrinsic(
        LearnedModel::Linear(p),
        channels,
        frames,
        sample_rate_hz,
        Vec::new(),
        source,
    )
}

fn measured_cost(
    weights: &[i16],
    base: &LinearPredictor,
    source: &[i32],
    channels: u8,
    frames: u64,
    sample_rate_hz: u32,
) -> u64 {
    match build(weights, base, source, channels, frames, sample_rate_hz) {
        Ok(o) => LearnedCost::of(&o)
            .map(|c| c.complete_bytes)
            .unwrap_or(u64::MAX),
        Err(_) => u64::MAX,
    }
}

/// Fit a linear predictor and then refine its **quantized** weights against the
/// actual encoded residual cost (quantization-aware training).
#[allow(clippy::too_many_arguments)]
pub fn fit_qat_linear(
    source: &[i32],
    channels: u8,
    frames: u64,
    sample_rate_hz: u32,
    taps: u16,
    block_frames: Option<u32>,
    analysis_frames: usize,
    budget: &TrainBudget,
    refine_iterations: u64,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    let sw = Stopwatch::start();
    let mut stats = TrainStats {
        candidates: 1,
        quantization_attempts: 1,
        ..Default::default()
    };
    let fit = fit_ridge(
        source,
        channels,
        frames as usize,
        taps,
        block_frames,
        analysis_frames,
        budget.ridge_lambda,
    )?;
    let base = compile_linear(channels, taps, block_frames, &fit.weights, &fit.bias);
    base.validate()?;

    // Coordinate descent on the *quantized* weights, minimizing the real
    // encoded object size. Bias stays at its fitted value.
    let mut evaluations = 0u64;
    let (best_weights, iters) =
        coordinate_descent_weights(&base.weights, 1, refine_iterations, |candidate| {
            evaluations += 1;
            measured_cost(candidate, &base, source, channels, frames, sample_rate_hz) as f64
        });
    stats.iterations += iters;
    stats.candidates += evaluations;

    let candidate = build(
        &best_weights,
        &base,
        source,
        channels,
        frames,
        sample_rate_hz,
    )?;
    let base_object = build(
        &base.weights,
        &base,
        source,
        channels,
        frames,
        sample_rate_hz,
    )?;
    let chosen = if LearnedCost::of(&candidate)?.complete_bytes
        <= LearnedCost::of(&base_object)?.complete_bytes
    {
        candidate
    } else {
        base_object
    };
    if !chosen.verify(source) {
        return Err(Error::internal(
            "quantization-aware refinement broke exact closure",
        ));
    }
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((chosen, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qat_never_returns_a_larger_object_than_its_base_and_is_deterministic() {
        let x: Vec<i32> = (0..1024)
            .map(|i| {
                let t = i as i64;
                ((t * t / 7) % 20011) as i32 - 10000
            })
            .collect();
        let budget = TrainBudget::default();
        let (a, sa) = fit_qat_linear(&x, 1, 1024, 48_000, 4, None, 1024, &budget, 32).unwrap();
        assert!(a.verify(&x));
        assert!(sa.quantization_attempts >= 1);
        assert!(sa.iterations > 0);
        let (b, _) = fit_qat_linear(&x, 1, 1024, 48_000, 4, None, 1024, &budget, 32).unwrap();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }
}

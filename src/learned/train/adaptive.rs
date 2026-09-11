//! Backward-adaptive fitting (Exp2, priority `10`).
//!
//! The initial coefficients come from a ridge fit; the frozen sign-sign step is
//! then chosen from a small canonical set by measured complete bytes. Only the
//! measured winner is returned.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::adaptive::AdaptivePredictor;
use crate::learned::arithmetic::{quantize_bias, quantize_weight};
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::train::linear::fit_ridge;
use crate::learned::train::{TrainBudget, TrainStats};

/// Canonical sign-sign step candidates (Q12 units).
pub const ADAPTIVE_STEPS: [i16; 4] = [1, 2, 4, 8];

/// Fit a backward-adaptive predictor object.
pub fn fit_adaptive_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    taps: u16,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let n = frames as usize;
    if source.len() != n {
        return Err(Error::malformed("adaptive fit geometry mismatch"));
    }
    let fit = fit_ridge(source, 1, n, taps, None, n, budget.ridge_lambda)?;
    let init_weights: Vec<i16> = fit.weights.iter().map(|&w| quantize_weight(w)).collect();
    let init_bias = quantize_bias(fit.bias[0]);
    let mut best: Option<(AdaptivePredictor, u64)> = None;
    for &step in &ADAPTIVE_STEPS {
        stats.candidates += 1;
        let p = AdaptivePredictor {
            channels: 1,
            taps,
            init_weights: init_weights.clone(),
            init_bias,
            step,
            block_frames: None,
        };
        if p.validate().is_err() {
            continue;
        }
        let o = match LearnedObject::from_intrinsic_exp2(
            LearnedModel::Adaptive(p.clone()),
            1,
            frames,
            sample_rate_hz,
            Vec::new(),
            source,
        ) {
            Ok(o) => o,
            Err(_) => {
                stats.rejected += 1;
                continue;
            }
        };
        if !o.verify(source) {
            stats.rejected += 1;
            continue;
        }
        let bytes = crate::learned::accounting::LearnedCost::of(&o)
            .map(|c| c.complete_bytes)
            .unwrap_or(u64::MAX);
        if best.as_ref().is_none_or(|(_, b)| bytes < *b) {
            best = Some((p, bytes));
        }
    }
    let (model, _) = best.ok_or_else(|| Error::internal("adaptive fit produced no candidate"))?;
    let object = LearnedObject::from_intrinsic_exp2(
        LearnedModel::Adaptive(model),
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
    fn adaptive_fit_closes_on_drifting_ar() {
        let mut x = vec![0i32; 4];
        for t in 4..3000 {
            let a = 0.5 + 0.15 * ((t / 400) as f64 * 0.1);
            let v = (x[t - 1] as f64 * a + x[t - 2] as f64 * 0.2 + ((t % 9) as f64 - 4.0) * 4.0)
                .round() as i64;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let budget = TrainBudget::default();
        let (o, _) = fit_adaptive_object(&x, 3000, 48_000, 8, &budget).unwrap();
        assert!(o.verify(&x));
    }
}

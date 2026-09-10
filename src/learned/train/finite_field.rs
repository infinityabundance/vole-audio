//! Nonlinear finite-field fitting (`O.8`).
//!
//! The nonlinear family is introduced only after the linear family has an
//! evidence record. Fitting is deliberately specialized and small: a linear
//! pre-activation from ridge regression, a piecewise-linear activation fitted
//! by binning the pre-activation against the target, and an optional bounded
//! coordinate-descent refinement of the quantized weights against the actual
//! encoded residual size.
//!
//! Mono only in this build (`O.17` establishes mono first); a multichannel
//! request is refused explicitly rather than handled approximately.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::accounting::LearnedCost;
use crate::learned::arithmetic::{Activation, quantize_bias, quantize_weight};
use crate::learned::graph::{DenseLayer, NonlinearGraph};
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::train::linear::fit_ridge;
use crate::learned::train::optimizer::coordinate_descent_weights;
use crate::learned::train::{TrainBudget, TrainStats};

/// Fit a mono nonlinear finite-field predictor.
#[allow(clippy::too_many_arguments)]
pub fn fit_nonlinear_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    taps: u16,
    bins: u32,
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
    let frames_usize = frames as usize;
    if source.len() != frames_usize {
        return Err(Error::malformed(
            "nonlinear fitting is mono-only in this build",
        ));
    }
    let k = usize::from(taps);
    if k == 0 {
        return Err(Error::malformed("nonlinear fit requires at least one tap"));
    }
    let analysis = frames_usize.min(budget.max_taps as usize * 64 + 1024);
    let fit = fit_ridge(
        source,
        1,
        frames_usize,
        taps,
        None,
        analysis,
        budget.ridge_lambda,
    )?;
    let wq: Vec<i16> = fit.weights.iter().map(|&w| quantize_weight(w)).collect();
    let bq = quantize_bias(fit.bias[0]);

    // Pre-activation series (Q12 accumulators), open-loop over the source.
    let mut accs = Vec::with_capacity(frames_usize);
    for t in 0..frames_usize {
        let mut a: i64 = i64::from(bq);
        for kk in 1..=k {
            let idx = t as i64 - kk as i64;
            if idx >= 0 {
                a += i64::from(wq[kk - 1]) * i64::from(source[idx as usize]);
            }
        }
        accs.push(a);
    }
    let amin = accs.iter().copied().min().unwrap_or(0);
    let amax = accs.iter().copied().max().unwrap_or(0);
    let nbins = bins.clamp(2, 64) as usize;
    let mut sx = vec![0i128; nbins];
    let mut sy = vec![0i128; nbins];
    let mut cnt = vec![0u64; nbins];
    for (t, &a) in accs.iter().enumerate() {
        let frac = if amax > amin {
            (a - amin) as f64 / (amax - amin) as f64
        } else {
            0.0
        };
        let b = ((frac * (nbins as f64 - 1.0)).round() as usize).min(nbins - 1);
        sx[b] += i128::from(a);
        sy[b] += i128::from(source[t]) << 12;
        cnt[b] += 1;
    }
    let mut points: Vec<(i32, i32)> = Vec::new();
    for b in 0..nbins {
        if cnt[b] == 0 {
            continue;
        }
        let x = (sx[b] / i128::from(cnt[b])) as i64;
        let y = (sy[b] / i128::from(cnt[b])) as i64;
        let x = x.clamp(i64::from(i32::MIN + 1), i64::from(i32::MAX - 1)) as i32;
        let y = y.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        // Strictly ascending breakpoints are required by the canonical
        // activation domain; collapse duplicates honestly.
        if points.last().is_none_or(|&(lx, _)| x > lx) {
            points.push((x, y));
        }
    }
    if points.len() < 2 {
        // Degenerate pre-activation range: a two-point identity in Q12.
        points = vec![(-(1 << 20), -(1 << 20)), (1 << 20, 1 << 20)];
    }

    let graph = NonlinearGraph {
        channels: 1,
        taps,
        layers: vec![DenseLayer {
            in_dim: u32::from(taps),
            out_dim: 1,
            weights: wq.clone(),
            bias: vec![bq],
            activation: Activation::PiecewiseLinear {
                points: points.clone(),
            },
        }],
        block_frames: None,
    };
    graph.validate()?;

    // Baseline graph object.
    let base_object = LearnedObject::from_intrinsic(
        LearnedModel::Nonlinear(graph.clone()),
        1,
        frames,
        sample_rate_hz,
        Vec::new(),
        source,
    );
    let mut best = match base_object {
        Ok(o) if o.verify(source) => o,
        _ => {
            // Fall back to the (always valid) linear object on this fit.
            return crate::learned::train::linear::fit_linear_object(
                source,
                1,
                frames,
                sample_rate_hz,
                taps,
                None,
                analysis,
                budget,
            );
        }
    };
    let base_cost = LearnedCost::of(&best)?.complete_bytes;

    // Bounded residual-aware refinement of the quantized weights.
    let mut evaluations = 0u64;
    let (refined, iters) = coordinate_descent_weights(&wq, 1, refine_iterations, |candidate| {
        evaluations += 1;
        let mut g = graph.clone();
        g.layers[0].weights = candidate.to_vec();
        match LearnedObject::from_intrinsic(
            LearnedModel::Nonlinear(g),
            1,
            frames,
            sample_rate_hz,
            Vec::new(),
            source,
        ) {
            Ok(o) => LearnedCost::of(&o)
                .map(|c| c.complete_bytes)
                .unwrap_or(u64::MAX) as f64,
            Err(_) => f64::MAX,
        }
    });
    stats.iterations += iters;
    stats.candidates += evaluations;
    let mut refined_graph = graph;
    refined_graph.layers[0].weights = refined;
    match LearnedObject::from_intrinsic(
        LearnedModel::Nonlinear(refined_graph),
        1,
        frames,
        sample_rate_hz,
        Vec::new(),
        source,
    ) {
        Ok(o)
            if o.verify(source)
                && LearnedCost::of(&o)
                    .map(|c| c.complete_bytes)
                    .unwrap_or(u64::MAX)
                    <= base_cost =>
        {
            best = o;
        }
        _ => {}
    }
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((best, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonlinear_fit_closes_exactly_or_falls_back_honestly() {
        // A clipped sinusoid-like sequence: genuinely nonlinear.
        let x: Vec<i32> = (0..1500)
            .map(|i| {
                let t = i as f64 / 40.0;
                ((t.sin() * 1_000_000.0) as i32).clamp(-400_000, 400_000)
            })
            .collect();
        let budget = TrainBudget::default();
        let (o, _stats) = fit_nonlinear_object(&x, 1500, 48_000, 8, 16, &budget, 24).unwrap();
        assert!(o.verify(&x));
        // Either a nonlinear graph or the honest linear fallback.
        let name = o.model.kind_name();
        assert!(name == "nonlinear_finite_field" || name == "linear_finite_field");
    }

    #[test]
    fn multichannel_nonlinear_is_refused_explicitly() {
        let budget = TrainBudget::default();
        let x = vec![0i32; 100];
        let r = fit_nonlinear_object(&x, 50, 48_000, 4, 8, &budget, 0);
        assert!(r.is_err());
    }
}

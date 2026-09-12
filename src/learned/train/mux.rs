//! Hard-selection expert-mux fitting (Phase 6 mechanism 4, `SampleExpertMux`).
//!
//! The expert set comes from a small canonical ladder of backward-adaptive
//! sign-sign predictors (short/fast versus long/slow). Because every expert is
//! stepped from the reconstructed value, the experts' trajectories do not depend
//! on the selector, so the per-microgroup choice is made exactly: for each group
//! the expert with the cheapest measured residual proxy is selected. The
//! canonical expert set, group size and selectors are then all charged through
//! the complete-byte objective, and only the measured winner is returned.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::adaptive::AdaptivePredictor;
use crate::learned::arithmetic::{quantize_bias, quantize_weight};
use crate::learned::carousel::proxy_residual_bits;
use crate::learned::model::LearnedModel;
use crate::learned::mux::ExpertMuxPredictor;
use crate::learned::object::LearnedObject;
use crate::learned::train::linear::fit_ridge;
use crate::learned::train::{TrainBudget, TrainStats};
use std::collections::HashMap;

/// Canonical expert pairs: `((taps, step), (taps, step))`, fast versus slow.
pub const MUX_PAIRS: [((u16, i16), (u16, i16)); 4] = [
    ((8, 8), (8, 1)),
    ((16, 8), (16, 1)),
    ((8, 8), (16, 1)),
    ((16, 4), (8, 2)),
];

/// Canonical microgroup sizes (frames).
pub const MUX_GROUPS: [u32; 3] = [64, 128, 256];

/// Fit a hard-selection expert mux object over the canonical ladder.
pub fn fit_expert_mux_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let n = frames as usize;
    if source.len() != n {
        return Err(Error::malformed("expert mux fit geometry mismatch"));
    }

    // Ridge initialisation per tap count, cached.
    let mut ridge: HashMap<u16, (Vec<i16>, i32)> = HashMap::new();
    let expert = |taps: u16, step: i16, ridge: &mut HashMap<u16, (Vec<i16>, i32)>| {
        let (w, b) = ridge.entry(taps).or_insert_with(|| {
            let fit = fit_ridge(source, 1, n, taps, None, n, budget.ridge_lambda)
                .expect("ridge fit for mux expert");
            let w: Vec<i16> = fit.weights.iter().map(|&x| quantize_weight(x)).collect();
            (w, quantize_bias(fit.bias[0]))
        });
        AdaptivePredictor {
            channels: 1,
            taps,
            init_weights: w.clone(),
            init_bias: *b,
            step,
            block_frames: None,
        }
    };

    // Expert hypotheses are independent of the selectors, so compute each once.
    let mut hyp: HashMap<(u16, i16), Vec<i32>> = HashMap::new();
    let hypothesis = |e: &AdaptivePredictor, hyp: &mut HashMap<(u16, i16), Vec<i32>>| {
        hyp.entry((e.taps, e.step))
            .or_insert_with(|| {
                e.hypothesis_all_from_source(source, n)
                    .expect("mux expert hypothesis")
            })
            .clone()
    };

    let mut best: Option<(ExpertMuxPredictor, u64)> = None;
    for (ca, cb) in MUX_PAIRS {
        let ea = expert(ca.0, ca.1, &mut ridge);
        let eb = expert(cb.0, cb.1, &mut ridge);
        let ha = hypothesis(&ea, &mut hyp);
        let hb = hypothesis(&eb, &mut hyp);
        for &group in &MUX_GROUPS {
            stats.candidates += 1;
            let groups = n.div_ceil(group as usize).max(1);
            let mut selectors = Vec::with_capacity(groups);
            for g in 0..groups {
                let lo = g * group as usize;
                let hi = (lo + group as usize).min(n);
                if lo >= hi {
                    break;
                }
                let ra: Vec<i32> = (lo..hi).map(|t| source[t] - ha[t]).collect();
                let rb: Vec<i32> = (lo..hi).map(|t| source[t] - hb[t]).collect();
                let ca_bits = proxy_residual_bits(&ra);
                let cb_bits = proxy_residual_bits(&rb);
                selectors.push(if ca_bits <= cb_bits { 0u8 } else { 1u8 });
            }
            let m = ExpertMuxPredictor {
                channels: 1,
                group_frames: group,
                experts: vec![ea.clone(), eb.clone()],
                selectors,
            };
            if m.validate().is_err() {
                stats.rejected += 1;
                continue;
            }
            let o = match LearnedObject::from_intrinsic_exp2(
                LearnedModel::ExpertMux(m.clone()),
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
                best = Some((m, bytes));
            }
        }
    }

    let (model, _) = best.ok_or_else(|| Error::internal("expert mux fit produced no candidate"))?;
    let object = LearnedObject::from_intrinsic_exp2(
        LearnedModel::ExpertMux(model),
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

    fn signal(n: usize, switches: &[(usize, f64)], seed: u64) -> Vec<i32> {
        let mut st = seed | 1;
        let mut x = 0i64;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = switches
                .iter()
                .rev()
                .find(|(at, _)| i >= *at)
                .map(|(_, a)| *a)
                .unwrap_or(0.9);
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (((st >> 33) & 0xffff) as i64 - 32768) >> 4;
            x = ((a * x as f64).round() as i64) + noise;
            x = x.clamp(-(1 << 22), 1 << 22);
            out.push(x as i32);
        }
        out
    }

    #[test]
    fn expert_mux_fit_closes_on_regime_switching_ar() {
        let x = signal(4096, &[(0, 0.95), (2048, -0.6)], 3);
        let budget = TrainBudget::default();
        let (o, _) = fit_expert_mux_object(&x, 4096, 48_000, &budget).unwrap();
        assert!(o.verify(&x));
    }
}

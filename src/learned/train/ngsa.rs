//! Natural-gradient adaptive fitting (Seal A0).
//!
//! Initial coefficients come from the same ridge fit the sign-sign family uses;
//! the learning rate and correlation-EMA shift are then chosen from small
//! canonical sets by measured complete bytes. Only the measured winner is
//! returned. Nothing here is normative: the canonical object carries only the
//! compiled integer hypothesis.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::arithmetic::{quantize_bias, quantize_weight};
use crate::learned::model::LearnedModel;
use crate::learned::ngsa::NgsaPredictor;
use crate::learned::object::LearnedObject;
use crate::learned::train::linear::fit_ridge;
use crate::learned::train::{TrainBudget, TrainStats};

/// Canonical natural-gradient sign step candidates (Q12 units).
pub const NGSA_STEPS: [i16; 4] = [64, 128, 256, 512];
/// Canonical correlation-EMA shift candidates.
pub const NGSA_RHO_SHIFTS: [u8; 2] = [5, 7];

/// Fit a natural-gradient adaptive predictor object.
pub fn fit_ngsa_object(
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
        return Err(Error::malformed("ngsa fit geometry mismatch"));
    }
    let fit = fit_ridge(source, 1, n, taps, None, n, budget.ridge_lambda)?;
    let init_weights: Vec<i16> = fit.weights.iter().map(|&w| quantize_weight(w)).collect();
    let init_bias = quantize_bias(fit.bias[0]);
    let mut best: Option<(NgsaPredictor, u64)> = None;
    for &step in &NGSA_STEPS {
        for &rho_shift in &NGSA_RHO_SHIFTS {
            stats.candidates += 1;
            let p = NgsaPredictor {
                channels: 1,
                taps,
                init_weights: init_weights.clone(),
                init_bias,
                step,
                rho_shift,
                block_frames: None,
            };
            if p.validate().is_err() {
                continue;
            }
            let o = match LearnedObject::from_intrinsic_exp2(
                LearnedModel::Ngsa(p.clone()),
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
    }
    let (model, _) = best.ok_or_else(|| Error::internal("ngsa fit produced no candidate"))?;
    let object = LearnedObject::from_intrinsic_exp2(
        LearnedModel::Ngsa(model),
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
    fn ngsa_fit_closes_on_drifting_ar() {
        let mut x = vec![0i32; 4];
        for t in 4..3000 {
            let a = 0.5 + 0.15 * ((t / 400) as f64 * 0.1);
            let v = (x[t - 1] as f64 * a + x[t - 2] as f64 * 0.2 + ((t % 9) as f64 - 4.0) * 4.0)
                .round() as i64;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let budget = TrainBudget::default();
        let (o, st) = fit_ngsa_object(&x, 3000, 48_000, 8, &budget).unwrap();
        assert!(st.candidates >= 1);
        assert!(o.verify(&x));
    }

    #[test]
    fn ngsa_is_stable_on_correlated_input() {
        // An AR(1)-with-noise signal: every step/rho configuration must stay
        // near the noise floor (no drift, no saturation).
        let mut x = vec![0i32; 8];
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for t in 8..8192 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = (s % 2001) as i64 - 1000;
            let v = (x[t - 1] as f64 * 0.95 + noise as f64).round() as i64;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let budget = TrainBudget::default();
        let fit = crate::learned::train::linear::fit_ridge(
            &x,
            1,
            8192,
            8,
            None,
            8192,
            budget.ridge_lambda,
        )
        .unwrap();
        let iw: Vec<i16> = fit.weights.iter().map(|&w| quantize_weight(w)).collect();
        let ib = quantize_bias(fit.bias[0]);
        for step in NGSA_STEPS {
            for &rho_shift in &NGSA_RHO_SHIFTS {
                let p = NgsaPredictor {
                    channels: 1,
                    taps: 8,
                    init_weights: iw.clone(),
                    init_bias: ib,
                    step,
                    rho_shift,
                    block_frames: None,
                };
                let o = LearnedObject::from_intrinsic_exp2(
                    LearnedModel::Ngsa(p),
                    1,
                    8192,
                    48_000,
                    Vec::new(),
                    &x,
                )
                .unwrap();
                assert!(o.verify(&x));
                let r = o.residual().unwrap();
                let early: f64 = r[..128]
                    .iter()
                    .map(|&v| f64::from(v.unsigned_abs()))
                    .sum::<f64>()
                    / 128.0;
                let late: f64 = r[r.len() - 128..]
                    .iter()
                    .map(|&v| f64::from(v.unsigned_abs()))
                    .sum::<f64>()
                    / 128.0;
                assert!(late < 3.0 * early, "step={step} drift {early} -> {late}");
            }
        }
    }

    #[test]
    fn ngsa_fit_reports_length_failure() {
        // 16k-sample speech-like signal at taps 16: must either close or report
        // a diagnostic reason (regression guard for the court failure).
        let mut s = 0x1234_5678u64;
        let x: Vec<i32> = (0..16384)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 40_000) as i64 - 20_000) as i32
            })
            .collect();
        let budget = TrainBudget::default();
        match fit_ngsa_object(&x, 16384, 48_000, 16, &budget) {
            Ok((o, _)) => assert!(o.verify(&x)),
            Err(e) => panic!("ngsa fit error: {e}"),
        }
    }
}

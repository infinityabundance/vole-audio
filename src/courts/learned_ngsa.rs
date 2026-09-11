//! `court learned-ngsa` — fourth-pass **Seal A0** natural-gradient experiment.
//!
//! The existing backward-adaptive family is a frozen integer sign-sign LMS whose
//! recorded negative was slow convergence. Seal A0 adds
//! [`crate::learned::ngsa::NgsaPredictor`], a clean-room fixed-point
//! natural-gradient predictor preconditioned by an O(p) AR(1) inverse. This
//! court is the falsifiable experiment: the two families are fitted on the same
//! windows with the same ridge initialisation and compared by **actual complete
//! bytes** and by residual magnitude (a convergence proxy).
//!
//! Two populations:
//!
//! * a synthetic drifting AR(2) whose coefficients change every 400 samples —
//!   the regime where a stationary FIR cannot follow and adaptation matters;
//! * the frozen real-speech effectiveness clips (the held-out Mode-C split is
//!   untouched).
//!
//! Nothing here is mixed into the E-ladder: this is an independent predictor-side
//! branch. Only deterministic integers enter the frozen projection.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::object::LearnedObject;
use crate::learned::train::TrainBudget;
use crate::learned::train::adaptive::fit_adaptive_object;
use crate::learned::train::ngsa::fit_ngsa_object;
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_NGSA_SHA256: &str =
    "99755e513542aa457273aa4d8f144070d14ce5cc2d89197fb25e9d66a50bd8cd";

/// Tap counts tried by both families.
const TAP_LADDER: [u16; 2] = [8, 16];

/// One fitted family result.
struct Fit {
    object: LearnedObject,
    complete_bytes: u64,
    residual_bytes: u64,
    mean_abs: f64,
}

fn evaluate_fit(o: LearnedObject, source: &[i32]) -> Result<Fit> {
    let cost = LearnedCost::of(&o)?;
    let residual = o.residual()?;
    let mean_abs = if residual.is_empty() {
        0.0
    } else {
        residual
            .iter()
            .map(|&v| f64::from(v.unsigned_abs()))
            .sum::<f64>()
            / residual.len() as f64
    };
    let _ = source;
    Ok(Fit {
        object: o,
        complete_bytes: cost.complete_bytes,
        residual_bytes: cost.residual_bytes,
        mean_abs,
    })
}

/// Fit both families over the tap ladder and keep each family's cheapest.
fn best_pair(source: &[i32], frames: u64, rate: u32, budget: &TrainBudget) -> Result<(Fit, Fit)> {
    let mut adaptive: Option<Fit> = None;
    let mut ngsa: Option<Fit> = None;
    for taps in TAP_LADDER {
        if let Ok((o, _)) = fit_adaptive_object(source, frames, rate, taps, budget) {
            let f = evaluate_fit(o, source)?;
            if adaptive
                .as_ref()
                .is_none_or(|b| f.complete_bytes < b.complete_bytes)
            {
                adaptive = Some(f);
            }
        }
        if let Ok((o, _)) = fit_ngsa_object(source, frames, rate, taps, budget) {
            let f = evaluate_fit(o, source)?;
            if ngsa
                .as_ref()
                .is_none_or(|b| f.complete_bytes < b.complete_bytes)
            {
                ngsa = Some(f);
            }
        }
    }
    let adaptive = adaptive.ok_or_else(|| crate::error::Error::internal("adaptive fit failed"))?;
    let ngsa = ngsa.ok_or_else(|| crate::error::Error::internal("ngsa fit failed"))?;
    Ok((adaptive, ngsa))
}

fn pair_json(id: &str, population: &str, a: &Fit, n: &Fit) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "population": population,
        "adaptive_bytes": a.complete_bytes,
        "ngsa_bytes": n.complete_bytes,
        "adaptive_residual_bytes": a.residual_bytes,
        "ngsa_residual_bytes": n.residual_bytes,
        "adaptive_mean_abs": a.mean_abs,
        "ngsa_mean_abs": n.mean_abs,
        "byte_gain": a.complete_bytes as i64 - n.complete_bytes as i64,
        "mean_abs_reduction": a.mean_abs - n.mean_abs,
        "ngsa_wins": n.complete_bytes < a.complete_bytes,
    })
}

/// A synthetic drifting **stable** AR(2): coefficients change every 400 samples.
fn drifting_ar2(n: usize) -> Vec<i32> {
    let mut x = vec![0i32; 4];
    for t in 4..n {
        let regime = (t / 400) as f64;
        let a = 0.50 + 0.06 * (regime % 4.0);
        let b = 0.18 - 0.02 * (regime % 3.0);
        let v = (x[t - 1] as f64 * a + x[t - 2] as f64 * b + ((t % 9) as f64 - 4.0) * 400.0).round()
            as i64;
        x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
    }
    x
}

/// Run the court; writes `receipts/learned-ngsa/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.ngsa.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    if !available() || !decoder_available() {
        return common::finish_exp2(
            "learned-ngsa",
            receipts_root,
            LEARNED_NGSA_SHA256,
            &projection,
            Verdict::Inconclusive,
            "real corpus audio or the external `flac` decoder is absent".to_string(),
            vec![(
                "limitations",
                serde_json::json!([
                    "the real-speech population requires the uncommitted LibriSpeech audio bulk \
                     under corpus/real/ and the external `flac` decoder; the held-out Mode-C split \
                     is deliberately not used by this branch"
                ]),
            )],
        );
    }

    let mut budget = common::train_budget();
    budget.max_iterations = 128;
    let scratch = PathBuf::from("target/real-corpus/scratch");
    let clips = load_cases(&effectiveness_clips(), 8, &scratch)?;
    let mut rows = Vec::new();
    let mut all_exact = true;
    let (mut a_total, mut n_total) = (0u64, 0u64);
    for case in &clips {
        let (a, n) = best_pair(&case.samples, case.frames(), case.rate(), &budget)?;
        all_exact &= a.object.verify(&case.samples) && n.object.verify(&case.samples);
        a_total += a.complete_bytes;
        n_total += n.complete_bytes;
        common::push_label(&mut projection, &case.clip.id);
        common::push_u64(&mut projection, a.complete_bytes);
        common::push_u64(&mut projection, n.complete_bytes);
        rows.push(pair_json(&case.clip.id, "speech_effectiveness", &a, &n));
    }

    let synth = drifting_ar2(4096);
    let (a, n) = best_pair(&synth, 4096, 48_000, &budget)?;
    all_exact &= a.object.verify(&synth) && n.object.verify(&synth);
    common::push_label(&mut projection, "synthetic-drifting-ar2");
    common::push_u64(&mut projection, a.complete_bytes);
    common::push_u64(&mut projection, n.complete_bytes);
    let synth_row = pair_json("synthetic-drifting-ar2", "synthetic", &a, &n);

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish_exp2(
        "learned-ngsa",
        receipts_root,
        LEARNED_NGSA_SHA256,
        &projection,
        verdict,
        format!(
            "natural-gradient (NNGSA, Seal A0) backward adaptation against the existing sign-sign \
             adaptive family over {} effectiveness clips and a synthetic drifting AR(2): \
             adaptive {a_total} B vs ngsa {n_total} B on speech",
            clips.len()
        ),
        vec![
            (
                "totals",
                serde_json::json!({
                    "speech_adaptive_bytes": a_total,
                    "speech_ngsa_bytes": n_total,
                    "speech_byte_gain": a_total as i64 - n_total as i64,
                }),
            ),
            ("synthetic", synth_row),
            ("clips", serde_json::json!(rows)),
            (
                "method",
                serde_json::json!({
                    "adaptive": "frozen integer sign-sign LMS (model kind 10)",
                    "ngsa": "clean-room fixed-point natural-gradient update preconditioned by an \
                             O(p) AR(1) inverse of the input autocorrelation (model kind 18)",
                    "initialisation": "the same ridge fit feeds both families",
                    "tap_ladder": TAP_LADDER,
                    "selection": "each family keeps its cheapest complete artifact per window",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "this is an independent predictor-side branch; its result is not mixed into the \
                     residual-entropy (E) ladder",
                    "the natural gradient is preconditioned by an AR(1) inverse approximation, not \
                     a full matrix inverse",
                    "per-window fitting is representation selection, never a generalization claim",
                ]),
            ),
        ],
    )
}

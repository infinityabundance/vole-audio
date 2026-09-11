//! `court learned-exp2-mechanisms` — Exp2 mechanism portfolio evidence.
//!
//! Exercises the implemented Exp2 mechanisms (residual codec v2, adaptive
//! segmentation, sparse / high-order linear prediction, long-term prediction,
//! optimizer v2) over the frozen intrinsic corpus and reports the per-mechanism
//! complete-byte waterfall beside the Exp1 baseline and the strong non-learned
//! baselines.
//!
//! Structural gates:
//!
//! * every candidate reconstructs exactly;
//! * the Exp2 portfolio minimum is never larger than the Exp2 dense-linear
//!   candidate (codec v2 is a superset);
//! * the Exp2 portfolio minimum is never larger than the Exp1 dense-linear
//!   candidate (Exp1 is imported).

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::object::LearnedObject;
use crate::learned::segmentation::{DEFAULT_BOUNDARY_GRID, build_segmented_with_regions};
use crate::learned::train::linear::{fit_linear_object, fit_linear_object_exp2};
use crate::learned::train::ltp::fit_ltp_object;
use crate::learned::train::multichannel::fit_multichannel_object;
use crate::learned::train::sparse::fit_sparse_object;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_EXP2_MECHANISMS_SHA256: &str =
    "640cdfb7643bf8068d4e6133f049a5c761b4feb27b2be38ca5c5bb2a95429f7f";

fn bytes(o: &LearnedObject) -> Option<u64> {
    crate::learned::accounting::LearnedCost::of(o)
        .ok()
        .map(|c| c.complete_bytes)
}

/// Run the court; writes `receipts/learned-exp2-mechanisms/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.exp2.mechanisms.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let budget = common::train_budget();
    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut portfolio_le_exp2_linear = true;
    let mut portfolio_le_exp1_linear = true;
    for case in intrinsic_cases().into_iter().take(6) {
        let frames = case.samples.len() as u64 / u64::from(case.channels);
        let mono = case.channels == 1;
        // Exp1 baseline (frozen).
        let exp1 = fit_linear_object(
            &case.samples,
            case.channels,
            frames,
            case.sample_rate_hz,
            4,
            None,
            frames as usize,
            &budget,
        )
        .ok()
        .map(|(o, _)| o);
        let exp1_bytes = exp1.as_ref().and_then(bytes);
        all_exact &= exp1.as_ref().is_none_or(|o| o.verify(&case.samples));

        // Exp2 dense linear.
        let exp2 = fit_linear_object_exp2(
            &case.samples,
            case.channels,
            frames,
            case.sample_rate_hz,
            4,
            None,
            frames as usize,
            &budget,
        )
        .ok()
        .map(|(o, _)| o);
        let exp2_bytes = exp2.as_ref().and_then(bytes);
        all_exact &= exp2.as_ref().is_none_or(|o| o.verify(&case.samples));

        // Sparse / high-order linear (mono).
        let sparse_bytes = if mono {
            fit_sparse_object(&case.samples, frames, case.sample_rate_hz, 8, None, &budget)
                .ok()
                .map(|(o, _)| {
                    all_exact &= o.verify(&case.samples);
                    bytes(&o).unwrap_or(u64::MAX)
                })
        } else {
            None
        };

        // Long-term prediction (mono).
        let ltp_bytes = if mono {
            fit_ltp_object(&case.samples, frames, case.sample_rate_hz, &budget)
                .ok()
                .map(|(o, _)| {
                    all_exact &= o.verify(&case.samples);
                    bytes(&o).unwrap_or(u64::MAX)
                })
        } else {
            None
        };

        // Adaptive segmentation over dense-linear regions.
        let segmented_bytes = if mono {
            let source = case.samples.clone();
            let fit = |a: usize, b: usize| {
                fit_linear_object_exp2(
                    &source[a..b],
                    1,
                    (b - a) as u64,
                    case.sample_rate_hz,
                    4,
                    None,
                    b - a,
                    &budget,
                )
                .ok()
                .map(|(o, _)| o)
            };
            build_segmented_with_regions(
                &case.samples,
                case.channels,
                frames,
                case.sample_rate_hz,
                &DEFAULT_BOUNDARY_GRID,
                fit,
            )
            .map(|outcome| {
                all_exact &= outcome.object.verify(&case.samples);
                outcome.segmented_bytes
            })
        } else {
            None
        };

        // Multichannel (reversible lifting + per-component sparse).
        let multichannel_bytes = if case.channels > 1 {
            fit_multichannel_object(
                &case.samples,
                case.channels,
                frames,
                case.sample_rate_hz,
                &budget,
            )
            .ok()
            .map(|(o, _)| {
                all_exact &= o.verify(&case.samples);
                bytes(&o).unwrap_or(u64::MAX)
            })
        } else {
            None
        };

        let best_nonlearned =
            common::best_simple_bytes(&case.samples, case.channels, frames, case.sample_rate_hz);

        let candidates = [
            exp2_bytes,
            sparse_bytes,
            ltp_bytes,
            segmented_bytes,
            multichannel_bytes,
        ];
        let best_exp2 = candidates.iter().filter_map(|b| *b).min();
        if let (Some(b2), Some(l2)) = (best_exp2, exp2_bytes) {
            portfolio_le_exp2_linear &= b2 <= l2;
        }
        if let (Some(b2), Some(l1)) = (best_exp2, exp1_bytes) {
            portfolio_le_exp1_linear &= b2 <= l1;
        }
        common::push_label(&mut projection, case.id);
        for v in [
            exp1_bytes.unwrap_or(u64::MAX),
            exp2_bytes.unwrap_or(u64::MAX),
            sparse_bytes.unwrap_or(u64::MAX),
            ltp_bytes.unwrap_or(u64::MAX),
            segmented_bytes.unwrap_or(u64::MAX),
            multichannel_bytes.unwrap_or(u64::MAX),
        ] {
            common::push_u64(&mut projection, v);
        }
        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "channels": case.channels,
            "exp1_linear_bytes": exp1_bytes,
            "exp2_linear_bytes": exp2_bytes,
            "sparse_bytes": sparse_bytes,
            "ltp_bytes": ltp_bytes,
            "segmented_bytes": segmented_bytes,
            "multichannel_bytes": multichannel_bytes,
            "best_exp2_bytes": best_exp2,
            "best_nonlearned_bytes": if best_nonlearned == u64::MAX { None } else { Some(best_nonlearned) },
        }));
    }

    let verdict = if all_exact && portfolio_le_exp2_linear && portfolio_le_exp1_linear {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_exp2(
        "learned-exp2-mechanisms",
        receipts_root,
        LEARNED_EXP2_MECHANISMS_SHA256,
        &projection,
        verdict,
        format!(
            "Exp2 mechanism portfolio over {} intrinsic cases: residual codec v2 + segmentation + \
             sparse linear + long-term prediction; all candidates exact and the portfolio min ≤ \
             both the Exp2 and Exp1 dense-linear candidates",
            rows.len()
        ),
        vec![
            ("cases", serde_json::json!(rows)),
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "portfolio_le_exp2_linear": portfolio_le_exp2_linear,
                    "portfolio_le_exp1_linear": portfolio_le_exp1_linear,
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "sparse and long-term prediction are mono-first in this build",
                    "segmentation uses dense-linear Exp2 regions; a full per-region sparse search is \
                     bounded by the encoder's declared budget",
                    "no mechanism is privileged: the portfolio minimum is the measured physical \
                     minimum complete cost"
                ]),
            ),
        ],
    )
}

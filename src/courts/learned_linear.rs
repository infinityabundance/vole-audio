//! `court learned-linear` — the first proof experiment (`O.48`, `O.5`, `O.63`).
//!
//! The smallest falsifiable object: one bounded mono integer window, one learned
//! linear finite-field predictor, one exact residual, one canonical encoding,
//! one scalar evaluator, compared honestly against the literal floor, the
//! existing VOLE inverse compiler, FLAC-5 and simple exact predictors.
//!
//! The learned predictor is permitted to lose; the court records both
//! outcomes. No broader Phase O claim is made until this court exists.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_LINEAR_SHA256: &str =
    "db4aed4478a0721ea919feb33f3850a443711184156df51ba657cbc8ff40cdaa";

/// Tap counts swept by the minimum linear proof.
pub const LINEAR_TAPS: [u16; 6] = [1, 2, 4, 8, 16, 32];

/// Run the court; writes `receipts/learned-linear/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.linear.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let mut rows = Vec::new();
    let mut buckets = [0u64; 3]; // learned cheaper / equal / larger than best non-learned
    let mut all_exact = true;
    let mut total_learned = 0u64;
    let mut total_baseline = 0u64;

    for case in intrinsic_cases() {
        let ch = case.channels;
        let frames = case.samples.len() as u64 / u64::from(ch);

        let literal = common::literal_floor(&case.samples, ch);
        let (u1_bytes, u1_kind) = common::u1_best(case.id, &case.samples, ch)?;
        let flac = common::flac5(&case.samples, ch, case.sample_rate_hz);
        let simple_bytes =
            common::best_simple_bytes(&case.samples, ch, frames, case.sample_rate_hz);
        let (learned, train) = common::best_learned_linear(
            &case.samples,
            ch,
            frames,
            case.sample_rate_hz,
            &LINEAR_TAPS,
            None,
        )?;

        let mut baseline_candidates = vec![u1_bytes, simple_bytes];
        if let Some(f) = flac {
            baseline_candidates.push(f);
        }
        let baseline = *baseline_candidates.iter().min().unwrap();
        let baseline_kind = if baseline == u1_bytes {
            format!("u1:{u1_kind}")
        } else if Some(baseline) == flac {
            "flac-5".to_string()
        } else {
            "simple".to_string()
        };

        let learned_bytes = match &learned {
            Some((o, b, k)) => {
                all_exact &= o.verify(&case.samples);
                common::push_u64(&mut projection, *b);
                common::push_u64(&mut projection, u64::from(*k));
                total_learned = total_learned.saturating_add(*b);
                Some((*b, *k))
            }
            None => {
                common::push_u64(&mut projection, u64::MAX);
                common::push_u64(&mut projection, 0);
                None
            }
        };
        total_baseline = total_baseline.saturating_add(baseline);

        match learned_bytes {
            Some((b, _)) if b < baseline => buckets[0] += 1,
            Some((b, _)) if b == baseline => buckets[1] += 1,
            _ => buckets[2] += 1,
        }
        common::push_u64(&mut projection, literal);
        common::push_u64(&mut projection, u1_bytes);
        common::push_u64(&mut projection, flac.unwrap_or(0));
        common::push_u64(&mut projection, simple_bytes);

        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "group": case.group,
            "channels": ch,
            "frames": frames,
            "literal_floor": literal,
            "u1_best_bytes": u1_bytes,
            "u1_best_kind": u1_kind,
            "flac5_bytes": flac,
            "simple_bytes": simple_bytes,
            "simple_kind": "simple_predictor",
            "learned_bytes": learned_bytes.map(|(b, _)| b),
            "learned_taps": learned_bytes.map(|(_, k)| k),
            "best_non_learned": baseline,
            "best_non_learned_kind": baseline_kind,
            "learned_over_best": learned_bytes.map(|(b, _)| b as f64 / baseline as f64),
            "training_candidates": train.candidates,
            "training_iterations": train.iterations,
        }));
    }

    common::push_u64(&mut projection, buckets[0]);
    common::push_u64(&mut projection, buckets[1]);
    common::push_u64(&mut projection, buckets[2]);
    common::push_u64(&mut projection, total_learned);
    common::push_u64(&mut projection, total_baseline);

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-linear",
        receipts_root,
        LEARNED_LINEAR_SHA256,
        &projection,
        verdict,
        format!(
            "learned linear finite-field over {} corpus windows vs literal/u1/FLAC-5/simple \
             predictors: {} cheaper, {} equal, {} larger (aggregate {total_learned} vs \
             {total_baseline} bytes)",
            rows.len(),
            buckets[0],
            buckets[1],
            buckets[2]
        ),
        vec![
            (
                "comparison",
                serde_json::json!({
                    "taps_swept": LINEAR_TAPS,
                    "learned_cheaper": buckets[0],
                    "learned_equal": buckets[1],
                    "learned_larger": buckets[2],
                    "aggregate_learned_bytes": total_learned,
                    "aggregate_best_non_learned_bytes": total_baseline,
                }),
            ),
            ("objects", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "the canonical container stores one exact residual per object; a candidate \
                     whose exact residual does not fit the i32 residual domain is rejected",
                    "the learned family is deliberately linear here: more complex families are \
                     gated on this court producing an evidence record first",
                    "MSE is not the objective: complete encoded bytes decide"
                ]),
            ),
        ],
    )
}

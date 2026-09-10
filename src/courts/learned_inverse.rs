//! `court learned-inverse` — inverse-compiler integration (`O.15`, `O.42`).
//!
//! Learned hypotheses enter the bounded candidate search as one more family.
//! Selection is by measured evidence only: an exact learned candidate is
//! admitted when it is a **Pareto improvement** (complete bytes at or below the
//! existing VOLE best, never worse on decode work), and rejected otherwise with
//! a typed reason. The frozen `u1/v1` taxonomy is untouched: admission happens
//! only inside the experimental learned profile.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_INVERSE_SHA256: &str =
    "6ce95cb02cf3811024f73c9f36de1a51649a6fc5855756a5e2fa92a3d5e7780d";

/// Negative-result vocabulary for learned admission (`O.44`).
pub const REJECT_MODEL_LARGER_THAN_LITERAL: &str = "MODEL_LARGER_THAN_LITERAL";
pub const REJECT_MODEL_PLUS_RESIDUAL_LARGER_THAN_BASELINE: &str =
    "MODEL_PLUS_RESIDUAL_LARGER_THAN_BASELINE";
pub const REJECT_RESIDUAL_NOT_ECONOMIC: &str = "RESIDUAL_NOT_ECONOMIC";

/// Run the court; writes `receipts/learned-inverse/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.inverse.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let mut rows = Vec::new();
    let mut admitted = 0u64;
    let mut rejected = 0u64;
    let mut all_exact = true;

    for case in intrinsic_cases() {
        let ch = case.channels;
        let frames = case.samples.len() as u64 / u64::from(ch);
        let (u1_bytes, u1_kind) = common::u1_best(case.id, &case.samples, ch)?;
        let literal = common::literal_floor(&case.samples, ch);

        let learned = common::best_learned_linear(
            &case.samples,
            ch,
            frames,
            case.sample_rate_hz,
            &[1, 2, 4, 8, 16, 32],
            None,
        )?
        .0;

        let (decision, reason, learned_bytes) = match learned {
            Some((o, bytes, _taps)) => {
                let exact = o.verify(&case.samples);
                all_exact &= exact;
                if !exact {
                    ("REJECTED", "CLOSURE_NOT_EXACT", Some(bytes))
                } else if bytes > literal {
                    ("REJECTED", REJECT_MODEL_LARGER_THAN_LITERAL, Some(bytes))
                } else if bytes > u1_bytes {
                    (
                        "REJECTED",
                        REJECT_MODEL_PLUS_RESIDUAL_LARGER_THAN_BASELINE,
                        Some(bytes),
                    )
                } else {
                    ("ADMITTED", "PARETO_IMPROVEMENT", Some(bytes))
                }
            }
            None => ("REJECTED", REJECT_RESIDUAL_NOT_ECONOMIC, None),
        };
        if decision == "ADMITTED" {
            admitted += 1;
        } else {
            rejected += 1;
        }

        common::push_label(&mut projection, case.id);
        common::push_label(&mut projection, decision);
        common::push_u64(&mut projection, learned_bytes.unwrap_or(u64::MAX));
        common::push_u64(&mut projection, u1_bytes);
        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "u1_best_bytes": u1_bytes,
            "u1_best_kind": u1_kind,
            "literal_floor": literal,
            "learned_bytes": learned_bytes,
            "decision": decision,
            "reason": reason,
        }));
    }

    common::push_u64(&mut projection, admitted);
    common::push_u64(&mut projection, rejected);

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-inverse",
        receipts_root,
        LEARNED_INVERSE_SHA256,
        &projection,
        verdict,
        format!(
            "learned candidates admitted by measured Pareto improvement over the existing VOLE \
             best: {admitted} admitted, {rejected} rejected (typed reasons), over {} windows",
            rows.len()
        ),
        vec![
            (
                "policy",
                serde_json::json!({
                    "admission": "exact closure AND complete_bytes <= existing best (Pareto, no weighted score)",
                    "profile": crate::learned::profile::LEARNED_PROFILE,
                    "u1_v1_unchanged": true,
                    "reasons": [
                        REJECT_MODEL_LARGER_THAN_LITERAL,
                        REJECT_MODEL_PLUS_RESIDUAL_LARGER_THAN_BASELINE,
                        REJECT_RESIDUAL_NOT_ECONOMIC,
                        "CLOSURE_NOT_EXACT",
                    ],
                }),
            ),
            ("objects", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "admission is a storage-economics decision, not a semantic one: the literal \
                     fallback and existing procedural representations remain first-class",
                    "the existing inverse compiler's candidate vocabulary is unchanged; learned \
                     candidates are an additive experimental family"
                ]),
            ),
        ],
    )
}

//! `court learned-exp2-baseline` — Exp2 baseline import (Seal A).
//!
//! Exp1 is frozen permanently. This court proves the Exp2 profile is a strict
//! *superset*: every Exp1 candidate remains available, the frozen identity is
//! byte-identical, and Exp2’s v2 residual family never produces a larger object
//! than the Exp1 candidate it imports.
//!
//! The structural claim is:
//!
//! ```text
//! exp1_candidates ⊂ exp2_candidates  ⇒  min(exp2) ≤ min(exp1)
//! ```

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::profile::{
    LEARNED_EXP2_PROFILE, LEARNED_EXP2_PROFILE_TAG, LEARNED_EXP2_PROFILE_VERSION, LEARNED_PROFILE,
    LEARNED_PROFILE_TAG, LEARNED_PROFILE_VERSION, LearnedProfile,
};
use crate::learned::residual_codec2::{ResidualCodecV2, encode_best_v1};
use crate::learned::train::linear::{fit_linear_object, fit_linear_object_exp2};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_EXP2_BASELINE_SHA256: &str =
    "3bd61665b214a0bb85e3eee311bf598cbcb7f76e94cc646c057406897682f111";

/// Run the court; writes `receipts/learned-exp2-baseline/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.exp2.baseline.v1");
    common::push_label(
        &mut projection,
        &String::from_utf8_lossy(LEARNED_PROFILE_TAG),
    );
    common::push_label(
        &mut projection,
        &String::from_utf8_lossy(LEARNED_EXP2_PROFILE_TAG),
    );
    common::push_u64(&mut projection, u64::from(LEARNED_PROFILE_VERSION));
    common::push_u64(&mut projection, u64::from(LEARNED_EXP2_PROFILE_VERSION));
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let frozen_identity_ok = LEARNED_PROFILE == "vole.audio.learned.exp1"
        && LEARNED_PROFILE_TAG == b"vole.audio.u1/vole.audio.learned.exp1"
        && LEARNED_PROFILE_VERSION == 1
        && LearnedProfile::from_tag(LEARNED_PROFILE_TAG) == Some(LearnedProfile::Exp1);
    let distinct_profiles = LEARNED_EXP2_PROFILE == "vole.audio.learned.exp2"
        && LEARNED_PROFILE != LEARNED_EXP2_PROFILE;

    // Every Exp1 codec id still resolves inside the v2 family.
    let v1_importable = ResidualCodecV2::ALL_V1
        .iter()
        .all(|c| c.is_v1() && ResidualCodecV2::from_id(c.id()) == Some(*c));
    // Every v2 codec id resolves.
    let all_ids_ok = (0u8..12).all(|id| ResidualCodecV2::from_id(id).is_some_and(|c| c.id() == id));

    let budget = common::train_budget();
    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut imported_all = true;
    for case in intrinsic_cases().into_iter().take(8) {
        let frames = case.samples.len() as u64 / u64::from(case.channels);
        let (exp1, _) = fit_linear_object(
            &case.samples,
            case.channels,
            frames,
            case.sample_rate_hz,
            4,
            None,
            case.samples.len() / usize::from(case.channels),
            &budget,
        )?;
        let (exp2, _) = fit_linear_object_exp2(
            &case.samples,
            case.channels,
            frames,
            case.sample_rate_hz,
            4,
            None,
            case.samples.len() / usize::from(case.channels),
            &budget,
        )?;
        let exp1_bytes = exp1.canonical_bytes();
        let exp2_bytes = exp2.canonical_bytes();
        let exact = exp1.verify(&case.samples) && exp2.verify(&case.samples);
        all_exact &= exact;
        imported_all &= exp2_bytes.len() <= exp1_bytes.len();
        // Re-encode the same residual under both families: v2 ≤ v1 structurally.
        let residual = exp1.residual()?;
        let v1 = encode_best_v1(&residual);
        let v2 = crate::learned::residual_codec2::encode_best_v2(&residual);
        imported_all &= v2.complete_bytes() <= v1.complete_bytes();
        common::push_label(&mut projection, case.id);
        common::push_u64(&mut projection, exp1_bytes.len() as u64);
        common::push_u64(&mut projection, exp2_bytes.len() as u64);
        common::push_u64(&mut projection, v1.complete_bytes());
        common::push_u64(&mut projection, v2.complete_bytes());
        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "exp1_object_bytes": exp1_bytes.len(),
            "exp2_object_bytes": exp2_bytes.len(),
            "exp1_residual_bytes": v1.complete_bytes(),
            "exp2_residual_bytes": v2.complete_bytes(),
            "exp2_selected_codec": v2.codec.name(),
            "exact": exact,
        }));
    }

    let verdict = if frozen_identity_ok
        && distinct_profiles
        && v1_importable
        && all_ids_ok
        && all_exact
        && imported_all
    {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_exp2(
        "learned-exp2-baseline",
        receipts_root,
        LEARNED_EXP2_BASELINE_SHA256,
        &projection,
        verdict,
        format!(
            "Exp2 baseline import: frozen Exp1 identity verified, {} intrinsic cases recompiled in \
             both profiles, exp2 bytes ≤ exp1 bytes and v2 residual ≤ v1 residual on every case",
            rows.len()
        ),
        vec![
            (
                "identity",
                serde_json::json!({
                    "exp1_profile": LEARNED_PROFILE,
                    "exp1_tag": String::from_utf8_lossy(LEARNED_PROFILE_TAG),
                    "exp2_profile": LEARNED_EXP2_PROFILE,
                    "exp2_tag": String::from_utf8_lossy(LEARNED_EXP2_PROFILE_TAG),
                    "frozen_identity_ok": frozen_identity_ok,
                    "distinct_profiles": distinct_profiles,
                    "v1_importable": v1_importable,
                    "all_codec_ids_resolve": all_ids_ok,
                }),
            ),
            ("cases", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "Seal A changes no algorithm: it only introduces the Exp2 profile namespace and \
                     proves the Exp1 baseline still reproduces exactly",
                    "the Exp2 profile tag is stored in the container in place of the Exp1 tag; the \
                     container layout is otherwise identical"
                ]),
            ),
        ],
    )
}

//! `court learned-u1-wasted` — U1-domain common-factor (wasted-bits) fix.
//!
//! Seal S0's `learned-real-corpus-u1` showed that under the frozen U1 s16 ingest
//! (`i32 = i16 << 16`) the learned portfolio loses roughly 3× to FLAC, because
//! FLAC strips the 16 low zero bits through its wasted-bits mechanism while the
//! learned model does not. This court measures the fix: the `Wasted` model
//! wrapper (kind 16) models the quotients `X >> 16` and the `FactorShift`
//! residual codec strips the common factor from the exact residual.
//!
//! The court compares the S0-style baseline portfolio against the wasted-wrapped
//! LPC portfolio over the U1-mapped effectiveness corpus, paired against FLAC-5.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{
    SampleDomain, available, decoder_available, load_cases_domain, u1_effectiveness_clips,
    u1_real_corpus_sha256,
};
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::train::TrainBudget;
use crate::learned::train::hierarchy::fit_hierarchy_object;
use crate::learned::train::linear::fit_linear_object_exp2;
use crate::learned::train::lpc::fit_lpc_object;
use crate::learned::train::sparse::fit_sparse_object;
use crate::learned::wasted::WastedPredictor;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_U1_WASTED_SHA256: &str =
    "8f58639a8648990a5d98575e693e9ca34c6dcfd823f3a9b0ab660088a0c04643";

const CLIPS: usize = 8;
/// The U1 s16 ingest shift.
const SHIFT: u8 = 16;

fn bytes(o: &LearnedObject) -> Option<u64> {
    LearnedCost::of(o).ok().map(|c| c.complete_bytes)
}

/// Fit LPC on the quotients `source >> SHIFT` and wrap it in the wasted model.
fn fit_wasted_lpc(
    source: &[i32],
    frames: u64,
    rate: u32,
    budget: &TrainBudget,
) -> Result<LearnedObject> {
    let q: Vec<i32> = source.iter().map(|&v| v >> SHIFT).collect();
    let (inner, _) = fit_lpc_object(&q, frames, rate, 4096, 8, budget)?;
    let model = LearnedModel::Wasted(WastedPredictor {
        channels: 1,
        shift: SHIFT,
        inner: Box::new(inner.model),
    });
    LearnedObject::from_intrinsic_exp3(model, 1, frames, rate, Vec::new(), source)
}

/// Run the court; writes `receipts/learned-u1-wasted/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.u1-wasted.v1");
    common::push_label(&mut projection, &u1_real_corpus_sha256());

    if !available() || !decoder_available() {
        return common::finish_exp3(
            "learned-u1-wasted",
            receipts_root,
            LEARNED_U1_WASTED_SHA256,
            &projection,
            Verdict::Inconclusive,
            "real corpus audio or the external `flac` decoder is absent".to_string(),
            vec![(
                "limitations",
                serde_json::json!([
                    "requires the uncommitted LibriSpeech audio bulk under corpus/real/ and the \
                     external `flac` decoder used only to read the frozen compressed clips"
                ]),
            )],
        );
    }

    let mut budget = common::train_budget();
    budget.max_iterations = 128;
    let scratch = std::path::PathBuf::from("target/real-corpus/scratch");
    let cases = load_cases_domain(
        &u1_effectiveness_clips(),
        CLIPS,
        &scratch,
        SampleDomain::U1S16,
    )?;

    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut baseline_total = 0u64;
    let mut wasted_total = 0u64;
    let mut flac_total = 0u64;
    let mut wins = 0u64;
    let mut losses = 0u64;
    for case in &cases {
        let frames = case.frames();
        let rate = case.rate();
        // S0-style baseline: the wired three-family Exp2 portfolio.
        let baseline = [
            fit_linear_object_exp2(
                &case.samples,
                1,
                frames,
                rate,
                4,
                None,
                frames as usize,
                &budget,
            )
            .ok()
            .and_then(|(o, _)| bytes(&o)),
            fit_sparse_object(&case.samples, frames, rate, 10, None, &budget)
                .ok()
                .map(|(o, _)| {
                    all_exact &= o.verify(&case.samples);
                    bytes(&o).unwrap_or(u64::MAX)
                }),
            fit_hierarchy_object(&case.samples, frames, rate, 2, &budget)
                .ok()
                .map(|(o, _)| {
                    all_exact &= o.verify(&case.samples);
                    bytes(&o).unwrap_or(u64::MAX)
                }),
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(u64::MAX);
        let wasted = fit_wasted_lpc(&case.samples, frames, rate, &budget)?;
        all_exact &= wasted.verify(&case.samples);
        let wasted_bytes = bytes(&wasted).unwrap_or(u64::MAX);
        let flac = common::flac5(&case.samples, 1, rate).unwrap_or(u64::MAX);
        baseline_total = baseline_total.saturating_add(baseline);
        wasted_total = wasted_total.saturating_add(wasted_bytes);
        flac_total = flac_total.saturating_add(flac);
        if wasted_bytes < baseline {
            wins += 1;
        } else if wasted_bytes > baseline {
            losses += 1;
        }
        common::push_label(&mut projection, &case.clip.id);
        common::push_u64(&mut projection, baseline);
        common::push_u64(&mut projection, wasted_bytes);
        common::push_u64(&mut projection, flac);
        rows.push(serde_json::json!({
            "id": case.clip.id,
            "baseline_bytes": if baseline == u64::MAX { None } else { Some(baseline) },
            "wasted_bytes": if wasted_bytes == u64::MAX { None } else { Some(wasted_bytes) },
            "flac_bytes": if flac == u64::MAX { None } else { Some(flac) },
            "wasted_residual_codec": wasted.residual_codec.name(),
            "wasted_model_bytes": LearnedCost::of(&wasted)?.model_bytes,
        }));
    }

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_exp3(
        "learned-u1-wasted",
        receipts_root,
        LEARNED_U1_WASTED_SHA256,
        &projection,
        verdict,
        format!(
            "U1-domain common-factor fix over {} clips: the Wasted wrapper + FactorShift residual \
             codec against the S0 baseline portfolio and FLAC-5",
            cases.len()
        ),
        vec![
            (
                "summary",
                serde_json::json!({
                    "baseline_bytes": baseline_total,
                    "wasted_bytes": wasted_total,
                    "flac_bytes": flac_total,
                    "wins_vs_baseline": wins,
                    "losses_vs_baseline": losses,
                    "baseline_over_wasted_ratio": if wasted_total == 0 { 0.0 } else { baseline_total as f64 / wasted_total as f64 },
                }),
            ),
            ("clips", serde_json::json!(rows)),
            (
                "method",
                serde_json::json!({
                    "domain": "frozen U1 s16 ingest: i32 = (i16 as i32) << 16",
                    "mechanism": "Wasted model wrapper (kind 16) models X >> 16; FactorShift \
                                  residual codec (id 14) strips the common factor from the exact \
                                  residual",
                    "baseline": "the S0 three-family Exp2 U1 portfolio",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the wrapper only closes when the scaled prediction does not saturate i32",
                    "object-specific fitting is representation selection, never a generalization \
                     claim"
                ]),
            ),
        ],
    )
}

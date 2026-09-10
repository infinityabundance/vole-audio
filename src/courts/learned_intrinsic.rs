//! `court learned-intrinsic` — learned intrinsic family comparison (`O.50`).
//!
//! Compares every implemented learned intrinsic family against the existing VOLE
//! hypotheses, simple deterministic predictors, the literal floor and the
//! conventional lossless baseline, on the frozen intrinsic corpus.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::train::finite_field::fit_nonlinear_object;
use crate::learned::train::linear::fit_linear_object;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_INTRINSIC_SHA256: &str =
    "6080d10682b8d9cb44dd1b34f58338a1832ca286feec1f3b5909ad878d7d1dee";

/// Run the court; writes `receipts/learned-intrinsic/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.intrinsic.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let budget = common::train_budget();
    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut family_wins = serde_json::Map::new();
    let mut family_totals = serde_json::Map::new();

    for case in intrinsic_cases() {
        let ch = case.channels;
        let frames = case.samples.len() as u64 / u64::from(ch);
        let (u1_bytes, u1_kind) = common::u1_best(case.id, &case.samples, ch)?;
        let flac = common::flac5(&case.samples, ch, case.sample_rate_hz);
        let simple_bytes =
            common::best_simple_bytes(&case.samples, ch, frames, case.sample_rate_hz);
        let baseline = [u1_bytes, simple_bytes]
            .into_iter()
            .chain(flac)
            .min()
            .unwrap_or(u1_bytes);

        // Family candidates.
        let mut families: Vec<(&'static str, LearnedObject)> = Vec::new();
        if let Some((o, _, _)) = common::best_learned_linear(
            &case.samples,
            ch,
            frames,
            case.sample_rate_hz,
            &[1, 2, 4, 8, 16, 32],
            None,
        )?
        .0
        {
            families.push(("linear_finite_field", o));
        }
        if let Some((o, _, _)) = common::best_learned_linear(
            &case.samples,
            ch,
            frames,
            case.sample_rate_hz,
            &[4, 8],
            Some(256),
        )?
        .0
        {
            families.push(("block_local", o));
        }
        if ch == 1
            && let Ok((fir_obj, _)) = fit_linear_object(
                &case.samples,
                1,
                frames,
                case.sample_rate_hz,
                4,
                None,
                frames as usize,
                &budget,
            )
            && let LearnedModel::Linear(fir) = &fir_obj.model
            && let Ok(p) = common::stateful_from_fir(fir, &case.samples, frames as usize, 256)
            && let Ok(o) = LearnedObject::from_intrinsic(
                LearnedModel::Stateful(p),
                1,
                frames,
                case.sample_rate_hz,
                Vec::new(),
                &case.samples,
            )
        {
            families.push(("stateful", o));
        }
        // Nonlinear, on non-adversarial mono cases.
        if ch == 1
            && !case.class.contains("random")
            && !case.class.contains("adversarial")
            && let Ok((o, _)) = fit_nonlinear_object(
                &case.samples,
                frames,
                case.sample_rate_hz,
                8,
                16,
                &budget,
                0,
            )
        {
            families.push(("nonlinear_finite_field", o));
        }

        let mut per_family = Vec::new();
        for (name, o) in &families {
            let exact = o.verify(&case.samples);
            all_exact &= exact;
            let bytes = common::learned_bytes(o)?;
            let entry = family_totals
                .entry((*name).to_string())
                .or_insert(serde_json::json!(0u64));
            *entry = serde_json::json!(entry.as_u64().unwrap_or(0).saturating_add(bytes));
            if bytes < baseline {
                let w = family_wins
                    .entry((*name).to_string())
                    .or_insert(serde_json::json!(0u64));
                *w = serde_json::json!(w.as_u64().unwrap_or(0) + 1);
            }
            common::push_label(&mut projection, case.id);
            common::push_label(&mut projection, name);
            common::push_u64(&mut projection, bytes);
            common::push_u64(
                &mut projection,
                o.model.replay_frames(frames as usize - 1) as u64,
            );
            common::push_u64(&mut projection, o.model.state_bytes());
            per_family.push(serde_json::json!({
                "family": name,
                "kind": o.model.kind_name(),
                "complete_bytes": bytes,
                "exact": exact,
                "replay_frames": o.model.replay_frames(frames as usize - 1),
                "state_bytes": o.model.state_bytes(),
                "checkpoints": o.model.checkpoint_count(),
                "over_best_non_learned": bytes as f64 / baseline as f64,
            }));
        }

        common::push_u64(&mut projection, u1_bytes);
        common::push_u64(&mut projection, flac.unwrap_or(0));
        common::push_u64(&mut projection, simple_bytes);
        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "group": case.group,
            "u1_best_bytes": u1_bytes,
            "u1_best_kind": u1_kind,
            "flac5_bytes": flac,
            "simple_bytes": simple_bytes,
            "best_non_learned": baseline,
            "families": per_family,
        }));
    }

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-intrinsic",
        receipts_root,
        LEARNED_INTRINSIC_SHA256,
        &projection,
        verdict,
        format!(
            "{} learned intrinsic candidates over {} corpus windows vs u1/FLAC-5/simple: every \
             candidate closes exactly; family wins vs best non-learned: {}",
            family_totals.len(),
            rows.len(),
            serde_json::Value::Object(family_wins.clone())
        ),
        vec![
            (
                "comparison",
                serde_json::json!({
                    "family_wins": family_wins,
                    "family_total_bytes": family_totals,
                }),
            ),
            ("objects", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "nonlinear and stateful families are mono-only in this build",
                    "stateful candidates carry canonical checkpoints; their cost includes checkpoint bytes",
                    "block-local candidates reset history every 256 frames, trading ratio for random access"
                ]),
            ),
        ],
    )
}

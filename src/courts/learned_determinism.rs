//! `court learned-determinism` — canonical determinism and exact closure (`O.46`).
//!
//! For every implemented learned family: repeated evaluation, canonical
//! serialization round trip, content-identity stability, chunked == contiguous,
//! seek == sequential, and block-local / checkpoint replay equality. Any
//! mismatch is a correctness failure.

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
pub const LEARNED_DETERMINISM_SHA256: &str =
    "a0c9f027c35a4dea4222d2150254b458607b90cb4f6f59dd370908d8b089e275";

/// Run the court; writes `receipts/learned-determinism/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.determinism.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let budget = common::train_budget();
    let mut checks: u64 = 0;
    let mut failures: u64 = 0;
    let mut rows = Vec::new();

    let record = |ok: bool, checks: &mut u64, failures: &mut u64| {
        *checks += 1;
        if !ok {
            *failures += 1;
        }
        ok
    };

    for case in intrinsic_cases() {
        let frames = case.samples.len() as u64 / u64::from(case.channels);
        let ch = case.channels;

        // Candidate family set for this case. A candidate whose exact residual
        // does not fit the canonical i32 residual domain is rejected honestly
        // (a negative result), not treated as a court failure.
        let mut candidates: Vec<(&str, LearnedObject)> = Vec::new();
        if let Ok(o) = common::simple_object(
            &case.samples,
            ch,
            frames,
            case.sample_rate_hz,
            common::SimplePredictor::Previous,
        ) {
            candidates.push(("simple_previous", o));
        }
        let tap_set: [u16; 3] = [1, 4, 16];
        let (lin, _) = common::best_learned_linear(
            &case.samples,
            ch,
            frames,
            case.sample_rate_hz,
            &tap_set,
            None,
        )?;
        if let Some((o, _, _)) = lin {
            candidates.push(("linear", o));
        }
        let (bl, _) = common::best_learned_linear(
            &case.samples,
            ch,
            frames,
            case.sample_rate_hz,
            &[8],
            Some(256),
        )?;
        if let Some((o, _, _)) = bl {
            candidates.push(("block_local", o));
        }
        // Stateful realization of the best small FIR (mono only).
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
            candidates.push(("stateful", o));
        }
        // Nonlinear (mono only, and only for cases that are not pure noise).
        if ch == 1
            && !case.class.contains("random")
            && !case.class.contains("adversarial")
            && let Ok((o, _)) = fit_nonlinear_object(
                &case.samples,
                frames,
                case.sample_rate_hz,
                if frames >= 8 { 8 } else { 1 },
                8,
                &budget,
                0,
            )
        {
            candidates.push(("nonlinear", o));
        }

        let mut per_case = Vec::new();
        for (name, o) in &candidates {
            // 1. exact closure.
            let closed = o.materialize()? == case.samples;
            record(closed, &mut checks, &mut failures);
            // 2. repeated evaluation is bit-identical.
            let repeat = o.materialize()? == o.materialize()?;
            record(repeat, &mut checks, &mut failures);
            // 3. canonical bytes are stable and re-parse identically.
            let bytes = o.canonical_bytes();
            let parsed = LearnedObject::parse(&bytes)?;
            let stable = parsed.canonical_bytes() == bytes;
            record(stable, &mut checks, &mut failures);
            record(
                o.content_id() == parsed.content_id(),
                &mut checks,
                &mut failures,
            );
            // 4. chunked == contiguous, on non-trivial boundaries.
            let f = frames as usize;
            let mut chunked_ok = true;
            for (start, len) in [
                (0usize, f.min(1)),
                (f / 3, (f / 3).max(1).min(f - f / 3)),
                (f.saturating_sub(7), (f - f.saturating_sub(7)).min(7)),
            ] {
                if len == 0 || start + len > f {
                    continue;
                }
                let contiguous = o.materialize_range(start, len)?;
                let sequential =
                    &o.materialize()?[start * usize::from(ch)..(start + len) * usize::from(ch)];
                chunked_ok &= contiguous == sequential;
            }
            record(chunked_ok, &mut checks, &mut failures);
            // 5. seek replay is deterministic.
            let seek_again = o.materialize_range(f / 2, (f - f / 2).min(64))?;
            let seek_repeat = o.materialize_range(f / 2, (f - f / 2).min(64))? == seek_again;
            record(seek_repeat, &mut checks, &mut failures);
            // 6. residual decode is an exact identity.
            let residual = o.residual()?;
            record(
                crate::learned::residual_codec::encode_best(&residual).decode(residual.len())?
                    == residual,
                &mut checks,
                &mut failures,
            );

            common::push_label(&mut projection, case.id);
            common::push_label(&mut projection, name);
            common::push_u64(&mut projection, o.model.kind_tag() as u64);
            common::push_u64(&mut projection, common::learned_bytes(o)?);
            common::push_u64(
                &mut projection,
                o.model.replay_frames(f.saturating_sub(1)) as u64,
            );
            per_case.push(serde_json::json!({
                "family": name,
                "kind": o.model.kind_name(),
                "complete_bytes": common::learned_bytes(o)?,
                "replay_frames": o.model.replay_frames(f.saturating_sub(1)),
                "closed_exactly": closed,
            }));
        }
        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "candidates": per_case,
        }));
    }

    common::push_u64(&mut projection, checks);
    common::push_u64(&mut projection, failures);

    let verdict = if failures == 0 {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-determinism",
        receipts_root,
        LEARNED_DETERMINISM_SHA256,
        &projection,
        verdict,
        format!(
            "{checks} determinism/closure checks over the intrinsic corpus with {failures} \
             failures; every candidate closes exactly, serializes canonically, and agrees chunked \
             vs contiguous and seek vs sequential"
        ),
        vec![
            (
                "checks",
                serde_json::json!({"total": checks, "failures": failures}),
            ),
            ("corpus", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "stateful candidates use canonical checkpoints and replay; determinism is \
                     asserted for the checkpointed evaluator",
                    "the family set is the vocabulary implemented in this build"
                ]),
            ),
        ],
    )
}

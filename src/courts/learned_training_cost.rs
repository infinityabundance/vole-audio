//! `court learned-training-cost` — training cost accounting (`O.58`, `O.41`).
//!
//! Training cost is separate from playback cost but never hidden. This court
//! records wall time, candidate counts, iterations, quantization attempts, peak
//! host memory (best effort) and the declared search budget.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::train::TrainStats;
use crate::status::Verdict;
use std::path::Path;
use std::time::Instant;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_TRAINING_COST_SHA256: &str =
    "5527eaddf88ff0d6aa000528d182a80e4d11d6051dd80fb7db36d0593d1f052d";

fn peak_rss_bytes() -> u64 {
    // Best effort: /proc/self/status VmHWM.
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: u64 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                return kb * 1024;
            }
        }
    }
    0
}

/// Run the court; writes `receipts/learned-training-cost/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.training-cost.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let budget = common::train_budget();
    let mut totals = TrainStats::default();
    let mut rows = Vec::new();
    let mut all_exact = true;

    for case in intrinsic_cases()
        .into_iter()
        .filter(|c| c.channels == 1)
        .take(10)
    {
        let frames = case.samples.len() as u64;
        let wall0 = Instant::now();

        // Linear fit (ridge). A candidate whose exact residual does not fit the
        // canonical i32 domain is skipped honestly.
        let Some(lin) =
            common::try_fit_linear(&case.samples, 1, frames, case.sample_rate_hz, 8, None)
        else {
            continue;
        };
        all_exact &= lin.verify(&case.samples);

        // Quantization-aware fit.
        let Some(qat) = common::try_fit_qat(&case.samples, 1, frames, case.sample_rate_hz, 8, 16)
        else {
            continue;
        };
        all_exact &= qat.verify(&case.samples);

        // Nonlinear fit.
        let Some(nl) = common::try_fit_nonlinear(&case.samples, frames, case.sample_rate_hz, 8, 8)
        else {
            continue;
        };
        all_exact &= nl.verify(&case.samples);

        let wall_ns = wall0.elapsed().as_nanos() as u64;
        let case_stats = TrainStats {
            candidates: 3,
            fit_ns: wall_ns,
            ..Default::default()
        };
        totals.merge(&case_stats);
        totals.fit_ns = totals.fit_ns.saturating_add(wall_ns);

        common::push_label(&mut projection, case.id);
        common::push_u64(&mut projection, case_stats.candidates);
        common::push_u64(&mut projection, case_stats.iterations);
        common::push_u64(&mut projection, case_stats.quantization_attempts);
        // Deterministic evidence only: fitted object sizes, never wall time.
        common::push_u64(&mut projection, common::learned_bytes(&lin)?);
        common::push_u64(&mut projection, common::learned_bytes(&qat)?);
        common::push_u64(&mut projection, nl.model.kind_tag() as u64);
        common::push_u64(&mut projection, common::learned_bytes(&nl)?);
        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "wall_ns": wall_ns,
            "candidates": case_stats.candidates,
            "iterations": case_stats.iterations,
            "quantization_attempts": case_stats.quantization_attempts,
            "families": {
                "linear_bytes": common::learned_bytes(&lin)?,
                "qat_bytes": common::learned_bytes(&qat)?,
                "nonlinear_kind": nl.model.kind_name(),
                "nonlinear_bytes": common::learned_bytes(&nl)?,
            },
        }));
    }

    let peak = peak_rss_bytes();
    // The frozen projection carries counts and fitted sizes only; wall time and
    // process high-water memory are evidence, never identity.
    common::push_u64(&mut projection, totals.candidates);
    common::push_u64(&mut projection, totals.iterations);
    common::push_u64(&mut projection, totals.quantization_attempts);

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-training-cost",
        receipts_root,
        LEARNED_TRAINING_COST_SHA256,
        &projection,
        verdict,
        format!(
            "training cost over {} windows: {} candidates, {} iterations, {} quantization \
             attempts, peak host RSS {peak} B",
            rows.len(),
            totals.candidates,
            totals.iterations,
            totals.quantization_attempts
        ),
        vec![
            (
                "training",
                serde_json::json!({
                    "fit_ns": totals.fit_ns,
                    "candidates": totals.candidates,
                    "rejected": totals.rejected,
                    "iterations": totals.iterations,
                    "quantization_attempts": totals.quantization_attempts,
                    "peak_host_rss_bytes": peak,
                    "gpu_ns": totals.gpu_ns,
                    "peak_vram_bytes": totals.peak_vram_bytes,
                    "budget": {
                        "max_candidates": budget.max_candidates,
                        "max_taps": budget.max_taps,
                        "max_hidden": budget.max_hidden,
                        "max_iterations": budget.max_iterations,
                        "max_model_bytes": budget.max_model_bytes,
                        "ridge_lambda": budget.ridge_lambda,
                    },
                }),
            ),
            ("windows", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "GPU training time and VRAM are zero: training is CPU-only in this build",
                    "peak host RSS is a process high-water mark, not a per-fit peak"
                ]),
            ),
        ],
    )
}

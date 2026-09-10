//! `court learned-capacity` — multi-capacity Pareto surface (`O.53`, `O.27`).
//!
//! For the same target, several bounded capacities are fitted and the complete
//! Pareto surface is reported: model bytes, residual bytes, total bytes, decode
//! work and seek cost. The largest or most accurate model is never privileged.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_CAPACITY_SHA256: &str =
    "7879ecc81d283d1fb2bc0a0dd339bbd06ff4a709f00c6a6c9a5ce60c65c825b6";

/// Capacities swept: causal tap counts, i.e. model budget.
pub const CAPACITIES: [u16; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

/// Run the court; writes `receipts/learned-capacity/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.capacity.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut frontier_sizes = Vec::new();

    for case in intrinsic_cases().into_iter().take(10) {
        let ch = case.channels;
        let frames = case.samples.len() as u64 / u64::from(ch);
        let mut points = Vec::new();
        for &k in &CAPACITIES {
            if u64::from(k) >= frames {
                continue;
            }
            let (o, _) = match crate::learned::train::linear::fit_linear_object(
                &case.samples,
                ch,
                frames,
                case.sample_rate_hz,
                k,
                None,
                frames as usize,
                &common::train_budget(),
            ) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if !o.verify(&case.samples) {
                all_exact = false;
                continue;
            }
            let cost = LearnedCost::of(&o)?;
            common::push_label(&mut projection, case.id);
            common::push_u64(&mut projection, u64::from(k));
            common::push_u64(&mut projection, cost.model_bytes);
            common::push_u64(&mut projection, cost.residual_bytes);
            common::push_u64(&mut projection, cost.complete_bytes);
            points.push(serde_json::json!({
                "taps": k,
                "model_bytes": cost.model_bytes,
                "residual_bytes": cost.residual_bytes,
                "complete_bytes": cost.complete_bytes,
                "ops_per_sample": cost.ops_per_sample,
                "replay_frames": cost.worst_case_replay_frames,
            }));
        }
        // Pareto frontier over (complete_bytes, ops_per_sample).
        let mut frontier: Vec<usize> = Vec::new();
        for (i, a) in points.iter().enumerate() {
            let ax = a["complete_bytes"].as_u64().unwrap_or(u64::MAX);
            let ao = a["ops_per_sample"].as_u64().unwrap_or(u64::MAX);
            let dominated = points.iter().enumerate().any(|(j, b)| {
                if i == j {
                    return false;
                }
                let bx = b["complete_bytes"].as_u64().unwrap_or(u64::MAX);
                let bo = b["ops_per_sample"].as_u64().unwrap_or(u64::MAX);
                bx <= ax && bo <= ao && (bx < ax || bo < ao)
            });
            if !dominated {
                frontier.push(i);
            }
        }
        frontier_sizes.push(frontier.len() as u64);
        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "points": points,
            "pareto_frontier": frontier,
        }));
    }

    for s in &frontier_sizes {
        common::push_u64(&mut projection, *s);
    }

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-capacity",
        receipts_root,
        LEARNED_CAPACITY_SHA256,
        &projection,
        verdict,
        format!(
            "multi-capacity Pareto surface over {} targets and {} capacities; frontier sizes \
             {frontier_sizes:?}",
            rows.len(),
            CAPACITIES.len()
        ),
        vec![
            (
                "selection",
                serde_json::json!({
                    "rule": "Pareto dominance over measured complete bytes and decode work",
                    "privilege": "none: the largest capacity is never preferred by construction",
                }),
            ),
            ("targets", serde_json::json!(rows)),
        ],
    )
}

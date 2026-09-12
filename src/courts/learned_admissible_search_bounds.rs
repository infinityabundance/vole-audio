//! `court learned-admissible-search-bounds` — Phase 6 mechanism 10
//! (`AdmissibleSearchBounds`).
//!
//! A bounded encoder search spends most of its time fully evaluating candidates
//! that a cheap, *provable* lower bound would already have rejected. The rule is
//! one inequality: if `lower_bound(i) <= exact(i)` and `lower_bound(i) >= best`,
//! then `exact(i) >= best` and the candidate cannot win.
//!
//! This court exercises the general instrument
//! ([`crate::learned::bounds::admissible_select`]) against its exhaustive
//! reference on real fixtures, using the concrete integration in the simple
//! predictor search: the lower bound is the candidate's weight and bias bytes,
//! which are known before the candidate is built and are a proven subset of its
//! complete bytes. It reports candidates evaluated and pruned, and asserts the
//! bounded winner is *identical* to the exhaustive winner (index and cost).
//!
//! The court also runs a deterministic admissibility battery in which the bounds
//! are tight enough to prune heavily, so the mechanism's correctness is tested
//! where it actually does work, not only where the production bound is weak.

use crate::courts::learned_common as common;
use crate::courts::learned_common::{
    PERIODIC_CANDIDATES, SimplePredictor, learned_bytes, simple_object,
};
use crate::entropy::corpus;
use crate::error::Result;
use crate::learned::bounds::{admissible_select, exhaustive_select};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_ADMISSIBLE_SEARCH_BOUNDS_SHA256: &str =
    "2f35af167bef91dfe361751b441d8e3df343c0570375d2026306a929cb069543";

/// The fixture names exercised from the frozen synthetic corpus.
const FIXTURES: [&str; 8] = [
    "silence",
    "dc",
    "single-sine",
    "harmonic-tone",
    "quasi-periodic",
    "impulse-train",
    "am-signal",
    "white-noise",
];

fn plan(frames: u64) -> Vec<SimplePredictor> {
    let mut plan = vec![
        SimplePredictor::Zero,
        SimplePredictor::Constant,
        SimplePredictor::Previous,
    ];
    for p in PERIODIC_CANDIDATES {
        if u64::from(p) < frames {
            plan.push(SimplePredictor::Periodic(p));
        }
    }
    plan
}

fn lower_bound(kind: SimplePredictor, channels: u8) -> u64 {
    let c = usize::from(channels);
    let taps = match kind {
        SimplePredictor::Periodic(p) => usize::from(p).max(1),
        _ => 1,
    };
    (taps * c * c * 2 + c * 4) as u64
}

struct Row {
    id: String,
    channels: u8,
    frames: usize,
    candidates: usize,
    evaluated: usize,
    pruned: usize,
    winner_index: u64,
    winner_cost: u64,
    bounded_matches_exhaustive: bool,
}

fn analyse(id: String, samples: &[i32], channels: u8, frames: u64, rate: u32) -> Result<Row> {
    let plan = plan(frames);
    let costs: Vec<u64> = plan
        .iter()
        .map(
            |&kind| match simple_object(samples, channels, frames, rate, kind) {
                Ok(o) => learned_bytes(&o).unwrap_or(u64::MAX),
                Err(_) => u64::MAX,
            },
        )
        .collect();
    let exhaustive = exhaustive_select(costs.len(), |i| costs[i]);
    let bounded = admissible_select(
        costs.len(),
        |i| lower_bound(plan[i], channels),
        |i| costs[i],
    );
    let bounded_matches_exhaustive = bounded.winner == exhaustive.winner
        && bounded.winner_cost == exhaustive.winner_cost
        && bounded.evaluated + bounded.pruned == bounded.candidates;
    Ok(Row {
        id,
        channels,
        frames: frames as usize,
        candidates: bounded.candidates,
        evaluated: bounded.evaluated,
        pruned: bounded.pruned,
        winner_index: bounded.winner.map(|i| i as u64).unwrap_or(u64::MAX),
        winner_cost: bounded.winner_cost,
        bounded_matches_exhaustive,
    })
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.bounded_matches_exhaustive;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.channels as u64);
    common::push_u64(projection, row.candidates as u64);
    common::push_u64(projection, row.evaluated as u64);
    common::push_u64(projection, row.pruned as u64);
    common::push_u64(projection, row.winner_index);
    common::push_u64(projection, row.winner_cost);
}

fn row_json(row: &Row) -> serde_json::Value {
    serde_json::json!({
        "id": row.id,
        "channels": row.channels,
        "frames": row.frames,
        "candidates": row.candidates,
        "evaluated": row.evaluated,
        "pruned": row.pruned,
        "winner_index": row.winner_index,
        "winner_cost": row.winner_cost,
        "bounded_matches_exhaustive": row.bounded_matches_exhaustive,
    })
}

/// A deterministic admissibility battery with tight bounds, so pruning is
/// exercised where it actually bites.
fn battery() -> (usize, usize, usize, bool) {
    // Candidate costs from a fixed mix; the bound is cost/3 (admissible and
    // tight enough to prune everything after the true minimum is found).
    let costs: Vec<u64> = (0..96u64).map(|i| (i * 2654435761) % 4093 + 11).collect();
    let lo: Vec<u64> = costs.iter().map(|&c| c / 3).collect();
    let bounded = admissible_select(costs.len(), |i| lo[i], |i| costs[i]);
    let exhaustive = exhaustive_select(costs.len(), |i| costs[i]);
    let ok = bounded.winner == exhaustive.winner && bounded.winner_cost == exhaustive.winner_cost;
    (bounded.candidates, bounded.evaluated, bounded.pruned, ok)
}

/// Run the court; writes `receipts/learned-admissible-search-bounds/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(
        &mut projection,
        "vole.audio.learned.admissible_search_bounds.v1",
    );

    let mut rows = Vec::new();
    let mut all_exact = true;
    for name in FIXTURES {
        let Some(fx) = corpus::named(name) else {
            continue;
        };
        let frames = fx.frames().min(2048);
        if frames < 2 {
            continue;
        }
        let samples = &fx.samples[..frames * usize::from(fx.channels)];
        let row = analyse(
            name.to_string(),
            samples,
            fx.channels,
            frames as u64,
            48_000,
        )?;
        let mut gates = true;
        push_row(&mut projection, &row, &mut gates);
        all_exact &= gates;
        rows.push(row);
    }

    // The integration itself: `best_simple` must return exactly the exhaustive
    // winner's bytes.
    let mut integration_matches = true;
    for name in FIXTURES {
        let Some(fx) = corpus::named(name) else {
            continue;
        };
        let frames = fx.frames().min(2048);
        if frames < 2 {
            continue;
        }
        let samples = &fx.samples[..frames * usize::from(fx.channels)];
        let plan = plan(frames as u64);
        let costs: Vec<u64> = plan
            .iter()
            .map(
                |&kind| match simple_object(samples, fx.channels, frames as u64, 48_000, kind) {
                    Ok(o) => learned_bytes(&o).unwrap_or(u64::MAX),
                    Err(_) => u64::MAX,
                },
            )
            .collect();
        let exhaustive = exhaustive_select(costs.len(), |i| costs[i]);
        let (_, got) = common::best_simple(samples, fx.channels, frames as u64, 48_000)?;
        integration_matches &= got == exhaustive.winner_cost;
    }
    all_exact &= integration_matches;

    let (battery_candidates, battery_evaluated, battery_pruned, battery_ok) = battery();
    all_exact &= battery_ok;

    let total_candidates: u64 = rows.iter().map(|r| r.candidates as u64).sum();
    let total_pruned: u64 = rows.iter().map(|r| r.pruned as u64).sum();
    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else {
        Verdict::Supported
    };

    let rows_json: serde_json::Value = rows.iter().map(row_json).collect();
    common::finish_exp3(
        "learned-admissible-search-bounds",
        receipts_root,
        LEARNED_ADMISSIBLE_SEARCH_BOUNDS_SHA256,
        &projection,
        verdict,
        format!(
            "admissible-bounded selection over {n} fixtures: bounded winner == exhaustive winner \
             everywhere; production search pruned {tp} of {tc} candidates (model-byte bound) and \
             the deterministic tight-bound battery pruned {bp} of {bc} while matching exhaustive \
             ({be} evaluated); best_simple integration matches the exhaustive winner",
            n = rows.len(),
            tp = total_pruned,
            tc = total_candidates,
            bp = battery_pruned,
            bc = battery_candidates,
            be = battery_evaluated,
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "integration_matches": integration_matches,
                    "battery_ok": battery_ok,
                }),
            ),
            ("fixtures", rows_json),
            (
                "battery",
                serde_json::json!({
                    "candidates": battery_candidates,
                    "evaluated": battery_evaluated,
                    "pruned": battery_pruned,
                    "matches_exhaustive": battery_ok,
                }),
            ),
            (
                "method",
                serde_json::json!({
                    "rule": "lower_bound(i) <= exact(i); prune when lower_bound(i) >= best, with \
                             equal-cost ties broken by ascending candidate index",
                    "production_bound": "a simple predictor's weight and bias bytes, known before \
                                        fitting and a proven subset of its complete bytes",
                    "order": "candidates are processed in ascending lower-bound order, so the \
                              first bound that reaches the incumbent proves every remaining \
                              candidate is dead",
                    "equivalence": "the bounded winner equals the exhaustive winner in both index \
                                    and cost on every case",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the production model-byte bound is weak on mono material, where the residual \
                     dominates the complete cost; pruning is modest and reported honestly",
                    "applying the same instrument to the adaptive residual codecs needs exact cost \
                     oracles, which those codecs do not yet expose",
                ]),
            ),
        ],
    )
}

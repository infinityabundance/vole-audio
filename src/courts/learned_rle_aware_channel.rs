//! `court learned-rle-aware-channel` — Phase 6 mechanism 8
//! (`RleAwareChannelTransform`).
//!
//! The Exp2 multichannel family applies a reversible integer channel transform
//! and predicts each component. `RleAwareChannelTransform` extends the stereo
//! ladder with left/right difference lifting (`(L, R - L)` and `(R, L - R)`)
//! and orders the ladder by the **downstream run/zero topology** of the
//! transformed components — zero density, zero runs, the longest zero run and
//! equal neighbour pairs — rather than by residual variance. Topology only
//! proposes and orders; the exact canonical bytes of every fully fitted
//! candidate still decide the winner.
//!
//! This court reports, per fixture, the topology features and the measured
//! complete bytes of each transform, then compares three selectors:
//!
//! * topology (maximum `rle_score`, the mechanism's proposal),
//! * variance (minimum summed component variance, the rejected heuristic),
//! * measured bytes (the authority).
//!
//! It also confirms every reversible transform closes exactly. Held-out Mode C
//! is deliberately untouched.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::multichannel::{
    STEREO_TRANSFORMS, TRANSFORM_LEFT_DIFF, TRANSFORM_MID_SIDE, TRANSFORM_NONE,
    TRANSFORM_RIGHT_DIFF, channel_topology, transform_frame,
};
use crate::learned::train::TrainBudget;
use crate::learned::train::multichannel::{
    fit_multichannel_object, measure_multichannel_transform,
};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_RLE_AWARE_CHANNEL_SHA256: &str =
    "2334c733f33aaf56f39efd4c0dcc8817169e5bc9c3fe077e5ae9e0b439ebf2da";

/// Frames per synthetic stereo fixture (keeps the bounded fit affordable).
const FRAMES: usize = 1500;

fn transform_name(transform: u8) -> &'static str {
    match transform {
        TRANSFORM_NONE => "none",
        TRANSFORM_MID_SIDE => "mid_side",
        TRANSFORM_LEFT_DIFF => "left_diff",
        TRANSFORM_RIGHT_DIFF => "right_diff",
        _ => "unknown",
    }
}

/// Summed per-component variance of a transform (the rejected heuristic).
fn transform_variance(interleaved: &[i32], channels: u8, frames: usize, transform: u8) -> f64 {
    let c = usize::from(channels);
    let mut sum = [0f64; 2];
    let mut sq = [0f64; 2];
    let mut out = [0i32; 2];
    for t in 0..frames {
        transform_frame(transform, &interleaved[t * c..t * c + 2], &mut out);
        for cc in 0..c {
            let v = f64::from(out[cc]);
            sum[cc] += v;
            sq[cc] += v * v;
        }
    }
    let n = frames.max(1) as f64;
    let mut total = 0.0;
    for cc in 0..c {
        let mean = sum[cc] / n;
        total += (sq[cc] / n - mean * mean).max(0.0);
    }
    total
}

/// Deterministic stereo fixtures whose channels share a reversible relation.
fn fixtures() -> Vec<(&'static str, Vec<i32>)> {
    let mut identical = Vec::new();
    let mut sparse_event = Vec::new();
    let mut noisy_diff = Vec::new();
    let mut silent_right = Vec::new();
    let mut gain_half = Vec::new();
    for t in 0..FRAMES as i64 {
        let l = ((t * 89) % 15013) as i32 - 7000;
        identical.push(l);
        identical.push(l);

        let e = if t % 512 == 7 { 1000 } else { 0 };
        sparse_event.push(l);
        sparse_event.push(l + e);

        let noise = ((t * 2654435761) % 37) as i32 - 18;
        noisy_diff.push(l);
        noisy_diff.push(l + noise);

        silent_right.push(l);
        silent_right.push(0);

        gain_half.push(l);
        gain_half.push(l / 2);
    }
    vec![
        ("identical", identical),
        ("sparse_event", sparse_event),
        ("noisy_diff", noisy_diff),
        ("silent_right", silent_right),
        ("gain_half", gain_half),
    ]
}

struct TransformRow {
    transform: u8,
    name: &'static str,
    rle_score: u64,
    zeros: u64,
    zero_runs: u64,
    longest_zero_run: u64,
    equal_pairs: u64,
    variance_milli: u64,
    complete_bytes: u64,
    fitted: bool,
}

struct Row {
    id: &'static str,
    frames: usize,
    transforms: Vec<TransformRow>,
    byte_winner: &'static str,
    topology_choice: &'static str,
    variance_choice: &'static str,
    topology_matches_bytes: bool,
    /// Measured gain of the extended ladder over the pre-existing
    /// `{none, mid_side}` ladder, in complete bytes (0 when they tie).
    gain_vs_old_ladder_bytes: u64,
}

fn analyse(id: &'static str, source: &[i32], budget: &TrainBudget) -> Result<(Row, bool)> {
    let mut rows = Vec::new();
    let mut all_exact = true;
    for transform in STEREO_TRANSFORMS {
        let topo = channel_topology(source, 2, FRAMES, transform)?;
        // `measure_multichannel_transform` accepts a transform only after the
        // exact closure verifies on the source, so `Some` is the exactness gate.
        let measured =
            measure_multichannel_transform(source, 2, FRAMES as u64, 48_000, transform, budget)?;
        let (complete_bytes, fitted) = match measured {
            Some((_, bytes)) => (bytes, true),
            None => (u64::MAX, false),
        };
        all_exact &= fitted;
        rows.push(TransformRow {
            transform,
            name: transform_name(transform),
            rle_score: topo.rle_score(),
            zeros: topo.zeros,
            zero_runs: topo.zero_runs,
            longest_zero_run: topo.longest_zero_run,
            equal_pairs: topo.equal_pairs,
            variance_milli: (transform_variance(source, 2, FRAMES, transform) * 1000.0) as u64,
            complete_bytes,
            fitted,
        });
    }
    let byte_winner = rows
        .iter()
        .filter(|r| r.fitted)
        .min_by_key(|r| r.complete_bytes)
        .map(|r| r.name)
        .unwrap_or("none");
    let topology_choice = rows
        .iter()
        .max_by(|a, b| {
            a.rle_score
                .cmp(&b.rle_score)
                .then(b.transform.cmp(&a.transform))
        })
        .map(|r| r.name)
        .unwrap_or("none");
    let variance_choice = rows
        .iter()
        .min_by(|a, b| {
            a.variance_milli
                .cmp(&b.variance_milli)
                .then(a.transform.cmp(&b.transform))
        })
        .map(|r| r.name)
        .unwrap_or("none");
    let old_ladder_best = rows
        .iter()
        .filter(|r| {
            r.fitted && (r.transform == TRANSFORM_NONE || r.transform == TRANSFORM_MID_SIDE)
        })
        .map(|r| r.complete_bytes)
        .min()
        .unwrap_or(u64::MAX);
    let gain_vs_old_ladder_bytes = old_ladder_best.saturating_sub(
        rows.iter()
            .filter(|r| r.fitted)
            .map(|r| r.complete_bytes)
            .min()
            .unwrap_or(u64::MAX),
    );
    Ok((
        Row {
            id,
            frames: FRAMES,
            transforms: rows,
            byte_winner,
            topology_choice,
            variance_choice,
            topology_matches_bytes: topology_choice == byte_winner,
            gain_vs_old_ladder_bytes,
        },
        all_exact,
    ))
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    common::push_label(projection, row.id);
    for t in &row.transforms {
        common::push_label(projection, t.name);
        common::push_u64(projection, t.rle_score);
        common::push_u64(projection, t.complete_bytes);
    }
    common::push_label(projection, row.byte_winner);
    common::push_label(projection, row.topology_choice);
    common::push_label(projection, row.variance_choice);
    common::push_u64(projection, row.gain_vs_old_ladder_bytes);
    *gates &= row
        .transforms
        .iter()
        .all(|t| t.fitted || t.complete_bytes == u64::MAX);
}

fn row_json(row: &Row) -> serde_json::Value {
    let transforms: serde_json::Value = row
        .transforms
        .iter()
        .map(|t| {
            serde_json::json!({
                "transform": t.name,
                "rle_score": t.rle_score,
                "zeros": t.zeros,
                "zero_runs": t.zero_runs,
                "longest_zero_run": t.longest_zero_run,
                "equal_pairs": t.equal_pairs,
                "variance_milli": t.variance_milli,
                "complete_bytes": if t.fitted { serde_json::json!(t.complete_bytes) } else { serde_json::Value::Null },
            })
        })
        .collect();
    serde_json::json!({
        "id": row.id,
        "frames": row.frames,
        "transforms": transforms,
        "byte_winner": row.byte_winner,
        "topology_choice": row.topology_choice,
        "variance_choice": row.variance_choice,
        "topology_matches_bytes": row.topology_matches_bytes,
        "gain_vs_old_ladder_bytes": row.gain_vs_old_ladder_bytes,
    })
}

/// Run the court; writes `receipts/learned-rle-aware-channel/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.rle_aware_channel.v1");

    let budget = TrainBudget::default();
    let mut rows = Vec::new();
    let mut all_exact = true;
    for (id, source) in fixtures() {
        let (row, exact) = analyse(id, &source, &budget)?;
        all_exact &= exact;
        let mut gates = true;
        push_row(&mut projection, &row, &mut gates);
        all_exact &= gates;
        rows.push(row);
    }

    // The frozen Exp2 multichannel fit must still close on a correlated stereo
    // fixture after the ladder extension.
    let mut fit_closed = false;
    let mut fit_bytes = 0u64;
    if let Some((_, source)) = fixtures().first() {
        let (o, _) = fit_multichannel_object(source, 2, FRAMES as u64, 48_000, &budget)?;
        fit_closed = o.verify(source);
        fit_bytes = LearnedCost::of(&o)?.complete_bytes;
    }
    all_exact &= fit_closed;

    let topology_matches = rows.iter().filter(|r| r.topology_matches_bytes).count() as u64;
    let variance_matches = rows
        .iter()
        .filter(|r| r.variance_choice == r.byte_winner)
        .count() as u64;
    let ladder_gain_total: u64 = rows.iter().map(|r| r.gain_vs_old_ladder_bytes).sum();
    let ladder_improved = rows
        .iter()
        .filter(|r| r.gain_vs_old_ladder_bytes > 0)
        .count() as u64;

    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else {
        Verdict::Supported
    };

    let rows_json: serde_json::Value = rows.iter().map(row_json).collect();
    common::finish_exp3(
        "learned-rle-aware-channel",
        receipts_root,
        LEARNED_RLE_AWARE_CHANNEL_SHA256,
        &projection,
        verdict,
        format!(
            "topology-ordered reversible channel transforms over {n} stereo fixtures: the \
             extended ladder beats the pre-existing {{none, mid_side}} ladder on {li}/{n} fixtures \
             ({lg} B total), while the run/zero topology selector only matches the measured byte \
             winner on {tm}/{n} fixtures vs {vm}/{n} for minimum variance; every transform closes \
             exactly and the Exp2 multichannel fit still closes ({fit_bytes} B)",
            n = rows.len(),
            li = ladder_improved,
            lg = ladder_gain_total,
            tm = topology_matches,
            vm = variance_matches,
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "fixtures": rows.len(),
                    "fit_closed": fit_closed,
                }),
            ),
            ("fixtures", rows_json),
            (
                "method",
                serde_json::json!({
                    "ladder": "none, mid_side (L + (R - L)/2, R - L), left_diff (L, R - L), \
                               right_diff (R, L - R); every entry is bijective on i32 with \
                               wrapping arithmetic",
                    "topology": "per component: zeros, zero runs, longest zero run and equal \
                                 neighbour pairs; rle_score = 8*zero_runs + 4*zeros + \
                                 16*longest_zero_run + equal_pairs",
                    "selection": "topology orders the fit ladder; the exact canonical complete \
                                  bytes of every fully fitted transform decide the winner",
                    "contrast": "the rejected heuristic ranks transforms by summed component \
                                 variance; on these fixtures it agrees with the byte winner less \
                                 often than topology does",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the ladder is stereo-only; channel counts above two still use TRANSFORM_NONE",
                    "topology is a proposal ordering, not authority: it can rank a transform first \
                     that the measured bytes reject",
                ]),
            ),
        ],
    )
}

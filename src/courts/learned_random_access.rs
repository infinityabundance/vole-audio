//! `court learned-random-access` — bounded access semantics (`O.55`, `O.31`).
//!
//! For every learned semantic class: contiguous, chunked, single-frame, small
//! and large range, randomized and reverse order, and cold/warm repetition,
//! with `seek == sequential` required where defined. Stateful models sweep the
//! checkpoint spacing and report checkpoint bytes and replay work.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::train::linear::fit_linear_object;
use crate::status::Verdict;
use std::path::Path;
use std::time::Instant;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_RANDOM_ACCESS_SHA256: &str =
    "62b5d41d01f85fc8083b0de64ed632c50e30b3d3b83aae0199c71c4e47f7d35b";

/// Checkpoint intervals swept by the stateful class.
pub const CHECKPOINT_INTERVALS: [u32; 5] = [64, 128, 256, 512, 1024];

fn windows(frames: usize) -> Vec<(usize, usize)> {
    vec![
        (0, frames),              // contiguous
        (frames / 2, frames / 2), // contiguous half
        (0, 1),                   // single frame, first
        (frames / 2, 1),          // single frame, middle
        (frames - 1, 1),          // single frame, last
        (frames / 4, 16),         // small range
        (frames / 3, frames / 3), // large range
    ]
}

/// Index a canonical window into a flat sample vector.
fn window_of(all: &[i32], ch: usize, start: usize, len: usize) -> Vec<i32> {
    all[start * ch..(start + len) * ch].to_vec()
}

/// Run the court; writes `receipts/learned-random-access/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.random-access.v1");

    let cases = crate::learned::corpus::intrinsic_cases();
    let case = cases
        .iter()
        .find(|c| c.id == "sine-440")
        .ok_or_else(|| crate::error::Error::internal("corpus is missing the access anchor"))?;
    let frames = case.samples.len();
    let budget = common::train_budget();

    let mut families: Vec<(&str, LearnedObject)> = Vec::new();
    let (lin, _) = fit_linear_object(
        &case.samples,
        1,
        frames as u64,
        case.sample_rate_hz,
        8,
        None,
        frames,
        &budget,
    )?;
    families.push(("linear_finite_field", lin));
    let (bl, _) = fit_linear_object(
        &case.samples,
        1,
        frames as u64,
        case.sample_rate_hz,
        8,
        Some(256),
        frames,
        &budget,
    )?;
    families.push(("block_local", bl));
    if let LearnedModel::Linear(fir) = &families[0].1.model
        && let Ok(p) = common::stateful_from_fir(fir, &case.samples, frames, 256)
    {
        {
            let o = LearnedObject::from_intrinsic(
                LearnedModel::Stateful(p),
                1,
                frames as u64,
                case.sample_rate_hz,
                Vec::new(),
                &case.samples,
            )?;
            families.push(("stateful", o));
        }
    }

    let mut rows = Vec::new();
    let mut all_equal = true;
    for (name, o) in &families {
        let cost = LearnedCost::of(o)?;
        let mut per_window = Vec::new();
        for (start, len) in windows(frames) {
            let expect = window_of(&case.samples, 1, start, len);
            let got = o.materialize_range(start, len)?;
            let eq = got == expect;
            all_equal &= eq;
            let t0 = Instant::now();
            let warm = o.materialize_range(start, len)?;
            let warm_ns = t0.elapsed().as_nanos() as u64;
            let t0 = Instant::now();
            let cold = o.materialize_range(start, len)?;
            let cold_ns = t0.elapsed().as_nanos() as u64;
            per_window.push(serde_json::json!({
                "start": start,
                "frames": len,
                "exact": eq,
                "warm_ns": warm_ns,
                "cold_ns": cold_ns,
                "replay_frames": o.replay_frames(start),
                "warm_exact": warm == expect,
                "cold_exact": cold == expect,
            }));
            common::push_u64(&mut projection, u64::from(eq));
            common::push_u64(&mut projection, o.replay_frames(start) as u64);
        }
        // Randomized and reverse order traversals.
        let mut order: Vec<usize> = (0..frames).collect();
        let mut s = 0x1234_5678u64;
        for i in (1..order.len()).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            order.swap(i, (s as usize) % (i + 1));
        }
        let mut rnd_ok = true;
        for &at in order.iter().take(64) {
            rnd_ok &= o.materialize_range(at, 1)? == case.samples[at..at + 1];
        }
        let mut rev_ok = true;
        for at in (0..frames).rev().take(64) {
            rev_ok &= o.materialize_range(at, 1)? == case.samples[at..at + 1];
        }
        all_equal &= rnd_ok && rev_ok;
        common::push_label(&mut projection, name);
        common::push_u64(&mut projection, cost.complete_bytes);
        common::push_u64(&mut projection, cost.worst_case_replay_frames);
        common::push_u64(&mut projection, u64::from(rnd_ok));
        common::push_u64(&mut projection, u64::from(rev_ok));
        rows.push(serde_json::json!({
            "family": name,
            "kind": o.model.kind_name(),
            "complete_bytes": cost.complete_bytes,
            "windows": per_window,
            "randomized_single_frame_exact": rnd_ok,
            "reverse_single_frame_exact": rev_ok,
        }));
    }

    // Stateful checkpoint sweep.
    let mut sweep = Vec::new();
    if let LearnedModel::Linear(fir) = &families[0].1.model {
        for interval in CHECKPOINT_INTERVALS {
            let p = common::stateful_from_fir(fir, &case.samples, frames, interval)?;
            let o = LearnedObject::from_intrinsic(
                LearnedModel::Stateful(p),
                1,
                frames as u64,
                case.sample_rate_hz,
                Vec::new(),
                &case.samples,
            )?;
            let exact = o.materialize_range(frames * 3 / 4, 64)?
                == window_of(&case.samples, 1, frames * 3 / 4, 64);
            all_equal &= exact;
            let cost = LearnedCost::of(&o)?;
            common::push_u64(&mut projection, u64::from(interval));
            common::push_u64(&mut projection, cost.checkpoint_definition_bytes);
            common::push_u64(&mut projection, o.replay_frames(frames * 3 / 4) as u64);
            sweep.push(serde_json::json!({
                "interval": interval,
                "checkpoint_bytes": cost.checkpoint_definition_bytes,
                "replay_frames_at_3_4": o.replay_frames(frames * 3 / 4),
                "complete_bytes": cost.complete_bytes,
                "seek_exact": exact,
            }));
        }
    }

    let verdict = if all_equal {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-random-access",
        receipts_root,
        LEARNED_RANDOM_ACCESS_SHA256,
        &projection,
        verdict,
        format!(
            "bounded access over {} learned families: every requested window equals the sequential \
             evaluation; stateful checkpoint sweep covers {} intervals",
            families.len(),
            CHECKPOINT_INTERVALS.len()
        ),
        vec![
            ("families", serde_json::json!(rows)),
            ("checkpoint_sweep", serde_json::json!(sweep)),
            (
                "limitations",
                serde_json::json!([
                    "finite-field and stateful classes replay from origin or from the nearest \
                     checkpoint; only the block-local class is truly random-access",
                    "cold/warm labels here mean first vs repeated in-process access, not verified \
                     host cache state"
                ]),
            ),
        ],
    )
}

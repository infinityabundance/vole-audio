//! `court compound` — native procedural composition (optimization Track A).
//!
//! A **known-construction** court: each case is built from an explicit
//! composition graph, materialized to its exact canonical signal, and then
//! priced against the literal floor, FLAC and the existing bounded VOLE inverse
//! compiler. It demonstrates the capability and the size delta; it does **not**
//! claim a generic inverse-compiler win, and it never treats a source label as
//! decoder authority.
//!
//! The composition profile is the separate experimental profile
//! `vole.audio.compound.exp1` (Compound semantics were not frozen in `u1/v1`,
//! so no payload is implemented behind tag `0x0A`).

use crate::compound::{CompoundGraph, CompoundNode, CompoundOp};
use crate::courts::learned_common as common;
use crate::error::Result;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const COMPOUND_SHA256: &str =
    "31ddcb6647ba4ade3b67db1805944b17023d68e225efaec051d3071385efce8b";

const FRAMES: u64 = 48_000;
const RATE: u32 = 48_000;

fn osc(freq: u32, amp_q16: i32, phase0: u64) -> CompoundNode {
    CompoundNode {
        op: CompoundOp::Oscillator {
            freq_hz: freq,
            amp_q16,
            phase0,
        },
        children: Vec::new(),
    }
}

fn add(children: Vec<u16>) -> CompoundNode {
    CompoundNode {
        op: CompoundOp::Add,
        children,
    }
}

/// A three-voice detuned polyphony.
fn polyphony() -> CompoundGraph {
    CompoundGraph {
        channels: 1,
        frames: FRAMES,
        sample_rate_hz: RATE,
        nodes: vec![
            osc(220, 1 << 14, 0),
            osc(221, 1 << 13, 1 << 40),
            osc(330, 1 << 13, 1 << 41),
            add(vec![0, 1, 2]),
        ],
    }
}

/// A percussive one-shot: a tone shaped by an exact ADSR.
fn percussion() -> CompoundGraph {
    CompoundGraph {
        channels: 1,
        frames: FRAMES,
        sample_rate_hz: RATE,
        nodes: vec![
            osc(180, 1 << 15, 0),
            CompoundNode {
                op: CompoundOp::Envelope {
                    attack_frames: 1,
                    decay_frames: 9_600,
                    sustain_q16: 0,
                    release_frames: 4_800,
                    t_on: 0,
                    t_off: Some(24_000),
                },
                children: vec![0],
            },
        ],
    }
}

/// A layered pad: oscillators + gain + delay + envelope + add.
fn layered_pad() -> CompoundGraph {
    CompoundGraph {
        channels: 1,
        frames: FRAMES,
        sample_rate_hz: RATE,
        nodes: vec![
            osc(110, 1 << 14, 0),
            osc(165, 1 << 13, 1 << 30),
            CompoundNode {
                op: CompoundOp::Gain { q16: 1 << 14 },
                children: vec![1],
            },
            CompoundNode {
                op: CompoundOp::Delay { frames: 64 },
                children: vec![2],
            },
            CompoundNode {
                op: CompoundOp::Envelope {
                    attack_frames: 2_400,
                    decay_frames: 0,
                    sustain_q16: 1 << 15,
                    release_frames: 2_400,
                    t_on: 0,
                    t_off: Some(36_000),
                },
                children: vec![0],
            },
            add(vec![4, 3]),
        ],
    }
}

/// Run the court; writes `receipts/compound/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.compound.court.v1");
    common::push_label(
        &mut projection,
        &String::from_utf8_lossy(crate::compound::COMPOUND_PROFILE_TAG),
    );

    let cases: Vec<(&str, CompoundGraph)> = vec![
        ("polyphony-3", polyphony()),
        ("percussion-adsr", percussion()),
        ("layered-pad", layered_pad()),
    ];
    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut compound_smaller_than_literal = true;
    for (name, graph) in &cases {
        let signal = graph.materialize()?;
        let exact = graph.closes_to(&signal) && signal.len() == FRAMES as usize;
        all_exact &= exact;
        let compound_bytes = graph.complete_bytes();
        let literal_bytes = crate::inverse::cost::canonical_u1_literal_bytes(FRAMES, 1);
        let flac_bytes = common::flac5(&signal, 1, RATE).unwrap_or(u64::MAX);
        let u1_best = common::u1_best(name, &signal, 1)
            .map(|(b, _)| b)
            .unwrap_or(u64::MAX);
        compound_smaller_than_literal &= compound_bytes < literal_bytes;
        common::push_label(&mut projection, name);
        common::push_u64(&mut projection, compound_bytes);
        common::push_u64(&mut projection, literal_bytes);
        common::push_u64(&mut projection, flac_bytes);
        common::push_u64(&mut projection, u1_best);
        rows.push(serde_json::json!({
            "id": name,
            "frames": FRAMES,
            "nodes": graph.nodes.len(),
            "compound_bytes": compound_bytes,
            "literal_bytes": literal_bytes,
            "flac_bytes": if flac_bytes == u64::MAX { None } else { Some(flac_bytes) },
            "u1_best_bytes": if u1_best == u64::MAX { None } else { Some(u1_best) },
            "ratio_literal_over_compound": literal_bytes as f64 / compound_bytes.max(1) as f64,
            "exact": exact,
        }));
    }

    let verdict = if all_exact && compound_smaller_than_literal {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_with_profile(
        "compound",
        crate::learned::profile::LearnedProfile::Exp2,
        receipts_root,
        COMPOUND_SHA256,
        &projection,
        verdict,
        format!(
            "native procedural composition (experimental profile vole.audio.compound.exp1) over {} \
             known constructions: exact materialization, priced against the literal floor, FLAC and \
             the bounded VOLE inverse compiler",
            rows.len()
        ),
        vec![
            (
                "profile",
                serde_json::json!({
                    "id": crate::compound::COMPOUND_PROFILE,
                    "magic": String::from_utf8_lossy(crate::compound::COMPOUND_MAGIC),
                    "note": "Compound semantics are NOT frozen in u1/v1; this is a separate versioned \
                             experimental profile. No payload is implemented behind tag 0x0A.",
                }),
            ),
            ("cases", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "known-construction court: candidate graphs are supplied explicitly, not \
                     inferred from arbitrary PCM; this is a capability and size-delta demonstration",
                    "composition is mono-first in this build",
                    "the vocabulary is deliberately small (silence, constant, frozen DDS oscillator, \
                     gain, delay, exact ADSR, exact Add); no open-ended DSP language",
                ]),
            ),
        ],
    )
}

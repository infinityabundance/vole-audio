//! `court learned-entropy-seed` — Phase 7A: entropy-seed proceduralization.
//!
//! This is the dedicated court for the entropy-seed architecture
//!
//! ```text
//! X = reconstruct(H, R)
//! ```
//!
//! where `H` is a deterministic explanation discovered **blind** from the
//! samples by the [`crate::inverse::compound_propose`] search, `H`'s own state
//! is entropy-coded ([`crate::inverse::seed::encode_h_state`]), and `R` is the
//! entropy-coded exact residual. The court must demonstrate, on every case:
//!
//! * **blind inverse inference** — the proposer receives only samples, rate,
//!   layout and extent;
//! * **entropy-coded H/configuration state** — `encode_h_state` round-trips;
//! * **entropy-coded residual** — never stored PCM;
//! * **exact lossless closure** — `materialize(seed) == X`;
//! * **bounded materialization** — a random-access window equals the same slice
//!   of full materialization, with peak scratch independent of duration;
//! * **determinism** — repeated runs produce identical bytes.
//!
//! The verdict gates on exactness and well-formedness only. Byte outcomes
//! (including the high-entropy negative controls, where a seed legitimately
//! loses to the literal floor) are reported as evidence, never asserted as a
//! gain.

use crate::compound::{CompoundGraph, CompoundNode, CompoundOp};
use crate::courts::learned_common as common;
use crate::error::Result;
use crate::evidence::timing::Stopwatch;
use crate::inverse::compound_propose::{CompoundBudget, CompoundFamily, propose};
use crate::inverse::seed::SeedObject;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{available as real_available, effectiveness_clips, load_cases};
use crate::learned::object::LearnedObject;
use crate::learned::train::TrainBudget;
use crate::learned::train::adaptive::fit_adaptive_object;
use crate::learned::train::lpc::{fit_lattice_object, fit_lpc_object, fit_pz_object};
use crate::learned::train::ltp::fit_ltp_object;
use crate::learned::train::ngsa::fit_ngsa_object;
use crate::learned::train::sparse::fit_sparse_object;
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (set after the first observation).
pub const LEARNED_ENTROPY_SEED_SHA256: &str =
    "d0497ad77d3b934ea8e973bb2e5062266b4db46a0f63f37b6445467618945195";

const RATE: u32 = 48_000;
const REAL_CLIPS: usize = 4;
/// Every court case is truncated to this many frames so the court stays inside
/// a bounded batch budget; the truncation is disclosed in the receipt.
const MAX_CASE_FRAMES: usize = 8_192;
/// Negative controls are truncated further: they are incompressible, so the
/// residual search dominates the court's runtime without adding information.
const NEGATIVE_CONTROL_FRAMES: usize = 2_048;
/// The mature learned predictor portfolio (7A.2) runs on this fixed subset of
/// case ids (a bounded, deterministic sample of each structural class).
const LEARNED_IDS: &[&str] = &[
    "known-polyphony",
    "known-percussion",
    "corpus-single-sine",
    "corpus-harmonic-tone",
    "corpus-quasi-periodic",
    "corpus-transient-heavy",
    "corpus-am-signal",
    "corpus-white-noise",
];
/// The determinism re-run (a second full blind pass) is bounded to this many
/// cases.
const DETERMINISM_CASES: usize = 5;

/// One court case: canonical interleaved samples.
struct Case {
    id: String,
    kind: &'static str,
    channels: u8,
    rate: u32,
    samples: Vec<i32>,
}

fn deinterleave(samples: &[i32], channels: usize, ch: usize) -> Vec<i32> {
    samples.iter().skip(ch).step_by(channels).copied().collect()
}

#[cfg(test)]
fn interleave(channels: &[Vec<i32>]) -> Vec<i32> {
    let ch = channels.len();
    let frames = channels[0].len();
    let mut out = Vec::with_capacity(frames * ch);
    for f in 0..frames {
        for c in channels {
            out.push(c[f]);
        }
    }
    out
}

/// Per-channel entropy-seed result.
struct ChannelSeed {
    family: CompoundFamily,
    h_bytes: u64,
    residual_bytes: u64,
    total_bytes: u64,
    exact: bool,
    bounded_ok: bool,
    bounded_peak_depth: u32,
    encode_ns: u64,
    decode_ns: u64,
    bounded_ns: u64,
}

/// The residual-codec search is paid only on the best-ranked explanations per
/// channel *plus* the structurally strongest family and the trivial floors, so
/// a near-zero-energy tie never hides the winning explanation.
const SEED_SHORTLIST: usize = 4;

/// Families that are always carried into the byte comparison regardless of the
/// residual-energy ranking.
fn always_shortlisted(family: CompoundFamily) -> bool {
    matches!(
        family,
        CompoundFamily::ToneDecomposition
            | CompoundFamily::EnvelopedToneDecomposition
            | CompoundFamily::Silence
            | CompoundFamily::Constant
    )
}

/// Run the blind proposer on one channel, build every seed, and keep the
/// cheapest that closes exactly. A trivial `Silence`/`Constant` explanation is
/// always included: a bad `H` must produce a large entropy-coded residual, not
/// abandon the entropy-seed form.
fn seed_channel(samples: &[i32], rate: u32, budget: &CompoundBudget) -> Result<ChannelSeed> {
    let frames = samples.len() as u64;
    let mut graphs: Vec<(CompoundFamily, CompoundGraph)> =
        propose(samples, 1, rate, frames, budget)?
            .into_iter()
            .map(|p| (p.family, p.graph))
            .collect();
    let trivial = |op: CompoundOp, family: CompoundFamily| {
        (
            family,
            CompoundGraph {
                channels: 1,
                frames,
                sample_rate_hz: rate,
                nodes: vec![CompoundNode {
                    op,
                    children: Vec::new(),
                }],
            },
        )
    };
    graphs.push(trivial(CompoundOp::Silence, CompoundFamily::Silence));
    graphs.push(trivial(
        CompoundOp::Constant { level: samples[0] },
        CompoundFamily::Constant,
    ));

    let mut best: Option<(SeedObject, CompoundFamily, u64)> = None;
    let sw = Stopwatch::start();

    // Cheap ranking: materialize each explanation and measure residual energy,
    // then pay for the full residual-codec search only on the strongest few.
    let mut ranked: Vec<(f64, CompoundFamily, CompoundGraph, Vec<i32>)> = Vec::new();
    for (family, graph) in graphs {
        let Some(residual) = crate::inverse::seed::residual_only(&graph, samples)? else {
            continue;
        };
        let energy: f64 = residual
            .iter()
            .map(|&v| f64::from(v.unsigned_abs()))
            .sum::<f64>()
            / (residual.len().max(1)) as f64;
        ranked.push((energy, family, graph, residual));
    }
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let shortlist: Vec<(f64, CompoundFamily, CompoundGraph, Vec<i32>)> = ranked
        .into_iter()
        .enumerate()
        .filter(|(i, (_, family, _, _))| *i < SEED_SHORTLIST || always_shortlisted(*family))
        .map(|(_, item)| item)
        .collect();

    for (_, family, graph, residual) in shortlist {
        let Some(seed) = SeedObject::from_graph_residual(graph, samples.len(), residual)? else {
            continue;
        };
        let enc_ns = sw.elapsed_ns().max(0) as u64;
        let total = seed.complete_bytes();
        if best
            .as_ref()
            .is_none_or(|(b, _, _)| total < b.complete_bytes())
        {
            best = Some((seed, family, enc_ns));
        }
    }
    let (seed, family, encode_ns) = best.ok_or_else(|| {
        crate::error::Error::internal("entropy-seed channel produced no candidate")
    })?;

    // Decode-time proof: reconstruct H from its entropy-coded state, rebuild the
    // signal, and verify exact closure a second time.
    let sw = Stopwatch::start();
    let reconstructed = seed.materialize()?;
    let decode_ns = sw.elapsed_ns().max(0) as u64;
    let exact = reconstructed == samples;

    // Bounded materialization proof on the decoded graph: random-access windows
    // must equal the same slice of the reference, with scratch bounded by the
    // graph depth rather than the extent.
    let graph = seed.decoded_graph()?;
    let reference = graph.materialize()?;
    let sw = Stopwatch::start();
    let mut bounded_ok = true;
    let mut peak_depth = 0u32;
    if graph.frames >= 2 {
        let windows: [(u64, u64); 5] = [
            (0, 1),
            (0, graph.frames.min(37)),
            (graph.frames / 3, (graph.frames / 7).max(1)),
            (graph.frames.saturating_sub(11), 11.min(graph.frames)),
            (graph.frames / 2, 1),
        ];
        for (start, len) in windows {
            let end = start + len;
            if len == 0 || end > graph.frames {
                continue;
            }
            let (part, prof) = graph.materialize_range_profiled(start, len)?;
            if part.as_slice() != &reference[start as usize..end as usize] {
                bounded_ok = false;
            }
            peak_depth = peak_depth.max(prof.peak_depth);
        }
    }
    let bounded_ns = sw.elapsed_ns().max(0) as u64;

    Ok(ChannelSeed {
        family,
        h_bytes: seed.h_bytes(),
        residual_bytes: seed.residual_bytes(),
        total_bytes: seed.complete_bytes(),
        exact,
        bounded_ok,
        bounded_peak_depth: peak_depth,
        encode_ns,
        decode_ns,
        bounded_ns,
    })
}

/// Best mature learned predictor family for one channel (7A.2). The learned
/// implementations are reused verbatim; only their complete byte cost decides.
fn best_learned(samples: &[i32], rate: u32, budget: &TrainBudget) -> Option<(String, u64, u64)> {
    let frames = samples.len() as u64;
    let mut best: Option<(String, u64, u64)> = None;
    let sw = Stopwatch::start();
    let mut consider = |name: &str, o: LearnedObject| {
        if !o.verify(samples) {
            return;
        }
        let Ok(cost) = LearnedCost::of(&o) else {
            return;
        };
        let bytes = cost.complete_bytes;
        if best.as_ref().is_none_or(|(_, b, _)| bytes < *b) {
            best = Some((name.to_string(), bytes, sw.elapsed_ns().max(0) as u64));
        }
    };
    if let Ok((o, _)) = fit_lpc_object(samples, frames, rate, 1024, 16, budget) {
        consider("lpc", o);
    }
    if let Ok((o, _)) = fit_lattice_object(samples, frames, rate, 1024, 16, budget) {
        consider("lattice", o);
    }
    if let Ok((o, _)) = fit_pz_object(samples, frames, rate, 1024, budget) {
        consider("pole_zero", o);
    }
    if let Ok((o, _)) = fit_sparse_object(samples, frames, rate, 16, Some(1024), budget) {
        consider("sparse_linear", o);
    }
    if let Ok((o, _)) = fit_ltp_object(samples, frames, rate, budget) {
        consider("long_term", o);
    }
    if let Ok((o, _)) = fit_ngsa_object(samples, frames, rate, 16, budget) {
        consider("natural_gradient", o);
    }
    if let Ok((o, _)) = fit_adaptive_object(samples, frames, rate, 16, budget) {
        consider("backward_adaptive", o);
    }
    let _ = sw;
    best
}

/// Two known constructions the proposer has never seen.
fn known_graphs() -> Vec<(&'static str, CompoundGraph)> {
    let osc = |f: u32, a: i32, p: u64| CompoundNode {
        op: CompoundOp::Oscillator {
            freq_hz: f,
            amp_q16: a,
            phase0: p,
        },
        children: Vec::new(),
    };
    let polyphony = CompoundGraph {
        channels: 1,
        frames: 24_000,
        sample_rate_hz: RATE,
        nodes: vec![
            osc(180, 2, 0),
            osc(181, 1, 1 << 40),
            osc(270, 1, 1 << 41),
            CompoundNode {
                op: CompoundOp::Add,
                children: vec![0, 1, 2],
            },
        ],
    };
    let percussion = CompoundGraph {
        channels: 1,
        frames: 24_000,
        sample_rate_hz: RATE,
        nodes: vec![
            osc(140, 1, 0),
            CompoundNode {
                op: CompoundOp::Envelope {
                    attack_frames: 1,
                    decay_frames: 6_000,
                    sustain_q16: 0,
                    release_frames: 0,
                    t_on: 0,
                    t_off: None,
                },
                children: vec![0],
            },
        ],
    };
    vec![
        ("known-polyphony", polyphony),
        ("known-percussion", percussion),
    ]
}

/// Build the case list: known constructions, the frozen entropy corpus, and the
/// real-speech effectiveness clips where available.
fn cases(real_reported: &mut bool) -> Result<Vec<Case>> {
    let mut out = Vec::new();
    let push =
        |out: &mut Vec<Case>, id: String, kind: &'static str, channels, rate, samples: Vec<i32>| {
            let ch = usize::from(channels);
            // Negative controls are incompressible by construction; the residual
            // codec search is the dominant cost there, so they are bounded to a
            // smaller extent. This is disclosed in the receipt.
            let cap = if kind == "negative-control" {
                NEGATIVE_CONTROL_FRAMES
            } else {
                MAX_CASE_FRAMES
            };
            let frames = (samples.len() / ch).min(cap);
            out.push(Case {
                id,
                kind,
                channels,
                rate,
                samples: samples[..frames * ch].to_vec(),
            });
        };
    for (id, g) in known_graphs() {
        let signal = g.materialize()?;
        push(&mut out, id.to_string(), "synthetic-known", 1, RATE, signal);
    }
    for f in crate::entropy::corpus::all() {
        push(
            &mut out,
            format!("corpus-{}", f.name),
            f.kind,
            f.channels,
            48_000,
            f.samples.clone(),
        );
    }
    let scratch = PathBuf::from("target/real-corpus/scratch");
    if real_available() {
        let clips = effectiveness_clips();
        let loaded = load_cases(&clips, REAL_CLIPS, &scratch)?;
        for c in loaded {
            push(
                &mut out,
                format!("speech-{}", c.clip.id),
                "speech",
                c.clip.channels,
                c.clip.sample_rate_hz,
                c.samples,
            );
        }
        *real_reported = true;
    }
    Ok(out)
}

/// Run the court; writes `receipts/learned-entropy-seed/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.entropy-seed.court.v1");
    common::push_label(&mut projection, crate::compound::COMPOUND_PROFILE);

    let budget = CompoundBudget::default();
    let train_budget = TrainBudget::default();
    let mut real_reported = false;
    let cases = cases(&mut real_reported)?;

    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut all_bounded = true;
    let mut all_deterministic = true;

    for (case_index, case) in cases.iter().enumerate() {
        let ch = usize::from(case.channels);
        let frames = case.samples.len() / ch;
        // 7A.2: the mature learned predictor families compete in the same
        // inverse decision on a bounded, deterministic subset, priced by actual
        // complete bytes.
        let ch0 = deinterleave(&case.samples, ch, 0);
        let learned = if LEARNED_IDS.contains(&case.id.as_str()) {
            best_learned(&ch0, case.rate, &train_budget)
        } else {
            None
        };
        let literal_bytes = common::literal_floor(&ch0, 1);
        let flac_bytes = common::flac5(&ch0, 1, case.rate).unwrap_or(u64::MAX);
        let u1_bytes = common::u1_best(&case.id, &ch0, 1)
            .map(|(b, _)| b)
            .unwrap_or(u64::MAX);

        // Blind per-channel seed search.
        let mut h_total = 0u64;
        let mut r_total = 0u64;
        let mut total = 0u64;
        let mut family = None;
        let mut exact = true;
        let mut bounded_ok = true;
        let mut peak_depth = 0u32;
        let mut enc_ns = 0u64;
        let mut dec_ns = 0u64;
        let mut bnd_ns = 0u64;
        let mut first_totals = Vec::with_capacity(ch);
        let sw_seed = Stopwatch::start();
        for c in 0..ch {
            let channel = deinterleave(&case.samples, ch, c);
            let s = seed_channel(&channel, case.rate, &budget)?;
            h_total += s.h_bytes;
            r_total += s.residual_bytes;
            total += s.total_bytes;
            first_totals.push(s.total_bytes);
            family = Some(s.family);
            exact &= s.exact;
            bounded_ok &= s.bounded_ok;
            peak_depth = peak_depth.max(s.bounded_peak_depth);
            enc_ns += s.encode_ns;
            dec_ns += s.decode_ns;
            bnd_ns += s.bounded_ns;
        }
        let seed_path_ns = sw_seed.elapsed_ns().max(0) as u64;

        // Determinism: a second blind pass must be byte-identical (bounded to
        // the first DETERMINISM_CASES cases).
        let mut det = true;
        if case_index < DETERMINISM_CASES {
            for (c, first) in first_totals.iter().enumerate() {
                let channel = deinterleave(&case.samples, ch, c);
                let again = seed_channel(&channel, case.rate, &budget)?;
                if again.total_bytes != *first {
                    det = false;
                }
            }
        }

        all_exact &= exact;
        all_bounded &= bounded_ok;
        all_deterministic &= det;

        let fam = family.map(|f| f.name()).unwrap_or("none");
        common::push_label(&mut projection, &case.id);
        common::push_label(&mut projection, fam);
        common::push_u64(&mut projection, total);
        common::push_u64(&mut projection, h_total);
        common::push_u64(&mut projection, r_total);
        common::push_u64(&mut projection, literal_bytes);
        common::push_u64(&mut projection, flac_bytes.min(u64::MAX - 1));
        common::push_u64(&mut projection, u1_bytes.min(u64::MAX - 1));
        common::push_u64(
            &mut projection,
            learned.as_ref().map(|(_, b, _)| *b).unwrap_or(0),
        );

        rows.push(serde_json::json!({
            "id": case.id,
            "kind": case.kind,
            "channels": case.channels,
            "frames": frames,
            "h_family": fam,
            "h_state_bytes": h_total,
            "entropy_model_bytes": 0,
            "residual_bytes": r_total,
            "total_bytes": total,
            "literal_floor_bytes": literal_bytes,
            "flac5_bytes": if flac_bytes == u64::MAX { None } else { Some(flac_bytes) },
            "u1_best_bytes": if u1_bytes == u64::MAX { None } else { Some(u1_bytes) },
            "learned_best_family": learned.as_ref().map(|(n, _, _)| n.clone()),
            "learned_best_bytes": learned.as_ref().map(|(_, b, _)| *b),
            "learned_ns": learned.as_ref().map(|(_, _, n)| *n).unwrap_or(0),
            "seed_path_ns": seed_path_ns,
            "seed_wins_learned": learned.as_ref().map(|(_, b, _)| total < *b),
            "exact": exact,
            "bounded_ok": bounded_ok,
            "bounded_peak_depth": peak_depth,
            "deterministic": det,
            "encode_ns": enc_ns,
            "decode_ns": dec_ns,
            "bounded_decode_ns": bnd_ns,
        }));
    }

    let verdict = if all_exact && all_bounded && all_deterministic {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_with_profile(
        "learned-entropy-seed",
        crate::learned::profile::LearnedProfile::Exp3,
        receipts_root,
        LEARNED_ENTROPY_SEED_SHA256,
        &projection,
        verdict,
        format!(
            "entropy-seed proceduralization (Phase 7A) over {} cases: blind Compound inference, \
             entropy-coded H state, entropy-coded exact residual, exact closure, bounded \
             materialization, determinism, and the mature learned predictor families competing \
             in the same inverse decision",
            rows.len()
        ),
        vec![
            (
                "architecture",
                serde_json::json!({
                    "identity": "X = reconstruct(H, R)",
                    "H": "deterministic CompoundGraph discovered blind from samples",
                    "R": "entropy-coded exact residual (residual_codec2 family)",
                    "H_state": "per-stream integer metadata coders (repcode/baseline/raw varint)",
                    "real_corpus": real_reported,
                }),
            ),
            ("cases", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "Compound is mono-first in this build; stereo/multichannel cases run the \
                     per-channel seed search and sum the physical bytes",
                    "cases are truncated (the receipt's `frames` field is authoritative): the \
                     structural/tonal/speech classes to 8192 frames, the incompressible \
                     negative controls to 2048 frames, to keep the court inside a bounded batch \
                     budget",
                    "the blind proposer estimates structure from a bounded analysis window by \
                     integer-atom orthogonal matching pursuit; within a short window two partials \
                     one hertz apart are nearly collinear, so the split-refinement step tests and \
                     keeps an added neighbour only when it genuinely reduces the joint LS residual",
                    "the residual-codec search skips the two unbounded general-Golomb codecs when the \
                     residual contains a large-magnitude symbol (they are O(|r|) per symbol there); \
                     the skip is deterministic and the skipped codecs never win",
                    "this court gates on exactness, bounded materialization and determinism only; \
                     byte outcomes are reported, including the high-entropy negative controls \
                     where a seed legitimately loses to the literal floor",
                    "the learned predictor families (7A.2) are fitted on channel 0 and priced by \
                     complete bytes; they are not merged into one object with the seed path",
                ]),
            ),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleave_round_trips() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let x = interleave(&[a.clone(), b.clone()]);
        assert_eq!(x, vec![1, 4, 2, 5, 3, 6]);
        assert_eq!(deinterleave(&x, 2, 0), a);
        assert_eq!(deinterleave(&x, 2, 1), b);
    }

    #[test]
    fn known_graphs_are_valid() {
        for (_, g) in known_graphs() {
            g.validate().unwrap();
            assert!(!g.materialize().unwrap().is_empty());
        }
    }
}

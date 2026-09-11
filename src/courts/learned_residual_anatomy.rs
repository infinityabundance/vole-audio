//! `court learned-residual-anatomy` — fourth-pass **Seal E0** (diagnostic only).
//!
//! The S8 speech campaign already beats FLAC-8, so the open question is no
//! longer "which predictor" but "how much *conditional* structure the exact
//! residual still carries". This court does **not** change any format, codec,
//! model or profile: it decodes the exact residual that the current S8 portfolio
//! already selects, binarizes it canonically, and measures the empirical
//! conditional entropy of each bit under the decoder-visible contexts named in
//! the fourth-pass report:
//!
//! ```text
//! H(bit)                              order-0
//! H(bit | bit position)               position within the Exp-Golomb token
//! H(bit | current prefix)             full already-decoded token prefix
//! H(bit | previous residual bucket)   previous |r| bit-length bucket
//! H(bit | previous two buckets)       previous two magnitude buckets
//! H(bit | previous sign)              previous residual sign
//! H(bit | residual FSM state)         magnitude-class finite-state machine
//! H(bit | matched-lag bucket)         oracle lag from residual autocorrelation
//! H(bit | local-energy bucket)        trailing 64-sample mean |r|
//! H(bit | predictor disagreement)     selected predictor vs previous sample
//! ```
//!
//! Two populations are measured, always kept separate:
//!
//! * **speech effectiveness** — the exact winner residual of the S8 portfolio
//!   (the same portfolio `court learned-speech` runs) over the frozen
//!   effectiveness clips; this is the population that drives the E-ladder;
//! * **representative Phase-M objects** — canonical flagship-corpus objects
//!   (generated, never stored) deinterleaved per channel, with a cheap exact
//!   fixed-difference closure as the predictor. Their purpose is only to show
//!   whether a context family has information on non-speech material too.
//!
//! The **held-out Mode C split is deliberately untouched.** The result is an
//! anatomy of what the current residual code still pays for, not a new codec.
//!
//! Only deterministic integers enter the frozen projection; every entropy rate
//! is recorded in the receipt as an extra. The receipt is written in the Exp3
//! profile because the residuals come from the Exp3 S8 portfolio.

use crate::corpus::{generate, specs};
use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::{Error, Result};
use crate::learned::accounting::LearnedCost;
use crate::learned::anatomy::{
    Anatomy, CONTEXT_COUNT, CONTEXT_NAMES, MAG_BUCKETS, anatomy, merge_into, savings_json,
};
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::object::LearnedObject;
use crate::learned::train::TrainBudget;
use crate::learned::train::fixed::fit_fixed_diff_sweep;
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_RESIDUAL_ANATOMY_SHA256: &str =
    "40881a37c73e439fc0f53c47a2a3fdb2d239a972c1fca000fb7282f8dd65aed7";

/// Maximum representative Phase-M objects (one per distinct class tuple).
const MAX_PHASE_M_OBJECTS: usize = 12;

/// Bound the residual samples analysed per channel stream (a statistical bound;
/// the entropy estimator does not need the whole object and the bound keeps the
/// diagnostic cheap enough for every seal).
const MAX_ANALYSIS_SAMPLES: usize = 32_768;

/// The fixed-difference block-size ladder used for the Phase-M predictor.
const PHASE_M_LADDER: [Option<u32>; 5] = [None, Some(4096), Some(2048), Some(1024), Some(512)];

/// One measured population: per-object rows, the merged anatomy and exactness.
struct Population {
    name: &'static str,
    rows: Vec<serde_json::Value>,
    merged: Anatomy,
    all_exact: bool,
}

impl Population {
    fn new(name: &'static str) -> Self {
        Population {
            name,
            rows: Vec::new(),
            merged: Anatomy::default(),
            all_exact: true,
        }
    }
}

/// Serialize one anatomy, including the derived saving versus order-0.
fn anatomy_json(a: &Anatomy) -> serde_json::Value {
    let contexts: serde_json::Map<String, serde_json::Value> = CONTEXT_NAMES
        .iter()
        .zip(a.contexts)
        .map(|(n, c)| ((*n).to_string(), serde_json::json!(c)))
        .collect();
    serde_json::json!({
        "residual_samples": a.residual_samples,
        "bits": a.bits,
        "lag": a.lag,
        "zero_rate": a.zero_rate,
        "neg_rate": a.neg_rate,
        "mean_abs": a.mean_abs,
        "max_abs": a.max_abs,
        "order0_bits_per_residual": a.order0,
        "by_pos": a.by_pos,
        "by_prefix": a.by_prefix,
        "by_prev_mag": a.by_prev_mag,
        "by_prev2_mag": a.by_prev2_mag,
        "by_prev_pair": a.by_prev_pair,
        "by_prev_sign": a.by_prev_sign,
        "by_fsm": a.by_fsm,
        "by_lag": a.by_lag,
        "by_energy": a.by_energy,
        "by_disagree": a.by_disagree,
        "bits_per_residual_saved_vs_order0": savings_json(a),
        "contexts": serde_json::Value::Object(contexts),
        "mag_hist": a.mag_hist,
    })
}

/// Pick the smallest exact candidate in a portfolio, verifying every member.
fn winner<'a>(
    cands: &'a [(&'static str, LearnedObject)],
    source: &[i32],
) -> Result<(&'a str, &'a LearnedObject)> {
    let mut best: Option<(&str, &LearnedObject, u64)> = None;
    for (name, o) in cands {
        if !o.verify(source) {
            return Err(Error::integrity(format!(
                "S8 candidate '{name}' does not close exactly to the intrinsic"
            )));
        }
        let b = LearnedCost::of(o)?.complete_bytes;
        if best.as_ref().is_none_or(|(_, _, bb)| b < *bb) {
            best = Some((name, o, b));
        }
    }
    best.map(|(n, o, _)| (n, o))
        .ok_or_else(|| Error::internal("the S8 portfolio is empty"))
}

/// The speech-effectiveness population: the exact S8 winner residual per clip.
fn speech_population(budget: &TrainBudget) -> Result<Population> {
    let scratch = PathBuf::from("target/real-corpus/scratch");
    let cases = load_cases(&effectiveness_clips(), speech::CLIPS_PER_SPLIT, &scratch)?;
    let mut pop = Population::new("speech_effectiveness");
    for case in &cases {
        let cands = speech::portfolio(case, budget)?;
        let (family, o) = winner(&cands, &case.samples)?;
        let cost = LearnedCost::of(o)?;
        let residual = o.residual()?;
        let a = anatomy(&residual, &case.samples);
        pop.all_exact &= o.verify(&case.samples);
        pop.rows.push(serde_json::json!({
            "id": case.clip.id,
            "population": pop.name,
            "split": case.clip.split,
            "family": family,
            "residual_codec": o.residual_codec.name(),
            "complete_bytes": cost.complete_bytes,
            "model_bytes": cost.model_bytes,
            "residual_bytes": cost.residual_bytes,
            "anatomy": anatomy_json(&a),
        }));
        merge_into(&mut pop.merged, &a);
    }
    Ok(pop)
}

/// Greedily choose one representative spec per distinct
/// `(source_structure, entropy, temporal)` class tuple, so the sample spans
/// literal / oscillator / wavetable / repetition / compound / residual / noise
/// structures and their entropy and temporal characters rather than burning
/// every slot on one amplitude sweep.
fn representative_specs(all: &[generate::Spec]) -> Vec<&generate::Spec> {
    let mut seen: Vec<String> = Vec::new();
    let mut out: Vec<&generate::Spec> = Vec::new();
    for spec in all {
        if out.len() >= MAX_PHASE_M_OBJECTS {
            break;
        }
        let key = format!(
            "{}/{}/{}",
            spec.source_structure.as_str(),
            spec.entropy.as_str(),
            spec.temporal.as_str(),
        );
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        out.push(spec);
    }
    out
}

/// The representative Phase-M population: cheap exact fixed-difference residuals
/// per deinterleaved channel stream of representative corpus objects.
fn phase_m_population(budget: &TrainBudget) -> Result<Population> {
    let all = specs::specs();
    let selected = representative_specs(&all);
    let mut pop = Population::new("phase_m_representative");
    for spec in selected {
        let samples = generate::generate(spec)?;
        let per_frame = usize::from(spec.channels).max(1);
        for ch in 0..per_frame {
            let mut chan: Vec<i32> = samples[ch..].iter().step_by(per_frame).copied().collect();
            let total = chan.len();
            chan.truncate(total.min(MAX_ANALYSIS_SAMPLES));
            let frames = chan.len() as u64;
            if frames == 0 {
                continue;
            }
            let (o, _) = fit_fixed_diff_sweep(
                &chan,
                frames,
                spec.sample_rate_hz,
                4,
                &PHASE_M_LADDER,
                budget,
            )?;
            if !o.verify(&chan) {
                pop.all_exact = false;
                continue;
            }
            let residual = o.residual()?;
            let a = anatomy(&residual, &chan);
            let cost = LearnedCost::of(&o)?;
            pop.rows.push(serde_json::json!({
                "id": format!("{}#ch{ch}", spec.id),
                "population": pop.name,
                "class": {
                    "source_structure": spec.source_structure.as_str(),
                    "temporal": spec.temporal.as_str(),
                    "entropy": spec.entropy.as_str(),
                    "amplitude": spec.amplitude.as_str(),
                    "channel_structure": spec.channel_structure.as_str(),
                },
                "frames_total": total,
                "frames_analyzed": frames,
                "family": "fixed_diff",
                "residual_codec": o.residual_codec.name(),
                "complete_bytes": cost.complete_bytes,
                "anatomy": anatomy_json(&a),
            }));
            merge_into(&mut pop.merged, &a);
        }
    }
    Ok(pop)
}

/// Push the deterministic integers of one anatomy into the projection.
fn push_anatomy_integers(projection: &mut Vec<u8>, a: &Anatomy) {
    common::push_u64(projection, a.residual_samples);
    common::push_u64(projection, a.bits);
    common::push_u64(projection, a.max_abs);
    for i in 0..MAG_BUCKETS {
        common::push_u64(projection, a.mag_hist[i]);
    }
    for i in 0..CONTEXT_COUNT {
        common::push_u64(projection, a.contexts[i]);
    }
}

/// Aggregate JSON: the merged entropies, the per-context saving and the pooled
/// histogram, reported in bits per residual sample.
fn population_json(pop: &Population) -> serde_json::Value {
    let a = &pop.merged;
    serde_json::json!({
        "population": pop.name,
        "objects": pop.rows.len(),
        "all_exact": pop.all_exact,
        "residual_samples": a.residual_samples,
        "bits": a.bits,
        "order0_bits_per_residual": a.order0,
        "by_pos": a.by_pos,
        "by_prefix": a.by_prefix,
        "by_prev_mag": a.by_prev_mag,
        "by_prev2_mag": a.by_prev2_mag,
        "by_prev_pair": a.by_prev_pair,
        "by_prev_sign": a.by_prev_sign,
        "by_fsm": a.by_fsm,
        "by_lag": a.by_lag,
        "by_energy": a.by_energy,
        "by_disagree": a.by_disagree,
        "bits_per_residual_saved_vs_order0": savings_json(a),
        "zero_rate": a.zero_rate,
        "neg_rate": a.neg_rate,
        "mean_abs": a.mean_abs,
        "max_abs": a.max_abs,
        "contexts": CONTEXT_NAMES.iter().zip(a.contexts)
            .map(|(n, c)| ((*n).to_string(), serde_json::json!(c)))
            .collect::<serde_json::Map<_, _>>(),
        "mag_hist": a.mag_hist,
    })
}

/// Run the court; writes `receipts/learned-residual-anatomy/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.residual_anatomy.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );
    for name in CONTEXT_NAMES {
        common::push_label(&mut projection, name);
    }

    if !available() || !decoder_available() {
        return common::finish_exp3(
            "learned-residual-anatomy",
            receipts_root,
            LEARNED_RESIDUAL_ANATOMY_SHA256,
            &projection,
            Verdict::Inconclusive,
            "real corpus audio or the external `flac` decoder is absent".to_string(),
            vec![(
                "limitations",
                serde_json::json!([
                    "the effectiveness S8 population requires the uncommitted LibriSpeech audio \
                     bulk under corpus/real/ and the external `flac` decoder used only to read \
                     the frozen compressed clips; the held-out Mode-C split is deliberately not \
                     used by this diagnostic"
                ]),
            )],
        );
    }

    let mut budget = common::train_budget();
    budget.max_iterations = 128;
    let speech_pop = speech_population(&budget)?;
    let phase_m_pop = phase_m_population(&budget)?;

    for pop in [&speech_pop, &phase_m_pop] {
        for row in &pop.rows {
            common::push_label(&mut projection, row["id"].as_str().unwrap_or(""));
            common::push_label(&mut projection, row["family"].as_str().unwrap_or(""));
            common::push_u64(
                &mut projection,
                row["complete_bytes"].as_u64().unwrap_or(u64::MAX),
            );
        }
        common::push_label(&mut projection, pop.name);
        // Re-derive the merged integer totals deterministically by re-reading the
        // per-object anatomies is unnecessary: the merge is a pure function of
        // the rows, and `push_anatomy_integers` binds its exact integer state.
        push_anatomy_integers(&mut projection, &pop.merged);
    }

    let all_exact = speech_pop.all_exact && phase_m_pop.all_exact;
    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    let speech_json = population_json(&speech_pop);
    let phase_m_json = population_json(&phase_m_pop);
    common::finish_exp3(
        "learned-residual-anatomy",
        receipts_root,
        LEARNED_RESIDUAL_ANATOMY_SHA256,
        &projection,
        verdict,
        format!(
            "residual entropy anatomy of the exact S8 winner residual over {} effectiveness \
             clips and of {} representative Phase-M channel streams; conditional entropies in \
             bits per residual sample under {} decoder-visible contexts (held-out Mode C \
             untouched)",
            speech_pop.rows.len(),
            phase_m_pop.rows.len(),
            CONTEXT_COUNT
        ),
        vec![
            ("speech_effectiveness", speech_json),
            ("phase_m_representative", phase_m_json),
            ("speech_objects", serde_json::json!(speech_pop.rows)),
            ("phase_m_objects", serde_json::json!(phase_m_pop.rows)),
            (
                "method",
                serde_json::json!({
                    "binarization": "sign bit, then Exp-Golomb(0) of |r|: v=|r|+1, \
                                     (bit_length(v)-1) zero bits, then the bits of v MSB-first",
                    "estimator": "plug-in empirical conditional entropy in bits per residual \
                                  sample; context cardinalities are reported so sparse-context \
                                  optimism is visible",
                    "matched_lag": "oracle upper bound chosen per residual from its own \
                                    autocorrelation (1..=128); a coder would transmit it",
                    "energy_window": 64,
                    "disagreement": "selected predictor hypothesis minus the previous \
                                     reconstructed sample",
                    "speech_predictor": "the active S8 portfolio winner, unchanged",
                    "phase_m_predictor": "cheap exact fixed-difference closure over the frozen \
                                          block ladder, per deinterleaved channel",
                    "phase_m_sample_bound": MAX_ANALYSIS_SAMPLES,
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                    "profile": crate::learned::profile::LEARNED_EXP3_PROFILE,
                    "format_change": "none: this court only decodes existing residuals",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "a plug-in conditional entropy is an optimistic estimate when a context is \
                     sparse; the per-context cardinalities are reported so the reader can judge \
                     dilution",
                    "the matched-lag context is an oracle ceiling, not a decoder-free quantity",
                    "the Phase-M population uses a cheaper predictor than voice does; it exists \
                     only to test whether a context family carries information outside speech",
                    "this is a diagnostic, not a codec: no representation, profile or format is \
                     changed and no compression claim is made",
                ]),
            ),
        ],
    )
}

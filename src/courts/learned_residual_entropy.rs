//! `court learned-residual-entropy` — fourth-pass **Seal E1** attributable
//! residual-codec ladder.
//!
//! Seal E0 showed *where* the residual's information is; this court asks
//! whether a new coder can actually capture it. The predictor is frozen: for
//! every effectiveness clip the court takes the exact S8 winner residual (the
//! same portfolio `court learned-speech` runs) and re-encodes **that same dense
//! residual** with each codec, so every byte difference is attributable to the
//! entropy coder alone. The held-out Mode C split is untouched.
//!
//! The ladder reported per population:
//!
//! ```text
//! pre-E1 best   the Seal-S4 Exp3 family (ids 0..=15) on the fixed residual
//! signed/fsm    the Seal-E1 adaptive binary range coder (id 16)
//! with-E1 best  the minimum of the two (what the portfolio would select)
//! ```
//!
//! **Seal E1** adds [`ResidualCodecV2::SignedFsm`]: a forward, carry-less
//! binary range coder (the LZMA arithmetic coder) that models each residual bit
//! with an online-adaptive 12-bit probability selected by the residual-event FSM
//! state, the previous magnitude bucket, the previous sign, the unary-length
//! position and the already-decoded value prefix. A bad probability estimate
//! costs bits and never correctness, and both sides update the same slot after
//! every bit from information the decoder already has.
//!
//! A second population (representative Phase-M objects with a cheap exact
//! fixed-difference closure) checks whether the coder generalizes beyond speech.
//! Only deterministic integers enter the frozen projection; per-codec totals and
//! wall-clock timings are recorded as extras, never hashed.

use crate::corpus::{generate, specs};
use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::evidence::timing::Stopwatch;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::object::LearnedObject;
use crate::learned::residual_codec2::ResidualCodecV2;
use crate::learned::train::TrainBudget;
use crate::learned::train::fixed::fit_fixed_diff_sweep;
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_RESIDUAL_ENTROPY_SHA256: &str =
    "0c83a9bbe0def2e2e8e26760ca2644f02561bfb929c0392fb9aba8b6dab8462a";

/// Maximum representative Phase-M objects (one per distinct class tuple).
const MAX_PHASE_M_OBJECTS: usize = 12;

/// Bound the residual samples analysed per channel stream.
const MAX_ANALYSIS_SAMPLES: usize = 32_768;

/// The fixed-difference block-size ladder used for the Phase-M predictor.
const PHASE_M_LADDER: [Option<u32>; 5] = [None, Some(4096), Some(2048), Some(1024), Some(512)];

/// Encoding repetitions used for the (unhashed) timing extra.
const TIMING_REPS: u32 = 3;

/// One exact residual with its identity.
struct Case {
    id: String,
    population: &'static str,
    family: &'static str,
    class: serde_json::Value,
    residual: Vec<i32>,
}

/// The cheapest codec (and complete bytes) over an explicit codec subset.
fn best_over(codecs: &[ResidualCodecV2], residual: &[i32]) -> (ResidualCodecV2, u64) {
    let mut best = (codecs[0], u64::MAX);
    for &codec in codecs {
        let b = codec.encode(residual).len() as u64 + 1;
        if b < best.1 {
            best = (codec, b);
        }
    }
    best
}

/// The exact S8 winner residual per effectiveness clip.
fn speech_cases(budget: &TrainBudget) -> Result<Vec<Case>> {
    let scratch = PathBuf::from("target/real-corpus/scratch");
    let clips = load_cases(&effectiveness_clips(), speech::CLIPS_PER_SPLIT, &scratch)?;
    let mut out = Vec::new();
    for case in &clips {
        let cands = speech::portfolio(case, budget)?;
        let mut best: Option<(&str, &LearnedObject, u64)> = None;
        for (name, o) in &cands {
            if !o.verify(&case.samples) {
                continue;
            }
            let b = LearnedCost::of(o)?.complete_bytes;
            if best.as_ref().is_none_or(|(_, _, bb)| b < *bb) {
                best = Some((name, o, b));
            }
        }
        if let Some((family, o, _)) = best {
            out.push(Case {
                id: case.clip.id.clone(),
                population: "speech_effectiveness",
                family,
                class: serde_json::json!({ "split": case.clip.split }),
                residual: o.residual()?,
            });
        }
    }
    Ok(out)
}

/// Greedily choose one representative spec per distinct class tuple.
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

/// The representative Phase-M residuals (cheap exact fixed-difference closure,
/// per deinterleaved channel).
fn phase_m_cases(budget: &TrainBudget) -> Result<Vec<Case>> {
    let all = specs::specs();
    let mut out = Vec::new();
    for spec in representative_specs(&all) {
        let samples = generate::generate(spec)?;
        let per_frame = usize::from(spec.channels).max(1);
        for ch in 0..per_frame {
            let mut chan: Vec<i32> = samples[ch..].iter().step_by(per_frame).copied().collect();
            chan.truncate(chan.len().min(MAX_ANALYSIS_SAMPLES));
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
                continue;
            }
            out.push(Case {
                id: format!("{}#ch{ch}", spec.id),
                population: "phase_m_representative",
                family: "fixed_diff",
                class: serde_json::json!({
                    "source_structure": spec.source_structure.as_str(),
                    "entropy": spec.entropy.as_str(),
                    "temporal": spec.temporal.as_str(),
                }),
                residual: o.residual()?,
            });
        }
    }
    Ok(out)
}

/// Encode-level ladder for one population.
///
/// The general-Golomb codecs (`Golomb`, `CenteredGolomb`) search every divisor
/// up to `MAX_GOLOMB_M` per block, which is unbounded work on the full-scale
/// full_i32 noise residuals of the representative Phase-M population. They are
/// still measured where they matter (the speech population, whose residuals are
/// small) and in every other Exp3 court; here the Phase-M ladder uses the
/// bounded-search encoders so the diagnostic stays cheap and deterministic.
fn phase_m_codecs() -> Vec<ResidualCodecV2> {
    ResidualCodecV2::ALL_V3
        .iter()
        .copied()
        .filter(|c| !matches!(c, ResidualCodecV2::Golomb | ResidualCodecV2::CenteredGolomb))
        .collect()
}

/// The fourth-pass E-ladder codecs measured against the pre-existing family.
const E_CODECS: [ResidualCodecV2; 8] = [
    ResidualCodecV2::SignedFsm,
    ResidualCodecV2::SignedFsmSse,
    ResidualCodecV2::SignedFsmLag,
    ResidualCodecV2::SignedFsmMix,
    ResidualCodecV2::SignedFsmRcm,
    ResidualCodecV2::Bgmc,
    ResidualCodecV2::Ctw,
    ResidualCodecV2::BgmcSse,
];

/// Evaluate the ladder over one population with an explicit codec set.
fn evaluate(
    population: &'static str,
    cases: &[Case],
    codecs: &[ResidualCodecV2],
) -> Result<(serde_json::Value, bool, Vec<u8>)> {
    let pre: Vec<ResidualCodecV2> = codecs
        .iter()
        .copied()
        .filter(|c| !E_CODECS.contains(c))
        .collect();
    let active: Vec<ResidualCodecV2> = E_CODECS
        .iter()
        .copied()
        .filter(|c| codecs.contains(c))
        .collect();
    let mut rows = Vec::with_capacity(cases.len());
    let mut projection = Vec::new();
    let mut exact = true;
    let mut pre_total = 0u64;
    let mut with_total = 0u64;
    let mut e_totals: Vec<u64> = vec![0; active.len()];
    let mut codec_totals = vec![0u64; codecs.len()];
    let mut concat: Vec<i32> = Vec::new();

    for case in cases {
        let (pre_codec, pre_b) = best_over(&pre, &case.residual);
        let mut ladder = serde_json::Map::new();
        let mut e_bytes = Vec::with_capacity(active.len());
        for (k, &codec) in active.iter().enumerate() {
            let payload = codec.encode(&case.residual);
            match codec.decode(&payload, case.residual.len()) {
                Ok(back) => exact &= back == case.residual,
                Err(_) => exact = false,
            }
            let b = payload.len() as u64 + 1;
            e_totals[k] += b;
            e_bytes.push(b);
            ladder.insert(codec.name().to_string(), serde_json::json!(b));
        }
        let (with_codec, with_b) = best_over(codecs, &case.residual);
        for (i, codec) in codecs.iter().enumerate() {
            codec_totals[i] += codec.encode(&case.residual).len() as u64 + 1;
        }
        pre_total += pre_b;
        with_total += with_b;
        concat.extend_from_slice(&case.residual);

        common::push_label(&mut projection, &case.id);
        common::push_u64(&mut projection, pre_b);
        for &b in &e_bytes {
            common::push_u64(&mut projection, b);
        }
        common::push_u64(&mut projection, with_b);

        rows.push(serde_json::json!({
            "id": case.id,
            "population": case.population,
            "family": case.family,
            "class": case.class,
            "samples": case.residual.len(),
            "pre_codec": pre_codec.name(),
            "pre_bytes": pre_b,
            "e_ladder_bytes": serde_json::Value::Object(ladder),
            "with_codec": with_codec.name(),
            "with_bytes": with_b,
            "ladder_gain_bytes": pre_b as i64 - with_b as i64,
        }));
    }

    // (Unhashed) per-sample wall-clock timings on the concatenated residual.
    let samples = concat.len() as u64;
    let mut timings = serde_json::Map::new();
    for &codec in &active {
        if concat.is_empty() {
            continue;
        }
        let sw = Stopwatch::start();
        for _ in 0..TIMING_REPS {
            let _ = codec.encode(&concat);
        }
        let enc_ns = sw.elapsed_ns().max(0) as u64 / u64::from(TIMING_REPS);
        let payload = codec.encode(&concat);
        let sw = Stopwatch::start();
        for _ in 0..TIMING_REPS {
            let _ = codec.decode(&payload, concat.len());
        }
        let dec_ns = sw.elapsed_ns().max(0) as u64 / u64::from(TIMING_REPS);
        timings.insert(
            codec.name().to_string(),
            serde_json::json!({
                "encode_ns_per_sample": enc_ns as f64 / samples as f64,
                "decode_ns_per_sample": dec_ns as f64 / samples as f64,
            }),
        );
    }

    let codec_map: serde_json::Map<String, serde_json::Value> = codecs
        .iter()
        .zip(&codec_totals)
        .map(|(c, b)| (c.name().to_string(), serde_json::json!(b)))
        .collect();
    let mut e_map = serde_json::Map::new();
    for (k, &codec) in active.iter().enumerate() {
        e_map.insert(codec.name().to_string(), serde_json::json!(e_totals[k]));
    }
    let summary = serde_json::json!({
        "population": population,
        "objects": cases.len(),
        "residual_samples": samples,
        "all_exact": exact,
        "active_codecs": codecs.iter().map(|c| c.name()).collect::<Vec<_>>(),
        "e_ladder": active.iter().map(|c| c.name()).collect::<Vec<_>>(),
        "pre_e_best_bytes": pre_total,
        "e_ladder_bytes": serde_json::Value::Object(e_map),
        "with_e_best_bytes": with_total,
        "ladder_gain_bytes": pre_total as i64 - with_total as i64,
        "codec_totals_bytes": serde_json::Value::Object(codec_map),
        "timing": {
            "per_codec": serde_json::Value::Object(timings),
            "repetitions": TIMING_REPS,
            "note": "wall-clock, never hashed",
        },
        "objects_detail": rows,
    });
    Ok((summary, exact, projection))
}

/// Run the court; writes `receipts/learned-residual-entropy/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.residual_entropy.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );
    for codec in ResidualCodecV2::ALL_V3 {
        common::push_label(&mut projection, codec.name());
    }

    if !available() || !decoder_available() {
        return common::finish_exp3(
            "learned-residual-entropy",
            receipts_root,
            LEARNED_RESIDUAL_ENTROPY_SHA256,
            &projection,
            Verdict::Inconclusive,
            "real corpus audio or the external `flac` decoder is absent".to_string(),
            vec![(
                "limitations",
                serde_json::json!([
                    "the effectiveness S8 population requires the uncommitted LibriSpeech audio \
                     bulk under corpus/real/ and the external `flac` decoder; the held-out Mode-C \
                     split is deliberately not used by this court"
                ]),
            )],
        );
    }

    let mut budget = common::train_budget();
    budget.max_iterations = 128;
    let speech_cases = speech_cases(&budget)?;
    let phase_m = phase_m_cases(&budget)?;
    let (speech_json, speech_exact, speech_projection) = evaluate(
        "speech_effectiveness",
        &speech_cases,
        &ResidualCodecV2::ALL_V3,
    )?;
    let phase_codecs = phase_m_codecs();
    let (phase_json, phase_exact, phase_projection) =
        evaluate("phase_m_representative", &phase_m, &phase_codecs)?;
    projection.extend_from_slice(&speech_projection);
    projection.extend_from_slice(&phase_projection);
    common::push_u64(&mut projection, speech_cases.len() as u64);
    common::push_u64(&mut projection, phase_m.len() as u64);

    let all_exact = speech_exact && phase_exact;
    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_exp3(
        "learned-residual-entropy",
        receipts_root,
        LEARNED_RESIDUAL_ENTROPY_SHA256,
        &projection,
        verdict,
        format!(
            "attributable residual-codec ladder over the fixed S8 winner residual ({} effectiveness \
             clips) and {} representative Phase-M channel streams: pre-E family vs the Seal E1 \
             signed/FSM adaptive range coder (id 16), the Seal E2 SSE/APM-corrected variant \
             (id 17) and the Seal E3 matched-lag variant (id 18), with the selected minimum \
             reported",
            speech_cases.len(),
            phase_m.len()
        ),
        vec![
            ("speech_effectiveness", speech_json),
            ("phase_m_representative", phase_json),
            (
                "method",
                serde_json::json!({
                    "fixed_predictor": "the S8 portfolio winner per effectiveness clip and a cheap \
                                        exact fixed-difference closure per Phase-M channel; the \
                                        dense residual is identical for every codec",
                    "ladder": [
                        "pre_e (ids 0..=15)",
                        "signed_fsm (id 16)",
                        "signed_fsm_sse (id 17)",
                        "signed_fsm_lag (id 18)",
                        "signed_fsm_mix (id 19)",
                        "signed_fsm_rcm (id 20)",
                        "bgmc (id 21)",
                        "ctw (id 22)",
                        "bgmc_sse (id 23)",
                        "with_e minimum",
                    ],
                    "e1_codec": "forward carry-less binary range coder (LZMA arithmetic coder) with \
                                 online 12-bit adaptive probabilities over the residual-event FSM \
                                 state, previous magnitude bucket, previous sign, unary-length \
                                 position and decoded value prefix",
                    "e2_codec": "the same base coder with an SSE/APM stage: the base 12-bit \
                                 probability is quantized in the integer stretch domain into 33 \
                                 interpolation points and adaptively corrected under a \
                                 phase × magnitude-bucket context, then interpolated",
                    "e3_codec": "the SSE/APM coder with a matched-lag context: the encoder stores a \
                                 decoder-visible lag (0..=128, or 0 for none) chosen from the \
                                 residual's own autocorrelation, and the lagged residual magnitude \
                                 bucket widens the SSE context; the lag never alters a value",
                    "e4_codec": "a PAQ-light residual context mixer: six context experts (position, \
                                 previous magnitude, previous sign, FSM state, prefix×lag, \
                                 magnitude×lag) are combined by an integer logistic mixer whose \
                                 weights are trained online to minimize -log P(bit), then the \
                                 mixed probability drives the same range coder",
                    "e5_codec": "full RCM research mode: the E4 mixture is followed by an ISSE \
                                 stage whose expanded context (phase × token position × previous \
                                 magnitude × FSM state) selects a 4+4 bit-history state that \
                                 indexes an adaptive probability map correcting the mixed \
                                 probability",
                    "binarization": "sign bit, then Exp-Golomb(0) of |r| (identical to Seal E0)",
                    "profile": crate::learned::profile::LEARNED_EXP3_PROFILE,
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "attribution is codec-only because the dense residual is held fixed; the \
                     portfolio-wide effect appears in the `learned-speech` and `learned-u1-wasted` \
                     courts",
                    "the Phase-M population uses a cheaper predictor than voice does and exists only \
                     to test whether the coder generalizes beyond speech",
                    "wall-clock per-sample timings are recorded as extras and are never hashed",
                ]),
            ),
        ],
    )
}

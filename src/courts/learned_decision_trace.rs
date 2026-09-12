//! `court learned-decision-trace` — Phase 6 mechanism 6 (`DecisionTraceRans`).
//!
//! `DecisionTrace` (residual codec id 26) keeps the `EmaRans` envelope — a
//! 17-symbol high-part alphabet with raw sign, raw low bits and an Exp-Golomb
//! escape — but makes the adaptive categorical model **conditioned**. Each
//! context owns its own integer EMA CDF, and the context is a deterministic
//! function of the high-part symbols already decoded. Like `EmaRans` it
//! transmits no probability table, but unlike `EmaRans` it is not order-0: the
//! decoder chooses the very same distribution from its own reconstructed
//! history.
//!
//! The codec is built on the `learned::decision_trace` primitive (forward
//! record, backward emit), which exists because rANS is LIFO and a
//! forward-adaptive context model cannot be priced on a single backward walk.
//!
//! This court isolates the mechanism in three ways:
//!
//! * it measures `DecisionTrace` beside `EmaRans` (the order-0 categorical
//!   baseline), `ContextRans` (the static transmitted-table rANS baseline) and
//!   `Reblock` (the Phase-6 partitioning codec), and beside the best
//!   pre-existing codec overall;
//! * it reports an **encoder-only context-function ablation** — the same
//!   envelope and model class under contexts of increasing richness (`order0`,
//!   `prev_band4` (shipped), `prev_band8`, `band4_prev2zero`, `band4_band4`,
//!   `prev_full`, `prev_full_prev2zero`, `prev2_full`) — so the choice of context
//!   is itself evidence, not a claim;
//! * every payload must round-trip exactly.
//!
//! Held-out Mode C is deliberately untouched.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::residual_codec2::{ResidualCodecV2, decision_trace_context_ablation};
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_DECISION_TRACE_SHA256: &str =
    "f1366fc403173de25626db49f19f1e5f3efa1e5aefe929d76b3d27441ab5fdd2";

fn payload_bytes(codec: ResidualCodecV2, residual: &[i32]) -> u64 {
    codec.encode(residual).len() as u64 + 1
}

/// Best complete bytes over the pre-existing family, excluding `DecisionTrace`.
fn best_existing(residual: &[i32]) -> (ResidualCodecV2, u64) {
    let mut best = (ResidualCodecV2::DenseI32, u64::MAX);
    for codec in ResidualCodecV2::ALL_V3 {
        if codec == ResidualCodecV2::DecisionTrace {
            continue;
        }
        let b = payload_bytes(codec, residual);
        if b < best.1 {
            best = (codec, b);
        }
    }
    best
}

/// Deterministic fixtures that expose conditional structure the order-0 EMA
/// model cannot see, plus two magnitude-skewed fixtures shared with the
/// `EmaRans` court so the comparison is like-for-like.
fn synthetic_fixtures() -> Vec<(&'static str, Vec<i32>)> {
    // March down magnitudes, then reverse back up: the next high part is a
    // deterministic function of the previous one while the marginal is broad,
    // so order-1 contexts are near-free but order-0 pays the full marginal
    // entropy.
    let march = {
        let mags = [1i32, 2, 5, 13, 40, 121, 40, 13, 5, 2];
        let mut v = Vec::with_capacity(8192);
        for i in 0..8192i32 {
            let m = mags[(i as usize) % mags.len()];
            v.push(if (i as usize).is_multiple_of(2) {
                m
            } else {
                -m
            });
        }
        v
    };
    // A deterministic bijection over six magnitudes: given the previous symbol
    // the next is certain, but the stationary marginal is uniform.
    let permutation = {
        let perm = [3usize, 0, 5, 2, 4, 1];
        let mags = [1i32, 2, 3, 5, 8, 13];
        let mut state = 0usize;
        let mut v = Vec::with_capacity(8192);
        for i in 0..8192usize {
            state = perm[state];
            let m = mags[state];
            v.push(if i.is_multiple_of(2) { m } else { -m });
        }
        v
    };
    // Alternating zero runs of cycling length separated by isolated nonzero
    // samples: the previous symbol decides whether a run continues.
    let zero_runs = {
        let mut v: Vec<i32> = Vec::with_capacity(8192);
        let mut i = 0i32;
        let mut run = 1usize;
        while v.len() < 8192 {
            v.resize(v.len() + run, 0);
            let m = 1 + (i % 5);
            v.push(if (i as usize).is_multiple_of(2) {
                m
            } else {
                -m
            });
            i += 1;
            run = run % 4 + 1;
        }
        v.truncate(8192);
        v
    };
    // Magnitude-skewed fixtures carried over from the `EmaRans` court.
    let laplace_mix = (0..8192i64)
        .map(|i| {
            let v = (i * 2654435761) % 100003 - 50001;
            if i % 7 == 0 { (v / 8) as i32 } else { v as i32 }
        })
        .collect();
    let quiet_loud = (0..6144i64)
        .map(|i| {
            if i < 3072 {
                (i % 3) as i32 - 1
            } else {
                (i * 991) as i32
            }
        })
        .collect();
    vec![
        ("march_highpart", march),
        ("permutation_highpart", permutation),
        ("zero_run_structure", zero_runs),
        ("laplace_mix", laplace_mix),
        ("quiet_loud", quiet_loud),
    ]
}

struct Row {
    id: String,
    population: &'static str,
    samples: usize,
    decision_bytes: u64,
    ema_bytes: u64,
    context_rans_bytes: u64,
    reblock_bytes: u64,
    best_existing_codec: &'static str,
    best_existing_bytes: u64,
    ablation: Vec<(&'static str, u64)>,
    exact: bool,
}

fn analyse(id: String, population: &'static str, residual: &[i32]) -> Row {
    let payload = ResidualCodecV2::DecisionTrace.encode(residual);
    let exact = ResidualCodecV2::DecisionTrace
        .decode(&payload, residual.len())
        .map(|back| back == residual)
        .unwrap_or(false);
    let (codec, best_bytes) = best_existing(residual);
    Row {
        id,
        population,
        samples: residual.len(),
        decision_bytes: payload.len() as u64 + 1,
        ema_bytes: payload_bytes(ResidualCodecV2::EmaRans, residual),
        context_rans_bytes: payload_bytes(ResidualCodecV2::ContextRans, residual),
        reblock_bytes: payload_bytes(ResidualCodecV2::Reblock, residual),
        best_existing_codec: codec.name(),
        best_existing_bytes: best_bytes,
        ablation: decision_trace_context_ablation(residual),
        exact,
    }
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.exact;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.decision_bytes);
    common::push_u64(projection, row.best_existing_bytes);
    common::push_u64(projection, row.ema_bytes);
    common::push_u64(projection, row.context_rans_bytes);
    for &(label, bytes) in &row.ablation {
        common::push_label(projection, label);
        common::push_u64(projection, bytes);
    }
}

fn row_json(row: &Row) -> serde_json::Value {
    let ablation: serde_json::Value = row
        .ablation
        .iter()
        .map(|&(label, bytes)| serde_json::json!({ "context": label, "bytes": bytes }))
        .collect();
    serde_json::json!({
        "id": row.id,
        "population": row.population,
        "samples": row.samples,
        "decision_trace_bytes": row.decision_bytes,
        "ema_rans_bytes": row.ema_bytes,
        "context_rans_bytes": row.context_rans_bytes,
        "reblock_bytes": row.reblock_bytes,
        "best_existing_codec": row.best_existing_codec,
        "best_existing_bytes": row.best_existing_bytes,
        "gain_vs_best_existing_bytes": row.best_existing_bytes as i64 - row.decision_bytes as i64,
        "gain_vs_ema_rans_bytes": row.ema_bytes as i64 - row.decision_bytes as i64,
        "gain_vs_context_rans_bytes": row.context_rans_bytes as i64 - row.decision_bytes as i64,
        "context_ablation": ablation,
        "exact": row.exact,
    })
}

/// Run the court; writes `receipts/learned-decision-trace/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.decision_trace.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    let mut all_exact = true;
    let mut synthetic = Vec::new();
    for (name, residual) in synthetic_fixtures() {
        let row = analyse(name.to_string(), "synthetic_conditional", &residual);
        push_row(&mut projection, &row, &mut all_exact);
        synthetic.push(row_json(&row));
    }

    let mut real = Vec::new();
    let mut real_present = false;
    if available() && decoder_available() {
        real_present = true;
        let mut budget = common::train_budget();
        budget.max_iterations = 128;
        let scratch = PathBuf::from("target/real-corpus/scratch");
        let clips = load_cases(&effectiveness_clips(), speech::CLIPS_PER_SPLIT, &scratch)?;
        for case in &clips {
            let cands = speech::portfolio(case, &budget)?;
            let mut best: Option<u64> = None;
            let mut residual: Vec<i32> = Vec::new();
            for (_, o) in &cands {
                if !o.verify(&case.samples) {
                    continue;
                }
                let b = LearnedCost::of(o)?.complete_bytes;
                if best.is_none_or(|bb| b < bb) {
                    best = Some(b);
                    residual = o.residual()?;
                }
            }
            if best.is_some() {
                let row = analyse(case.clip.id.clone(), "speech_effectiveness", &residual);
                push_row(&mut projection, &row, &mut all_exact);
                real.push(row_json(&row));
            }
        }
    }

    let count_wins = |rows: &[serde_json::Value], key: &str| -> u64 {
        rows.iter()
            .filter(|r| r[key].as_i64().is_some_and(|v| v > 0))
            .count() as u64
    };
    let syn_wins = count_wins(&synthetic, "gain_vs_best_existing_bytes");
    let real_wins = count_wins(&real, "gain_vs_best_existing_bytes");
    let syn_wins_vs_ema = count_wins(&synthetic, "gain_vs_ema_rans_bytes");
    let real_wins_vs_ema = count_wins(&real, "gain_vs_ema_rans_bytes");
    let real_wins_vs_ctx = count_wins(&real, "gain_vs_context_rans_bytes");

    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else if !real_present {
        Verdict::Inconclusive
    } else {
        Verdict::Supported
    };

    common::finish_exp3(
        "learned-decision-trace",
        receipts_root,
        LEARNED_DECISION_TRACE_SHA256,
        &projection,
        verdict,
        format!(
            "context-conditioned forward-adaptive categorical rANS (codec id 26, 17-symbol \
             alphabet, {nctx} learned contexts) over {ns} synthetic + {nr} real cases: beats the \
             best pre-existing codec on {sw}/{ns} synthetic and {rw}/{nr} real; beats the order-0 \
             EmaRans on {swe}/{ns} synthetic and {we}/{nr} real, and the static ContextRans on \
             {wc}/{nr} real; encoder-only context-function ablation reported per case",
            nctx = 4,
            ns = synthetic.len(),
            nr = real.len(),
            sw = syn_wins,
            rw = real_wins,
            swe = syn_wins_vs_ema,
            we = real_wins_vs_ema,
            wc = real_wins_vs_ctx,
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "synthetic_cases": synthetic.len(),
                    "real_cases": real.len(),
                }),
            ),
            ("synthetic", serde_json::json!(synthetic)),
            ("real_effectiveness", serde_json::json!(real)),
            (
                "method",
                serde_json::json!({
                    "alphabet": "17 symbols: 0 = high part zero, 1..=15 = high part h, 16 = escape \
                                 (Exp-Golomb high part); sign and low bits are raw",
                    "model": "one integer EMA categorical CDF per decoder-visible context, \
                              CDF[s] += (Target(s) - CDF[s])/2^k, updated after every symbol; no \
                              table is transmitted",
                    "shipped_context": "context_of(previous high part): four magnitude bands, \
                                        chosen from this court's ablation as the context variant \
                                        with the smallest complete cost on the frozen \
                                        effectiveness set",
                    "trace": "rANS is LIFO, so the encoder runs the model forward to record each \
                              symbol's interval and emits the symbols backward through the \
                              `learned::decision_trace` primitive; the decoder recomputes the \
                              identical context model forward; work is chunked so the trace is \
                              bounded",
                    "ablation": "encoder-only: the same envelope and model class under contexts \
                                 order0, prev_band4 (shipped), prev_band8, band4_prev2zero, \
                                 band4_band4, prev_full, prev_full_prev2zero and prev2_full",
                    "contrast": "EmaRans (id 25) is the order-0 member of this family; ContextRans \
                                 (id 11) transmits static tables instead of adapting",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the adaptation rate, alphabet size and shipped context are fixed a priori \
                     for this seal; tuning them on held-out data is not permitted",
                    "the escape and low bits pay raw bits, so the model's advantage is confined \
                     to the coarse high part",
                    "the ablation variants are encoder-only analyses and are never decoded; only \
                     the shipped context is a bitstream",
                ]),
            ),
        ],
    )
}

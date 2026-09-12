//! `court learned-ema-rans` — Phase 6 mechanism 5 (`EmaRans17`).
//!
//! `EmaRans17` (residual codec id 25) is a forward-adaptive **categorical** rANS
//! coder over a 17-symbol alphabet of coarse high-part classes, with the sign and
//! the low bits sent raw. The categorical model is an integer EMA over the CDF,
//! `CDF[s] += (Target(s) - CDF[s])/2^k`, updated after every symbol, and **no
//! histogram is transmitted** — the contrast with `ContextRans` (id 11), which
//! scans the residual, builds static tables and serializes them.
//!
//! This court isolates the mechanism: per case it reports the `EmaRans` size
//! beside the best pre-existing codec and beside `ContextRans` (the static
//! rANS baseline) and `Reblock` (the Phase-6 partitioning codec). Every payload
//! must round-trip exactly. Held-out Mode C is deliberately untouched.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::residual_codec2::ResidualCodecV2;
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_EMA_RANS_SHA256: &str =
    "1dfba6d4334eef579800c85c6d9daf4a9449e777af0415cfb376fbd3f1ac3093";

fn payload_bytes(codec: ResidualCodecV2, residual: &[i32]) -> u64 {
    codec.encode(residual).len() as u64 + 1
}

/// Best complete bytes over the pre-existing family, excluding `EmaRans`.
fn best_existing(residual: &[i32]) -> (ResidualCodecV2, u64) {
    let mut best = (ResidualCodecV2::DenseI32, u64::MAX);
    for codec in ResidualCodecV2::ALL_V3 {
        if codec == ResidualCodecV2::EmaRans {
            continue;
        }
        let b = payload_bytes(codec, residual);
        if b < best.1 {
            best = (codec, b);
        }
    }
    best
}

/// Deterministic magnitude-skewed fixtures: the shape an adaptive categorical
/// model targets.
fn synthetic_fixtures() -> Vec<(&'static str, Vec<i32>)> {
    vec![
        (
            "laplace_mix",
            (0..8192i64)
                .map(|i| {
                    // Heavier tails than a fixed model assumes.
                    let v = (i * 2654435761) % 100003 - 50001;
                    let scaled = if i % 7 == 0 { v / 8 } else { v };
                    scaled as i32
                })
                .collect(),
        ),
        (
            "quiet_loud",
            (0..6144i64)
                .map(|i| {
                    if i < 3072 {
                        (i % 3) as i32 - 1
                    } else {
                        (i * 991) as i32
                    }
                })
                .collect(),
        ),
        (
            "zero_heavy",
            (0..6144i64)
                .map(|i| if i % 5 == 0 { 0 } else { (i % 17) as i32 - 8 })
                .collect(),
        ),
    ]
}

struct Row {
    id: String,
    population: &'static str,
    samples: usize,
    ema_bytes: u64,
    context_rans_bytes: u64,
    reblock_bytes: u64,
    best_existing_codec: &'static str,
    best_existing_bytes: u64,
    exact: bool,
}

fn analyse(id: String, population: &'static str, residual: &[i32]) -> Row {
    let payload = ResidualCodecV2::EmaRans.encode(residual);
    let exact = ResidualCodecV2::EmaRans
        .decode(&payload, residual.len())
        .map(|back| back == residual)
        .unwrap_or(false);
    let (codec, best_bytes) = best_existing(residual);
    Row {
        id,
        population,
        samples: residual.len(),
        ema_bytes: payload.len() as u64 + 1,
        context_rans_bytes: payload_bytes(ResidualCodecV2::ContextRans, residual),
        reblock_bytes: payload_bytes(ResidualCodecV2::Reblock, residual),
        best_existing_codec: codec.name(),
        best_existing_bytes: best_bytes,
        exact,
    }
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.exact;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.ema_bytes);
    common::push_u64(projection, row.best_existing_bytes);
    common::push_u64(projection, row.context_rans_bytes);
}

fn row_json(row: &Row) -> serde_json::Value {
    serde_json::json!({
        "id": row.id,
        "population": row.population,
        "samples": row.samples,
        "ema_rans_bytes": row.ema_bytes,
        "context_rans_bytes": row.context_rans_bytes,
        "reblock_bytes": row.reblock_bytes,
        "best_existing_codec": row.best_existing_codec,
        "best_existing_bytes": row.best_existing_bytes,
        "gain_vs_best_existing_bytes": row.best_existing_bytes as i64 - row.ema_bytes as i64,
        "gain_vs_context_rans_bytes": row.context_rans_bytes as i64 - row.ema_bytes as i64,
        "exact": row.exact,
    })
}

/// Run the court; writes `receipts/learned-ema-rans/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.ema_rans.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    let mut all_exact = true;
    let mut synthetic = Vec::new();
    for (name, residual) in synthetic_fixtures() {
        let row = analyse(name.to_string(), "synthetic_magnitude_skewed", &residual);
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
    let real_wins_vs_ctx = count_wins(&real, "gain_vs_context_rans_bytes");

    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else if !real_present {
        Verdict::Inconclusive
    } else {
        Verdict::Supported
    };

    common::finish_exp3(
        "learned-ema-rans",
        receipts_root,
        LEARNED_EMA_RANS_SHA256,
        &projection,
        verdict,
        format!(
            "forward-adaptive categorical EMA rANS (codec id 25, 17-symbol alphabet) over {} \
             synthetic + {} real cases: beats the best pre-existing codec on {syn_wins}/{} \
             synthetic and {real_wins}/{} real, and the static ContextRans on {real_wins_vs_ctx}/{} \
             real",
            synthetic.len(),
            real.len(),
            synthetic.len(),
            real.len(),
            real.len(),
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
                    "model": "integer EMA over the CDF, CDF[s] += (Target(s) - CDF[s])/2^k, \
                              updated after every symbol; no histogram is transmitted",
                    "alphabet": "17 symbols: 0 = high part zero, 1..=15 = high part h, 16 = escape \
                                 (Exp-Golomb high part); sign and low bits are raw",
                    "trace": "rANS is LIFO, so the encoder runs the model forward to record each \
                              symbol's interval and emits the symbols backward; the decoder \
                              recomputes the identical model forward; work is chunked so the \
                              trace is bounded",
                    "contrast": "ContextRans (id 11) scans the residual, builds static tables and \
                                 serializes them; EmaRans adapts online and transmits nothing",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the adaptation rate and alphabet size are fixed a priori for this seal; \
                     tuning them on held-out data is not permitted",
                    "the escape and low bits pay raw bits, so the model's advantage is confined \
                     to the coarse high part",
                ]),
            ),
        ],
    )
}

//! `court learned-valuation-split` — Phase 6 mechanism 7 (`ValuationSplit`).
//!
//! `ValuationSplit` (residual codec id 27) factors every nonzero magnitude as
//! `|r| = odd * 2^e` and codes the four exact streams separately: the gaps
//! between nonzeros (or nothing at all when the residual is dense), the sign of
//! each nonzero, the per-symbol valuation `e`, and the odd core `(odd - 1) / 2`.
//! `FactorShift` (id 14) strips a single shared power of two from the whole
//! block, which is wasted whenever one odd sample is present.
//!
//! This court isolates the mechanism against `FactorShift` (the shared-factor
//! baseline), `ZeroMaskRice` (the sparse/magnitude baseline) and the best
//! pre-existing codec, over integer-scaled synthetic fixtures and the real
//! speech effectiveness residuals. Every payload must round-trip exactly.
//! Held-out Mode C is deliberately untouched.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::residual_codec2::ResidualCodecV2;
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_VALUATION_SPLIT_SHA256: &str =
    "a155ed76b43197aaa55bb32ebfd2c654a1aeb82263cb87bdad609b890c5cc64e";

fn payload_bytes(codec: ResidualCodecV2, residual: &[i32]) -> u64 {
    codec.encode(residual).len() as u64 + 1
}

/// Best complete bytes over the pre-existing family, excluding `ValuationSplit`.
fn best_existing(residual: &[i32]) -> (ResidualCodecV2, u64) {
    let mut best = (ResidualCodecV2::DenseI32, u64::MAX);
    for codec in ResidualCodecV2::ALL_V3 {
        if codec == ResidualCodecV2::ValuationSplit {
            continue;
        }
        let b = payload_bytes(codec, residual);
        if b < best.1 {
            best = (codec, b);
        }
    }
    best
}

/// Deterministic fixtures whose residuals carry per-symbol powers of two.
fn synthetic_fixtures() -> Vec<(&'static str, Vec<i32>)> {
    // Mixed valuations and a tiny odd alphabet: `FactorShift` can strip only the
    // minimum exponent, leaving the rest mixed.
    let integer_scaled = (0..8192i32)
        .map(|i| {
            let e = (i % 6) as u32 + 1;
            let odd = 2 * ((i / 6) % 4) + 1;
            let m = odd << e;
            if (i as usize).is_multiple_of(2) {
                m
            } else {
                -m
            }
        })
        .collect();
    // Pure powers of two: every sample is one odd core (1) times an exponent.
    let powers_of_two = (0..8192i32)
        .map(|i| {
            let e = (i % 20) as u32;
            let m = 1i32 << e;
            if (i as usize).is_multiple_of(2) {
                m
            } else {
                -m
            }
        })
        .collect();
    // A single odd sample pins `FactorShift`'s shared shift at zero.
    let odd_anchor = (0..8192i32)
        .map(|i| {
            if i == 4096 {
                1
            } else {
                let e = (i % 12) as u32 + 3;
                let m = (2 * (i % 3) + 1) << e;
                if (i as usize).is_multiple_of(2) {
                    m
                } else {
                    -m
                }
            }
        })
        .collect();
    // Sparse scaled nonzeros separated by silence.
    let sparse_scaled = (0..8192i32)
        .map(|i| {
            if i % 5 == 0 {
                let e = (i % 9) as u32;
                let m = (2 * (i % 4) + 1) << e;
                if (i as usize).is_multiple_of(2) {
                    m
                } else {
                    -m
                }
            } else {
                0
            }
        })
        .collect();
    // A natural-ish magnitude-skewed fixture, carried over for comparability.
    let laplace_mix: Vec<i32> = (0..8192i64)
        .map(|i| {
            let v = (i * 2654435761) % 100003 - 50001;
            if i % 7 == 0 { (v / 8) as i32 } else { v as i32 }
        })
        .collect();
    vec![
        ("integer_scaled", integer_scaled),
        ("powers_of_two", powers_of_two),
        ("odd_anchor", odd_anchor),
        ("sparse_scaled", sparse_scaled),
        ("laplace_mix", laplace_mix),
    ]
}

struct Row {
    id: String,
    population: &'static str,
    samples: usize,
    split_bytes: u64,
    factor_shift_bytes: u64,
    zero_mask_bytes: u64,
    best_existing_codec: &'static str,
    best_existing_bytes: u64,
    exact: bool,
}

fn analyse(id: String, population: &'static str, residual: &[i32]) -> Row {
    let payload = ResidualCodecV2::ValuationSplit.encode(residual);
    let exact = ResidualCodecV2::ValuationSplit
        .decode(&payload, residual.len())
        .map(|back| back == residual)
        .unwrap_or(false);
    let (codec, best_bytes) = best_existing(residual);
    Row {
        id,
        population,
        samples: residual.len(),
        split_bytes: payload.len() as u64 + 1,
        factor_shift_bytes: payload_bytes(ResidualCodecV2::FactorShift, residual),
        zero_mask_bytes: payload_bytes(ResidualCodecV2::ZeroMaskRice, residual),
        best_existing_codec: codec.name(),
        best_existing_bytes: best_bytes,
        exact,
    }
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.exact;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.split_bytes);
    common::push_u64(projection, row.factor_shift_bytes);
    common::push_u64(projection, row.zero_mask_bytes);
    common::push_u64(projection, row.best_existing_bytes);
}

fn row_json(row: &Row) -> serde_json::Value {
    serde_json::json!({
        "id": row.id,
        "population": row.population,
        "samples": row.samples,
        "valuation_split_bytes": row.split_bytes,
        "factor_shift_bytes": row.factor_shift_bytes,
        "zero_mask_rice_bytes": row.zero_mask_bytes,
        "best_existing_codec": row.best_existing_codec,
        "best_existing_bytes": row.best_existing_bytes,
        "gain_vs_best_existing_bytes": row.best_existing_bytes as i64 - row.split_bytes as i64,
        "gain_vs_factor_shift_bytes": row.factor_shift_bytes as i64 - row.split_bytes as i64,
        "gain_vs_zero_mask_rice_bytes": row.zero_mask_bytes as i64 - row.split_bytes as i64,
        "exact": row.exact,
    })
}

/// Run the court; writes `receipts/learned-valuation-split/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.valuation_split.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    let mut all_exact = true;
    let mut synthetic = Vec::new();
    for (name, residual) in synthetic_fixtures() {
        let row = analyse(name.to_string(), "synthetic_integer_scaled", &residual);
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
    let syn_wins_fs = count_wins(&synthetic, "gain_vs_factor_shift_bytes");
    let real_wins_fs = count_wins(&real, "gain_vs_factor_shift_bytes");
    let real_wins_zm = count_wins(&real, "gain_vs_zero_mask_rice_bytes");

    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else if !real_present {
        Verdict::Inconclusive
    } else {
        Verdict::Supported
    };

    common::finish_exp3(
        "learned-valuation-split",
        receipts_root,
        LEARNED_VALUATION_SPLIT_SHA256,
        &projection,
        verdict,
        format!(
            "per-symbol trailing-zero valuation split (codec id 27) over {ns} synthetic + {nr} \
             real cases: beats the best pre-existing codec on {sw}/{ns} synthetic and {rw}/{nr} \
             real, FactorShift on {sfs}/{ns} synthetic and {rfs}/{nr} real, and ZeroMaskRice on \
             {rzm}/{nr} real",
            ns = synthetic.len(),
            nr = real.len(),
            sw = syn_wins,
            rw = real_wins,
            sfs = syn_wins_fs,
            rfs = real_wins_fs,
            rzm = real_wins_zm,
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
                    "factorization": "every nonzero magnitude is odd * 2^e with odd cores and a \
                                      per-symbol valuation e",
                    "streams": "zero gaps (Exp-Golomb, or absent in the dense mode), one sign bit \
                                per nonzero, the valuations (the best Exp2 coder, or absent when \
                                all are zero) and the odd cores (odd - 1) / 2 (the best Exp2 \
                                coder)",
                    "contrast": "FactorShift (id 14) strips one shared power of two for the whole \
                                 block; ValuationSplit pays it per symbol",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the two substantive streams each pay their own coder header, and a noisy \
                     valuation stream adds entropy the single-stream coders do not pay at all; \
                     that is why the split is domain-specific",
                    "on dithered or natural 24-bit material most valuations are effectively random \
                     and the split loses to a plain magnitude coder rather than beating one",
                ]),
            ),
        ],
    )
}

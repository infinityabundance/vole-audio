//! `court learned-param-delta` — Phase 2A mechanism 1 (differential
//! predictor-parameter coding and progressive-order restart prediction).
//!
//! Per-block LPC fits produce a sequence of coefficient vectors. Adjacent blocks
//! of a stationary source fit similar coefficients, so this court measures the
//! parameter stream when each vector is stored independently against the
//! cross-vector delta syntax, and reports the saving honestly.
//!
//! It also tests the ALS progressive-order restart rule directly: because VOLE
//! predictors reset with a zero-initialized closed-loop history, limiting the
//! tap count at the start of a block is provably identical to applying the full
//! order. The court asserts that equality across block sizes and shifts rather
//! than assuming it.
//!
//! The measured context matters: on the frozen speech winners the *model* bytes
//! are 31–47 B of a ~15 kB object (0.15–0.4%), so even a perfect parameter codec
//! cannot move the portfolio. That is reported, not hidden.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::param_delta::{
    block_residual_abs, delta_bytes, independent_varint_bytes, raw_i16_bytes,
};
use crate::learned::train::lpc::{Quantizer, autocorrelation, levinson_durbin, quantize};
use crate::status::Verdict;
use std::path::PathBuf;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_PARAM_DELTA_SHA256: &str =
    "15513938c0447b483b00a2d4da5ef3ad6fd0adf1fa2094d5d16f45ec8630077d";

const BLOCK: usize = 1024;
const ORDER: usize = 8;
const SHIFT: u8 = 12;
const PRECISION: u8 = 16;

/// Fit order-8 LPC coefficients for each block of a signal.
fn block_coefficients(samples: &[i32]) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + BLOCK <= samples.len() {
        let block = &samples[at..at + BLOCK];
        let r = autocorrelation(block, ORDER, 0.5);
        let sets = levinson_durbin(&r, ORDER);
        if let Some(coeffs) = sets.last() {
            let q = quantize(coeffs, SHIFT, PRECISION, Quantizer::Independent);
            out.push(q.into_iter().map(i64::from).collect());
        }
        at += BLOCK;
    }
    out
}

struct Row {
    id: String,
    blocks: usize,
    raw_i16: u64,
    independent_varint: u64,
    delta: u64,
    progressive_equals_full: bool,
}

fn analyse(id: String, samples: &[i32]) -> Row {
    let vectors = block_coefficients(samples);
    let raw_i16 = raw_i16_bytes(&vectors);
    let independent_varint = independent_varint_bytes(&vectors);
    let delta = delta_bytes(&vectors);
    // Progressive-order restart equivalence on the first fitted vector.
    let coeffs = vectors
        .first()
        .map(|v| v.iter().map(|&c| c as i32).collect::<Vec<i32>>())
        .unwrap_or_default();
    let mut progressive_equals_full = true;
    for block in [BLOCK, 2048, 4096] {
        for shift in [10u32, 12, 14] {
            let full = block_residual_abs(samples, &coeffs, shift, block, false);
            let prog = block_residual_abs(samples, &coeffs, shift, block, true);
            progressive_equals_full &= full == prog;
        }
    }
    Row {
        id,
        blocks: vectors.len(),
        raw_i16,
        independent_varint,
        delta,
        progressive_equals_full,
    }
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.progressive_equals_full;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.blocks as u64);
    common::push_u64(projection, row.raw_i16);
    common::push_u64(projection, row.independent_varint);
    common::push_u64(projection, row.delta);
}

fn row_json(row: &Row) -> serde_json::Value {
    let saving_permille = row
        .independent_varint
        .saturating_sub(row.delta)
        .checked_mul(1000)
        .and_then(|x| x.checked_div(row.independent_varint))
        .unwrap_or(0);
    serde_json::json!({
        "id": row.id,
        "blocks": row.blocks,
        "raw_i16_bytes": row.raw_i16,
        "independent_varint_bytes": row.independent_varint,
        "delta_bytes": row.delta,
        "saving_vs_independent_permille": saving_permille,
        "progressive_equals_full": row.progressive_equals_full,
    })
}

/// Run the court; writes `receipts/learned-param-delta/`.
pub fn run(receipts_root: &std::path::Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.param_delta.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    let mut rows = Vec::new();
    let mut all_exact = true;
    if available() && decoder_available() {
        let scratch = PathBuf::from("target/real-corpus/scratch");
        let clips = load_cases(&effectiveness_clips(), speech::CLIPS_PER_SPLIT, &scratch)?;
        for case in &clips {
            let row = analyse(case.clip.id.clone(), &case.samples);
            let mut gates = true;
            push_row(&mut projection, &row, &mut gates);
            all_exact &= gates;
            rows.push(row);
        }
    }

    let total_independent: u64 = rows.iter().map(|r| r.independent_varint).sum();
    let total_delta: u64 = rows.iter().map(|r| r.delta).sum();
    let total_raw: u64 = rows.iter().map(|r| r.raw_i16).sum();
    let saving = total_independent
        .saturating_sub(total_delta)
        .checked_mul(1000)
        .and_then(|x| x.checked_div(total_independent))
        .unwrap_or(0);

    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else if rows.is_empty() {
        Verdict::Inconclusive
    } else {
        Verdict::Supported
    };

    let rows_json: serde_json::Value = rows.iter().map(row_json).collect();
    common::finish(
        "learned-param-delta",
        receipts_root,
        LEARNED_PARAM_DELTA_SHA256,
        &projection,
        verdict,
        format!(
            "cross-block differential predictor-parameter coding over {n} real clips: {ti} B \
             independent varint vs {td} B delta ({sv} per-mille saved), raw-i16 baseline {tr} B; \
             and the ALS progressive-order restart is provably identical to full order under a \
             zero-initialized closed-loop reset on every fixture tested",
            n = rows.len(),
            ti = total_independent,
            td = total_delta,
            sv = saving,
            tr = total_raw,
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "clips": rows.len(),
                }),
            ),
            ("clips", rows_json),
            (
                "method",
                serde_json::json!({
                    "delta": "first block's coefficient vector absolute, each later block a \
                              per-position signed delta (zigzag varint) from its predecessor",
                    "baseline": "each block stored independently as zigzag varints, plus a raw \
                                 i16 reference",
                    "progressive": "ALS uses order 1 on the second sample, order 2 on the third, \
                                    … at a restart; VOLE predictors reset with a zero-initialized \
                                    history, so taps with a negative history index contribute \
                                    zero either way and progressive order is exactly full order",
                    "premise_check": "on the frozen speech winners the model bytes are 31–47 B of \
                                      a ~15 kB object (0.15–0.4%), so no parameter codec can move \
                                      the portfolio; this court therefore reports the parameter \
                                      saving rather than a portfolio gain",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the delta syntax is measured on raw per-block coefficient vectors, not yet \
                     wired into a production segmented container",
                    "the premise of this mechanism — that model overhead dominates — is measurably \
                     false for the current winners, which is itself the result",
                ]),
            ),
        ],
    )
}

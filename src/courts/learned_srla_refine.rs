//! `court learned-srla-refine` — Phase 2A mechanism 2 (SRLA code-length-shaped
//! coefficient refinement).
//!
//! SRLA refines quantized integer predictor coefficients against a code-length
//! objective instead of squared error. This court fits a real per-clip LPC
//! predictor, refines its coefficients with an exact Rice-length hill climb, and
//! then keeps the refined candidate **only if its exact canonical complete bytes
//! improve** — the proposal may be shaped by a code-length proxy, but the
//! physical artifact remains the authority.
//!
//! The mechanism's premise is checked against the fact that the `lpc` family is
//! not the current portfolio winner (NGSA is), so the court reports the per-clip
//! exact-byte gain rather than a portfolio gain.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::lpc::LpcPredictor;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::srla::{refine_coefficients, residual_rice_bits};
use crate::learned::train::lpc::{Quantizer, autocorrelation, levinson_durbin, quantize};
use crate::status::Verdict;
use std::path::PathBuf;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_SRLA_REFINE_SHA256: &str =
    "7190315e47476974a8bac274f59a68c7067046758e8513443e6e35f0e27153f5";

const ORDER: usize = 8;
const SHIFT: u8 = 12;
const PRECISION: u8 = 16;
const PASSES: usize = 8;

fn build_object(
    coeffs: &[i32],
    channels: u8,
    frames: u64,
    rate: u32,
    samples: &[i32],
) -> Option<(LearnedObject, u64)> {
    let p = LpcPredictor {
        channels,
        order: coeffs.len() as u16,
        precision: PRECISION,
        shift: SHIFT,
        coeffs: coeffs.to_vec(),
        block_frames: None,
    };
    let o = LearnedObject::from_intrinsic_exp2(
        LearnedModel::Lpc(p),
        channels,
        frames,
        rate,
        Vec::new(),
        samples,
    )
    .ok()?;
    if !o.verify(samples) {
        return None;
    }
    let bytes = LearnedCost::of(&o).ok()?.complete_bytes;
    Some((o, bytes))
}

struct Row {
    id: String,
    seed_bits: u64,
    refined_bits: u64,
    seed_bytes: u64,
    refined_bytes: u64,
    accepted: bool,
    exact: bool,
}

fn analyse(id: String, samples: &[i32], rate: u32) -> Option<Row> {
    let frames = samples.len() as u64;
    let r = autocorrelation(samples, ORDER, 0.5);
    let sets = levinson_durbin(&r, ORDER);
    let coeffs = quantize(sets.last()?, SHIFT, PRECISION, Quantizer::Independent);
    let seed_bits = residual_rice_bits(samples, &coeffs, u32::from(SHIFT));
    let refined = refine_coefficients(samples, &coeffs, u32::from(SHIFT), PASSES);
    let refined_bits = residual_rice_bits(samples, &refined, u32::from(SHIFT));

    let (_so, seed_bytes) = build_object(&coeffs, 1, frames, rate, samples)?;
    let refined_obj = build_object(&refined, 1, frames, rate, samples);
    // The physical artifact decides: keep the refinement only when it is
    // strictly smaller, otherwise fall back to the seed exactly.
    let (refined_bytes, accepted, exact) = match refined_obj {
        Some((_o, b)) if b < seed_bytes => (b, true, true),
        Some(_) => (seed_bytes, false, true),
        None => (seed_bytes, false, true),
    };
    Some(Row {
        id,
        seed_bits,
        refined_bits,
        seed_bytes,
        refined_bytes,
        accepted,
        exact,
    })
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.exact && row.refined_bytes <= row.seed_bytes;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.seed_bits);
    common::push_u64(projection, row.refined_bits);
    common::push_u64(projection, row.seed_bytes);
    common::push_u64(projection, row.refined_bytes);
}

fn row_json(row: &Row) -> serde_json::Value {
    serde_json::json!({
        "id": row.id,
        "seed_rice_bits": row.seed_bits,
        "refined_rice_bits": row.refined_bits,
        "seed_bytes": row.seed_bytes,
        "refined_bytes": row.refined_bytes,
        "gain_bytes": row.seed_bytes as i64 - row.refined_bytes as i64,
        "accepted": row.accepted,
        "exact": row.exact,
    })
}

/// Run the court; writes `receipts/learned-srla-refine/`.
pub fn run(receipts_root: &std::path::Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.srla_refine.v1");
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
            let rate = case.rate();
            if let Some(row) = analyse(case.clip.id.clone(), &case.samples, rate) {
                let mut gates = true;
                push_row(&mut projection, &row, &mut gates);
                all_exact &= gates;
                rows.push(row);
            }
        }
    }

    let seed_total: u64 = rows.iter().map(|r| r.seed_bytes).sum();
    let refined_total: u64 = rows.iter().map(|r| r.refined_bytes).sum();
    let accepted = rows.iter().filter(|r| r.accepted).count() as u64;
    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else if rows.is_empty() {
        Verdict::Inconclusive
    } else {
        Verdict::Supported
    };

    let rows_json: serde_json::Value = rows.iter().map(row_json).collect();
    common::finish(
        "learned-srla-refine",
        receipts_root,
        LEARNED_SRLA_REFINE_SHA256,
        &projection,
        verdict,
        format!(
            "code-length-shaped coefficient refinement over {n} real clips: exact complete bytes \
             {s} B seed vs {r} B refined ({g} B gained), accepted on {a}/{n}; the Rice objective \
             guides the proposal and the exact bytes accept it — on the winning ngsa family there \
             are no such coefficients, so this cannot move the portfolio",
            n = rows.len(),
            s = seed_total,
            r = refined_total,
            g = seed_total as i64 - refined_total as i64,
            a = accepted,
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
                    "objective": "exact minimum Rice bit length of the open-loop residual over \
                                  k = 0..=20 — the discrete form of SRLA's continuous Recursive \
                                  Golomb-Rice approximation",
                    "proposal": "deterministic hill climb over ±1 coefficient perturbations, \
                                 accepting strict objective improvements until a pass is unchanged",
                    "authority": "the refined candidate is kept only when its exact canonical \
                                  complete bytes are smaller, so the code-length proxy never \
                                  overrides the physical artifact",
                    "premise_check": "the lpc family is not the portfolio winner (ngsa is), so the \
                                      gain is reported per clip, not as a portfolio claim",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the refinement is measured on a whole-clip order-8 LPC, not wired into the \
                     production fit pipeline",
                    "SRLA's own reported gain was small and its encoder cost large; this court \
                     measures the byte gain only",
                ]),
            ),
        ],
    )
}

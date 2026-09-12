//! `court learned-residual-fusion` — fourth-pass **Seal F0** diagnostic.
//!
//! Multi-hypothesis probability fusion (report Tier 1 #4) asks whether the
//! *disagreement* between several already-known predictor hypotheses carries
//! entropy-side information: if every cheap hypothesis agrees with the selected
//! one, the residual is probably near zero; wild disagreement widens it.
//!
//! A structural fact shapes how it must be built: VOLE decodes the dense residual
//! **before** reconstruction, so no predictor hypothesis is decoder-visible while
//! the residual bits are being coded. Fusion therefore either (a) transmits the
//! disagreement features as a fully-charged side stream, or (b) becomes a
//! closed-loop entropy coder. Before paying for either, this court measures the
//! *upper bound*: the plug-in conditional entropy `H(bit | disagreement feature)`
//! of the exact S8 winner residual, over a small exact hypothesis ensemble whose
//! members are deterministic functions of reconstructed history.
//!
//! It computes no new format and makes no compression claim: it answers "is the
//! fusion worth an architecture change?" on effectiveness data only. The held-out
//! Mode C split is untouched.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::anatomy::{conditional_bits_per_sample, order0_bits_per_sample};
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::object::LearnedObject;
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_RESIDUAL_FUSION_SHA256: &str =
    "b01dd7e46723561166e2066436254ce739381cb2561a1efdc0cd33a3cbbcbf1f";

/// Bit-length bucket (0..=20).
fn bucket(v: u64) -> u64 {
    let bl = 64 - v.leading_zeros();
    u64::from(bl).min(20)
}

/// The hypothesis-ensemble disagreement features for one sample.
///
/// Returns `(range, within2, sign_consensus, min_offset, combined)`.
fn features(h0: i64, x1: i64, x2: i64, mean: i64) -> (u64, u64, u64, u64, u64) {
    let a = [x1, x2, (x1 + x2) / 2, 2 * x1 - x2, mean];
    let d: Vec<i64> = a.iter().map(|&h| h - h0).collect();
    let range = d.iter().copied().max().unwrap_or(0) - d.iter().copied().min().unwrap_or(0);
    let within2 = d.iter().filter(|&&v| v.unsigned_abs() <= 2).count() as u64;
    let consensus = d.iter().filter(|&&v| v >= 0).count() as u64;
    let min_off = d.iter().map(|v| v.unsigned_abs()).min().unwrap_or(0);
    let combined = (bucket(range as u64) * 6 + within2.min(5)) * 6 + consensus.min(5);
    (
        bucket(range as u64),
        within2,
        consensus,
        bucket(min_off),
        combined,
    )
}

/// Per-clip feature-entropy summary.
struct Row {
    id: String,
    samples: u64,
    order0: f64,
    by_range: f64,
    by_within2: f64,
    by_consensus: f64,
    by_min_offset: f64,
    by_combined: f64,
}

/// Analyse one exact residual against its ensemble.
fn analyse(id: &str, residual: &[i32], source: &[i32]) -> Row {
    let n = residual.len();
    let sum: i128 = source.iter().map(|&v| i128::from(v)).sum();
    let mean = if n == 0 { 0 } else { (sum / n as i128) as i64 };
    let mut c_range = Vec::with_capacity(n);
    let mut c_within2 = Vec::with_capacity(n);
    let mut c_cons = Vec::with_capacity(n);
    let mut c_min = Vec::with_capacity(n);
    let mut c_comb = Vec::with_capacity(n);
    for t in 0..n {
        let x1 = if t >= 1 { i64::from(source[t - 1]) } else { 0 };
        let x2 = if t >= 2 { i64::from(source[t - 2]) } else { x1 };
        let h0 = i64::from(source[t]) - i64::from(residual[t]);
        let (r, w, s, mo, comb) = features(h0, x1, x2, mean);
        c_range.push(r);
        c_within2.push(w);
        c_cons.push(s);
        c_min.push(mo);
        c_comb.push(comb);
    }
    Row {
        id: id.to_string(),
        samples: n as u64,
        order0: order0_bits_per_sample(residual),
        by_range: conditional_bits_per_sample(residual, &c_range),
        by_within2: conditional_bits_per_sample(residual, &c_within2),
        by_consensus: conditional_bits_per_sample(residual, &c_cons),
        by_min_offset: conditional_bits_per_sample(residual, &c_min),
        by_combined: conditional_bits_per_sample(residual, &c_comb),
    }
}

fn aggregate(rows: &[Row]) -> serde_json::Value {
    let total: u64 = rows.iter().map(|r| r.samples).sum();
    let w = |f: fn(&Row) -> f64| -> f64 {
        if total == 0 {
            0.0
        } else {
            rows.iter().map(|r| f(r) * r.samples as f64).sum::<f64>() / total as f64
        }
    };
    let order0 = w(|r| r.order0);
    let combined = w(|r| r.by_combined);
    serde_json::json!({
        "clips": rows.len(),
        "residual_samples": total,
        "order0_bits_per_residual": order0,
        "by_range": w(|r| r.by_range),
        "by_within2": w(|r| r.by_within2),
        "by_consensus": w(|r| r.by_consensus),
        "by_min_offset": w(|r| r.by_min_offset),
        "by_combined": combined,
        "combined_saving_bits_per_residual": order0 - combined,
    })
}

/// Run the court; writes `receipts/learned-residual-fusion/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.residual_fusion.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    if !available() || !decoder_available() {
        return common::finish_exp2(
            "learned-residual-fusion",
            receipts_root,
            LEARNED_RESIDUAL_FUSION_SHA256,
            &projection,
            Verdict::Inconclusive,
            "real corpus audio or the external `flac` decoder is absent".to_string(),
            vec![(
                "limitations",
                serde_json::json!([
                    "the diagnostic requires the uncommitted LibriSpeech audio bulk under \
                     corpus/real/ and the external `flac` decoder; the held-out Mode-C split is \
                     deliberately not used"
                ]),
            )],
        );
    }

    let mut budget = common::train_budget();
    budget.max_iterations = 128;
    let scratch = PathBuf::from("target/real-corpus/scratch");
    let clips = load_cases(&effectiveness_clips(), speech::CLIPS_PER_SPLIT, &scratch)?;
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut analysed: Vec<Row> = Vec::new();
    let mut all_exact = true;
    for case in &clips {
        let cands = speech::portfolio(case, &budget)?;
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
        let (family, o, bytes) = best.ok_or_else(|| {
            crate::error::Error::internal("empty speech portfolio in the fusion court")
        })?;
        let residual = o.residual()?;
        all_exact &= residual.len() == case.samples.len();
        let row = analyse(&case.clip.id, &residual, &case.samples);
        common::push_label(&mut projection, &case.clip.id);
        common::push_label(&mut projection, family);
        common::push_u64(&mut projection, bytes);
        common::push_u64(&mut projection, row.samples);
        common::push_label(&mut projection, &format!("{:.6}", row.by_combined));
        rows.push(serde_json::json!({
            "id": row.id.clone(),
            "family": family,
            "bytes": bytes,
            "samples": row.samples,
            "order0_bits_per_residual": row.order0,
            "by_range": row.by_range,
            "by_within2": row.by_within2,
            "by_consensus": row.by_consensus,
            "by_min_offset": row.by_min_offset,
            "by_combined": row.by_combined,
        }));
        analysed.push(row);
    }
    let summary = aggregate(&analysed);

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish_exp2(
        "learned-residual-fusion",
        receipts_root,
        LEARNED_RESIDUAL_FUSION_SHA256,
        &projection,
        verdict,
        format!(
            "multi-hypothesis disagreement anatomy over {} effectiveness clips: the combined \
             disagreement feature saves {:.4} bits/residual over order-0 (upper bound; a real \
             coder would have to transmit the features)",
            rows.len(),
            summary["combined_saving_bits_per_residual"]
                .as_f64()
                .unwrap_or(0.0)
        ),
        vec![
            ("summary", summary),
            ("clips", serde_json::json!(rows)),
            (
                "method",
                serde_json::json!({
                    "ensemble": "the selected S8 hypothesis H0 plus five deterministic exact \
                                 alternatives from reconstructed history: previous sample, \
                                 second-previous, their average, their linear extrapolation, and \
                                 the clip mean",
                    "features": "range bucket, count within ±2, sign consensus, min |offset| \
                                 bucket, and a combined key",
                    "estimator": "plug-in conditional entropy of the frozen canonical \
                                  binarization, in bits per residual sample",
                    "architectural_fact": "VOLE decodes the dense residual before reconstruction, \
                                           so a real fusion coder needs a fully-charged side \
                                           stream or closed-loop entropy coding; this measures \
                                           the upper bound before paying for either",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "an upper bound only: the plug-in estimate is optimistic and the features' \
                     transmission cost is not charged here",
                    "the ensemble is a small deterministic set chosen a priori, not an optimized \
                     predictor bank",
                    "no representation, profile or format is changed",
                ]),
            ),
        ],
    )
}

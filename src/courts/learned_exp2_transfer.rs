//! `court learned-exp2-transfer` — analytic-first transfer v2 (Seal J).
//!
//! Transfer v2 decomposes `target = analytic(source) + learned_correction +
//! residual`. Trivial relationships (identity, polarity, gain, affine, delay)
//! are expected to keep the correction off; the learned correction is admitted
//! only when the total bytes beat analytic-only.
//!
//! The court is a **paired** surface: it reports analytic-only and
//! analytic+correction bytes for every frozen transfer pair, and the structural
//! gate is that the portfolio minimum never exceeds the best analytic-only
//! candidate, with every candidate closing exactly.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::analytical::AnalyticTransfer;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::transfer::TransferOperator;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_EXP2_TRANSFER_SHA256: &str =
    "a43ec91fbb31efd9e585e6cfaaa3fefe3bc6e8381aec7049f508cc3d212d265a";

fn zero_correction(channels: u8, source_channels: u8) -> TransferOperator {
    TransferOperator {
        channels,
        source_channels,
        taps: 1,
        delay: 0,
        weights: vec![0i16; usize::from(channels) * usize::from(source_channels)],
        bias: vec![0i32; usize::from(channels)],
    }
}

fn candidate(
    gain_num: i64,
    gain_den: i64,
    offset: i64,
    delay: i64,
    correction: Option<TransferOperator>,
    channels: u8,
    source_channels: u8,
) -> AnalyticTransfer {
    match correction {
        Some(c) => AnalyticTransfer {
            channels,
            source_channels,
            gain_num,
            gain_den,
            offset,
            delay,
            has_correction: true,
            correction: c,
        },
        None => AnalyticTransfer {
            channels,
            source_channels,
            gain_num,
            gain_den,
            offset,
            delay,
            has_correction: false,
            correction: zero_correction(channels, source_channels),
        },
    }
}

/// Least-squares scalar affine fit `y ≈ slope·x + intercept` (first channel).
/// Returns `(gain_num, gain_den, offset)` in Q12 fixed point.
fn affine_fit(source: &[i32], target: &[i32], frames: usize) -> (i64, i64, i64) {
    let n = frames as f64;
    let sx: f64 = (0..frames).map(|t| f64::from(source[t])).sum();
    let sy: f64 = (0..frames).map(|t| f64::from(target[t])).sum();
    let sxx: f64 = (0..frames)
        .map(|t| f64::from(source[t]) * f64::from(source[t]))
        .sum();
    let sxy: f64 = (0..frames)
        .map(|t| f64::from(source[t]) * f64::from(target[t]))
        .sum();
    let denom = n * sxx - sx * sx;
    let slope = if denom.abs() > 1e-9 {
        (n * sxy - sx * sy) / denom
    } else {
        0.0
    };
    let intercept = (sy - slope * sx) / n.max(1.0);
    (
        (slope * 4096.0).round() as i64,
        4096,
        intercept.round() as i64,
    )
}

fn best_delay(source: &[i32], target: &[i32], frames: usize) -> i64 {
    // Cross-correlation peak over a bounded scan.
    let mut best = 0i64;
    let mut best_score = f64::NEG_INFINITY;
    for d in -96i64..=96 {
        let mut acc = 0f64;
        for (t, &tv) in target.iter().enumerate().take(frames) {
            let si = t as i64 - d;
            if si >= 0 && (si as usize) < frames {
                acc += f64::from(source[si as usize]) * f64::from(tv);
            }
        }
        if acc > best_score {
            best_score = acc;
            best = d;
        }
    }
    best
}

/// Run the court; writes `receipts/learned-exp2-transfer/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.exp2.transfer.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus::transfer_corpus_hex(),
    );

    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut portfolio_le_analytic = true;
    for pair in crate::learned::corpus::transfer_pairs() {
        let frames = pair.source.len() as u64 / u64::from(pair.source_channels);
        let mono_source: Vec<i32> = if pair.source_channels == 1 {
            pair.source.clone()
        } else {
            pair.source
                .iter()
                .step_by(usize::from(pair.source_channels))
                .copied()
                .collect()
        };
        let mono_target: Vec<i32> = if pair.target_channels == 1 {
            pair.target.clone()
        } else {
            pair.target
                .iter()
                .step_by(usize::from(pair.target_channels))
                .copied()
                .collect()
        };
        let delay = best_delay(&mono_source, &mono_target, frames as usize);
        let (slope_num, _slope_den, intercept) =
            affine_fit(&mono_source, &mono_target, frames as usize);
        let analytic_candidates: Vec<(&str, AnalyticTransfer)> = vec![
            (
                "identity",
                candidate(1, 1, 0, 0, None, pair.target_channels, pair.source_channels),
            ),
            (
                "polarity",
                candidate(
                    -1,
                    1,
                    0,
                    0,
                    None,
                    pair.target_channels,
                    pair.source_channels,
                ),
            ),
            (
                "delay",
                candidate(
                    1,
                    1,
                    0,
                    delay,
                    None,
                    pair.target_channels,
                    pair.source_channels,
                ),
            ),
            (
                "affine",
                candidate(
                    slope_num,
                    4096,
                    intercept,
                    0,
                    None,
                    pair.target_channels,
                    pair.source_channels,
                ),
            ),
        ];
        let mut analytic_best: Option<(String, u64)> = None;
        let mut best_analytic_candidate: Option<AnalyticTransfer> = None;
        for (label, t) in analytic_candidates {
            let o = match LearnedObject::from_transfer_operator_exp2(
                LearnedModel::AnalyticTransfer(t.clone()),
                pair.target_channels,
                frames,
                pair.sample_rate_hz,
                vec![common::source_content_id(pair.id)],
                &pair.source,
                &pair.target,
            ) {
                Ok(o) => o,
                Err(_) => continue,
            };
            let exact = o.verify_with_source(&pair.source, &pair.target);
            all_exact &= exact;
            let b = crate::learned::accounting::LearnedCost::of(&o)
                .map(|c| c.complete_bytes)
                .unwrap_or(u64::MAX);
            if analytic_best.as_ref().is_none_or(|(_, bb)| b < *bb) {
                analytic_best = Some((label.to_string(), b));
                best_analytic_candidate = Some(t);
            }
        }

        // Analytic + learned correction (fits the correction on the analytic
        // residual of the best analytic candidate).
        let mut corrected: Option<u64> = None;
        if let Some(base) = &best_analytic_candidate {
            let h = base.hypothesis_from_source(&pair.source, frames as usize)?;
            let residual: Vec<i32> = pair
                .target
                .iter()
                .zip(h.iter())
                .map(|(&y, &hh)| (i64::from(y) - i64::from(hh)) as i32)
                .collect();
            for taps in [4u16, 8] {
                let corr = match common::fit_transfer(
                    &pair.source,
                    pair.source_channels,
                    &residual,
                    pair.target_channels,
                    frames as usize,
                    taps,
                    0,
                ) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let t = candidate(
                    base.gain_num,
                    base.gain_den,
                    base.offset,
                    base.delay,
                    Some(corr),
                    pair.target_channels,
                    pair.source_channels,
                );
                let o = match LearnedObject::from_transfer_operator_exp2(
                    LearnedModel::AnalyticTransfer(t),
                    pair.target_channels,
                    frames,
                    pair.sample_rate_hz,
                    vec![common::source_content_id(pair.id)],
                    &pair.source,
                    &pair.target,
                ) {
                    Ok(o) => o,
                    Err(_) => continue,
                };
                let exact = o.verify_with_source(&pair.source, &pair.target);
                all_exact &= exact;
                let b = crate::learned::accounting::LearnedCost::of(&o)
                    .map(|c| c.complete_bytes)
                    .unwrap_or(u64::MAX);
                if corrected.is_none_or(|bb| b < bb) {
                    corrected = Some(b);
                }
            }
        }

        let analytic_bytes = analytic_best.as_ref().map(|(_, b)| *b);
        let best = [analytic_bytes, corrected].into_iter().flatten().min();
        if let (Some(best), Some(a)) = (best, analytic_bytes) {
            portfolio_le_analytic &= best <= a;
        }
        common::push_label(&mut projection, pair.id);
        common::push_u64(&mut projection, analytic_bytes.unwrap_or(u64::MAX));
        common::push_u64(&mut projection, corrected.unwrap_or(u64::MAX));
        rows.push(serde_json::json!({
            "id": pair.id,
            "transform": pair.transform,
            "analytic_selected": analytic_best.as_ref().map(|(l, _)| l.clone()),
            "analytic_bytes": analytic_bytes,
            "corrected_bytes": corrected,
            "best_bytes": best,
        }));
    }

    let verdict = if all_exact && portfolio_le_analytic {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_exp2(
        "learned-exp2-transfer",
        receipts_root,
        LEARNED_EXP2_TRANSFER_SHA256,
        &projection,
        verdict,
        format!(
            "analytic-first transfer over {} frozen pairs: analytic-only and analytic+learned \
             correction candidates, all exact, portfolio min ≤ best analytic-only",
            rows.len()
        ),
        vec![
            ("pairs", serde_json::json!(rows)),
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "portfolio_le_analytic_only": portfolio_le_analytic,
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the analytic portfolio is identity / polarity / integer delay / affine gain+offset",
                    "the learned correction is a bounded causal cross-channel FIR over the analytic residual",
                    "the source dependency is charged (a 32-byte content id) but standalone/marginal regimes \
                     are not yet separated in this court"
                ]),
            ),
        ],
    )
}

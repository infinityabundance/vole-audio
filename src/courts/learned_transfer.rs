//! `court learned-transfer` — learned transfer operators vs analytic baselines
//! (`O.4`, `O.5`, `O.6`, `O.49`).
//!
//! Every learned transfer candidate competes against simple deterministic
//! relationships (identity, fixed gain, affine, delay, FIR, IIR, convolution,
//! static polynomial, piecewise-linear, moving average) *and* against the
//! independent target. Source dependencies are never free: standalone and
//! marginal economics are reported separately.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{transfer_corpus_hex, transfer_pairs};
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::residual_codec::encode_best;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_TRANSFER_SHA256: &str =
    "10fc0eb677418832f74f7c539a6fed9e7dce092dedbddb4e6cef69d363d2641c";

const TRANSFER_TAPS: [u16; 7] = [1, 2, 4, 8, 16, 32, 64];
const TRANSFER_DELAYS: [i64; 14] = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 7, 13, 16, 23, 31];

/// Framing bytes of a transfer container (mirrors the learned object's fixed
/// fields: magic, version, profile, geometry, counts).
const FRAMING: u64 = 84;

fn price_residual(target: &[i32], h: &[i32], param_bytes: u64) -> Option<u64> {
    if target.len() != h.len() {
        return None;
    }
    let mut residual = vec![0i32; target.len()];
    for i in 0..target.len() {
        let d = i64::from(target[i]) - i64::from(h[i]);
        if d < i64::from(i32::MIN) || d > i64::from(i32::MAX) {
            return None;
        }
        residual[i] = d as i32;
    }
    let enc = encode_best(&residual);
    Some(FRAMING + 32 + 32 + param_bytes + enc.complete_bytes())
}

fn analytic(source: &[i32], target: &[i32]) -> Vec<(&'static str, u64)> {
    let n = source.len();
    let mut out = Vec::new();
    let push =
        |out: &mut Vec<(&'static str, u64)>, name: &'static str, h: Vec<i32>, params: u64| {
            if let Some(c) = price_residual(target, &h, params) {
                out.push((name, c));
            }
        };
    push(&mut out, "identity", source.to_vec(), 0);
    push(
        &mut out,
        "gain_0.5",
        source.iter().map(|&x| x / 2).collect(),
        4,
    );
    push(
        &mut out,
        "affine",
        source.iter().map(|&x| x / 2 + 100_000).collect(),
        8,
    );
    {
        let mut h = vec![0i32; n];
        h[13..].copy_from_slice(&source[..n - 13]);
        push(&mut out, "delay_13", h, 8);
    }
    {
        let taps = [0.6f64, 0.3, 0.1];
        let mut h = vec![0i32; n];
        for t in 0..n {
            let mut acc = 0f64;
            for (k, &a) in taps.iter().enumerate() {
                if t >= k {
                    acc += a * f64::from(source[t - k]);
                }
            }
            h[t] = acc as i64 as i32;
        }
        push(&mut out, "fir3", h, 6);
    }
    {
        let mut y = 0f64;
        let h: Vec<i32> = source
            .iter()
            .map(|&v| {
                y = 0.8 * y + 0.4 * f64::from(v);
                y as i64 as i32
            })
            .collect();
        push(&mut out, "iir1", h, 16);
    }
    {
        // Room impulse response (same shape as the corpus convolution).
        let mut ir = vec![0f64; 64];
        ir[0] = 0.7;
        ir[7] = 0.35;
        ir[23] = 0.2;
        ir[41] = -0.12;
        ir[59] = 0.07;
        let mut h = vec![0i32; n];
        for t in 0..n {
            let mut acc = 0f64;
            for (k, &a) in ir.iter().enumerate() {
                if t >= k {
                    acc += a * f64::from(source[t - k]);
                }
            }
            h[t] = acc as i64 as i32;
        }
        push(&mut out, "convolution", h, 128);
    }
    push(
        &mut out,
        "poly_softclip",
        source
            .iter()
            .map(|&x| {
                let u = f64::from(x) / f64::from(1 << 24);
                ((1i64 << 24) as f64 * (u - u * u * u / 3.0)) as i64 as i32
            })
            .collect(),
        8,
    );
    push(
        &mut out,
        "piecewise_halfwave",
        source
            .iter()
            .map(|&x| if x > 0 { x / 2 } else { x / 4 })
            .collect(),
        16,
    );
    {
        let mut h = vec![0i32; n];
        for t in 0..n {
            let mut acc = 0i64;
            for k in 0..9 {
                if t >= k {
                    acc += i64::from(source[t - k]);
                }
            }
            h[t] = (acc / 9) as i32;
        }
        push(&mut out, "moving_average_9", h, 18);
    }
    out
}

/// Run the court; writes `receipts/learned-transfer/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.transfer.v1");
    common::push_label(&mut projection, &transfer_corpus_hex());

    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut learned_wins = 0u64;
    let mut worst_learned_ratio = 0f64;

    for pair in transfer_pairs() {
        let frames = pair.source.len();
        let target_frames = pair.target.len();
        assert_eq!(frames, target_frames);

        let mut best_analytic: Option<(&'static str, u64)> = None;
        for (name, cost) in analytic(&pair.source, &pair.target) {
            if best_analytic.is_none_or(|(_, c)| cost < c) {
                best_analytic = Some((name, cost));
            }
        }
        let (analytic_name, analytic_bytes) = best_analytic.unwrap_or(("none", u64::MAX));

        // Independent target representation (no source dependency).
        let independent = common::best_simple(
            &pair.target,
            pair.target_channels,
            frames as u64,
            pair.sample_rate_hz,
        )
        .map(|(_, b)| b)
        .unwrap_or(u64::MAX);

        // Learned transfer over a bounded tap/delay sweep.
        let dep = common::source_content_id(pair.id);
        let mut best_learned: Option<(LearnedObject, u64, u16, i64)> = None;
        for &taps in &TRANSFER_TAPS {
            for &delay in &TRANSFER_DELAYS {
                let op = match common::fit_transfer(
                    &pair.source,
                    pair.source_channels,
                    &pair.target,
                    pair.target_channels,
                    frames,
                    taps,
                    delay,
                ) {
                    Ok(o) => o,
                    Err(_) => continue,
                };
                let model = LearnedModel::Transfer(op);
                let obj = match LearnedObject::from_transfer_operator(
                    model,
                    pair.target_channels,
                    frames as u64,
                    pair.sample_rate_hz,
                    vec![dep],
                    &pair.source,
                    &pair.target,
                ) {
                    Ok(o) => o,
                    Err(_) => continue,
                };
                if !obj.verify_with_source(&pair.source, &pair.target) {
                    all_exact = false;
                    continue;
                }
                let bytes = common::learned_bytes(&obj)?;
                if best_learned.as_ref().is_none_or(|(_, b, _, _)| bytes < *b) {
                    best_learned = Some((obj, bytes, taps, delay));
                }
            }
        }

        let (learned_bytes, learned_taps, learned_delay, marginal) = match &best_learned {
            Some((o, b, t, d)) => {
                let cost = crate::learned::accounting::LearnedCost::of(o)?;
                (
                    *b,
                    Some(*t),
                    Some(*d),
                    cost.complete_bytes - cost.dependency_bytes,
                )
            }
            None => (u64::MAX, None, None, u64::MAX),
        };
        if learned_bytes <= analytic_bytes {
            learned_wins += 1;
        }
        if learned_bytes != u64::MAX && analytic_bytes != u64::MAX {
            worst_learned_ratio =
                worst_learned_ratio.max(learned_bytes as f64 / analytic_bytes as f64);
        }

        common::push_label(&mut projection, pair.id);
        common::push_u64(&mut projection, analytic_bytes);
        common::push_u64(&mut projection, learned_bytes);
        common::push_u64(&mut projection, independent);
        common::push_u64(&mut projection, marginal);
        common::push_u64(&mut projection, u64::from(learned_taps.unwrap_or(0)));
        common::push_u64(&mut projection, learned_delay.unwrap_or(0) as u64);

        rows.push(serde_json::json!({
            "id": pair.id,
            "transform": pair.transform,
            "best_analytic": analytic_name,
            "best_analytic_bytes": analytic_bytes,
            "learned_bytes": if learned_bytes == u64::MAX { None } else { Some(learned_bytes) },
            "learned_marginal_bytes_given_source": if marginal == u64::MAX { None } else { Some(marginal) },
            "learned_taps": learned_taps,
            "learned_delay": learned_delay,
            "independent_target_bytes": independent,
            "regimes": {
                "standalone": if learned_bytes == u64::MAX { None } else { Some(learned_bytes) },
                "marginal_given_source": if marginal == u64::MAX { None } else { Some(marginal) },
                "learned_over_analytic": if learned_bytes == u64::MAX || analytic_bytes == u64::MAX {
                    None
                } else {
                    Some(learned_bytes as f64 / analytic_bytes as f64)
                },
            },
        }));
    }

    common::push_u64(&mut projection, learned_wins);

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    let n = rows.len() as u64;
    common::finish(
        "learned-transfer",
        receipts_root,
        LEARNED_TRANSFER_SHA256,
        &projection,
        verdict,
        format!(
            "learned transfer operators over {n} paired relationships: {learned_wins} at or below \
             the best analytic baseline; worst learned/analytic ratio {worst_learned_ratio:.3}; \
             every learned candidate closes exactly to its target"
        ),
        vec![
            (
                "regimes",
                serde_json::json!({
                    "standalone": "includes the source dependency's bytes",
                    "marginal_given_source": "excludes the source bytes when the source already exists",
                    "note": "a ratio is never formed across the two regimes",
                }),
            ),
            ("pairs", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "the source dependency is a declared content id; its bytes are priced in the \
                     standalone regime and excluded only in the explicitly labelled marginal regime",
                    "the learned operator is a bounded causal cross-channel FIR; nonlinear \
                     transfer operators are not implemented in this build"
                ]),
            ),
        ],
    )
}

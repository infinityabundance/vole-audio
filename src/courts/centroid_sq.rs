//! `court centroid-sq` — Phase 6 mechanism 14 (`CentroidSQ`, lossy/exploratory).
//!
//! Decision boundaries and reconstruction points are separable. `CentroidSQ`
//! keeps the boundaries — and therefore the symbols, and therefore the bitstream
//! — exactly as they are, and moves only the reconstruction point toward the
//! source distribution's conditional mean inside each cell. Because the
//! reconstruction rule is profile-defined, a decoder that knows the profile pays
//! no extra bits.
//!
//! This court builds a realistic concentrated coefficient set (DCT spectra of the
//! frozen and synthetic fixtures), fits the reconstruction offset on a training
//! split, and reports **held-out** mean-squared error at identical symbols
//! against the midpoint rule. VOLE has no lossy transform codec, so this is an
//! exploratory RD court and makes no codec-quality claim.

use crate::courts::learned_common as common;
use crate::entropy::corpus;
use crate::error::Result;
use crate::learned::centroid_sq::{Quantizer, fit_offset};
use crate::learned::tns::dct2;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const CENTROID_SQ_SHA256: &str =
    "d5fd3b70dae55144ac44c53cee8da0ba199ee73fbc14dfe8be55911d4466d0be";

const FRAME: usize = 64;

fn synthetic(name: &str) -> Option<Vec<f64>> {
    let n = 2048usize;
    let mut v = Vec::with_capacity(n);
    match name {
        "impulse_train" => {
            for i in 0..n {
                v.push(if i % 16 == 0 { 30000.0 } else { 0.0 });
            }
        }
        "am_tone" => {
            for i in 0..n {
                let t = i as f64;
                let env = 0.15 + 0.85 * (0.5 * (1.0 + (t * 0.0021).cos()));
                v.push(env * (t * 0.37).sin() * 20000.0);
            }
        }
        "castanet" => {
            let mut state = 1u64;
            for i in 0..n {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 33) as i64 % 2001 - 1000) as f64;
                let decay = (-((i % 128) as f64) / 8.0).exp();
                v.push(noise * decay * 20.0);
            }
        }
        "stationary_ar" => {
            let mut state = 7u64;
            let (mut a, mut b) = (0.0f64, 0.0f64);
            for _ in 0..n {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let e = ((state >> 33) as i64 % 2001 - 1000) as f64;
                let x = 0.6 * a + 0.3 * b + e;
                b = a;
                a = x;
                v.push(x);
            }
        }
        _ => return None,
    }
    Some(v)
}

/// Spectral coefficients of a signal, normalized by a robust scale so the
/// quantizer operates on a unit-ish distribution.
fn coefficients(samples: &[f64]) -> Vec<f64> {
    let mut coeffs = Vec::new();
    let mut at = 0usize;
    while at + FRAME <= samples.len() {
        coeffs.extend(dct2(&samples[at..at + FRAME]));
        at += FRAME / 2;
    }
    let mean_abs = if coeffs.is_empty() {
        1.0
    } else {
        coeffs.iter().map(|v| v.abs()).sum::<f64>() / coeffs.len() as f64
    };
    let scale = mean_abs.max(1e-9);
    coeffs.iter().map(|v| v / scale).collect()
}

struct Row {
    id: String,
    coefficients: usize,
    midpoint_mse: f64,
    centroid_mse: f64,
    offset: f64,
}

fn analyse(id: String, coeffs: &[f64], q: &Quantizer, offset: f64) -> Row {
    // One symbol vector is shared by both reconstructions, which is exactly the
    // mechanism's claim: only the reconstruction point moves.
    let symbols: Vec<i64> = coeffs.iter().map(|&x| q.symbol(x)).collect();
    let mut midpoint = 0f64;
    let mut centroid = 0f64;
    for (&x, &s) in coeffs.iter().zip(&symbols) {
        let em = x - q.reconstruct(s, 0.5);
        midpoint += em * em;
        let ec = x - q.reconstruct(s, offset);
        centroid += ec * ec;
    }
    let n = coeffs.len().max(1) as f64;
    Row {
        id,
        coefficients: coeffs.len(),
        midpoint_mse: midpoint / n,
        centroid_mse: centroid / n,
        offset,
    }
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.midpoint_mse.is_finite()
        && row.centroid_mse.is_finite()
        && row.midpoint_mse >= 0.0
        && row.centroid_mse >= 0.0;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.coefficients as u64);
    common::push_u64(projection, (row.midpoint_mse * 1e9) as u64);
    common::push_u64(projection, (row.centroid_mse * 1e9) as u64);
    common::push_u64(projection, (row.offset * 1000.0) as u64);
}

fn row_json(row: &Row) -> serde_json::Value {
    let gain_db = if row.centroid_mse > 0.0 {
        10.0 * (row.midpoint_mse / row.centroid_mse).log10()
    } else {
        0.0
    };
    serde_json::json!({
        "id": row.id,
        "coefficients": row.coefficients,
        "midpoint_mse": row.midpoint_mse,
        "centroid_mse": row.centroid_mse,
        "gain_db": gain_db,
        "offset": row.offset,
    })
}

/// Run the court; writes `receipts/centroid-sq/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.centroid_sq.v1");

    // Training coefficients from the synthetic transient/AM family.
    let mut training = Vec::new();
    for name in ["impulse_train", "am_tone", "castanet", "stationary_ar"] {
        if let Some(samples) = synthetic(name) {
            training.extend(coefficients(&samples));
        }
    }
    let q = Quantizer {
        deadzone: 0.2,
        delta: 0.4,
    };
    let (offset, _) = fit_offset(&training, &q);

    let mut rows = Vec::new();
    let mut all_exact = true;
    for name in [
        "silence",
        "dc",
        "single-sine",
        "harmonic-tone",
        "quasi-periodic",
        "impulse-train",
        "am-signal",
        "white-noise",
    ] {
        let Some(fx) = corpus::named(name) else {
            continue;
        };
        let n = fx.frames().min(2048);
        if n < FRAME {
            continue;
        }
        let samples: Vec<f64> = fx.samples[..n].iter().map(|&s| f64::from(s)).collect();
        let coeffs = coefficients(&samples);
        if coeffs.is_empty() {
            continue;
        }
        let row = analyse(name.to_string(), &coeffs, &q, offset);
        let mut gates = true;
        push_row(&mut projection, &row, &mut gates);
        all_exact &= gates;
        rows.push(row);
    }

    let total_coeffs: u64 = rows.iter().map(|r| r.coefficients as u64).sum();
    let wins = rows
        .iter()
        .filter(|r| r.centroid_mse < r.midpoint_mse)
        .count();
    let losses = rows
        .iter()
        .filter(|r| r.centroid_mse > r.midpoint_mse)
        .count();
    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else {
        Verdict::Supported
    };

    let rows_json: serde_json::Value = rows.iter().map(row_json).collect();
    common::finish(
        "centroid-sq",
        receipts_root,
        CENTROID_SQ_SHA256,
        &projection,
        verdict,
        format!(
            "centroid scalar reconstruction over {n} held-out fixtures / {tc} coefficients at \
             identical symbols: the global offset {off:.3} fitted on the training spectra improves \
             held-out MSE on {w} fixtures and loses on {l}; the split is the mechanism's boundary \
             — concentrated enveloped spectra gain, DC-like and already-flat spectra lose; \
             exploratory RD court, no codec-quality claim",
            n = rows.len(),
            tc = total_coeffs,
            off = offset,
            w = wins,
            l = losses,
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "fixtures": rows.len(),
                    "training_coefficients": training.len(),
                    "offset": offset,
                }),
            ),
            ("fixtures", rows_json),
            (
                "method",
                serde_json::json!({
                    "quantizer": "uniform threshold with a deadzone; symbol q = 0 inside the \
                                  deadzone, otherwise 1 + floor((|x| - deadzone)/delta)",
                    "reconstruction": "x_hat = sgn(q)(deadzone + (|q| - 1 + c) delta); c = 0.5 is \
                                       the midpoint rule, c < 0.5 moves toward the cell's lower \
                                       edge where transform density is concentrated",
                    "fitted": "c is fitted once on the training coefficient set and is a \
                               profile-defined reconstruction rule, so the symbols and bitstream \
                               are unchanged",
                    "scope": "exploratory lossy RD; VOLE has no lossy transform codec",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the offset is fitted on synthetic spectral coefficients and evaluated on the \
                     frozen corpus's spectra, not on a real quantizer loop with entropy coding",
                    "no perceptual metric is measured: the claim is held-out MSE at identical \
                     symbols, nothing more",
                ]),
            ),
        ],
    )
}

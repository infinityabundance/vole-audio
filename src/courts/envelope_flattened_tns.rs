//! `court envelope-flattened-tns` — Phase 6 mechanism 13
//! (`EnvelopeFlattenedTNS`, lossy/exploratory).
//!
//! A low-order spectral predictor estimated directly on a transform's spectrum
//! spends its degrees of freedom fitting the gross spectral envelope, which the
//! quantizer's scalefactors already carry, instead of the fine structure it is
//! meant to shape. This court estimates the same predictor twice — once on the
//! raw spectrum, once after flattening the spectrum by a smoothed envelope — and
//! measures both on the flattened target, where the comparison is meaningful.
//!
//! VOLE has no lossy transform codec, so this is explicitly an **exploratory
//! spectral-estimator** court: it reports frames, per-frame wins and prediction
//! gain, and gates only on the computations being finite and well-formed. It
//! makes no codec-quality or bitrate claim.

use crate::courts::learned_common as common;
use crate::entropy::corpus;
use crate::error::Result;
use crate::learned::tns::envelope_flattened_tns;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const ENVELOPE_FLATTENED_TNS_SHA256: &str =
    "2b86059edaf2a483eb11dfcea662d6e3244be6e0f60b18702fa6feb65d14b2ef";

/// Frame length and hop for the analysis transform.
const FRAME: usize = 64;
const HOP: usize = 32;
/// TNS predictor order and envelope smoothing radius.
const ORDER: usize = 4;
const RADIUS: usize = 3;

struct Row {
    id: String,
    frames: usize,
    flattened_wins: usize,
    raw_wins: usize,
    mean_gain_ppm: u64,
    finite: bool,
}

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
                let burst = (i % 128) as f64;
                let decay = (-burst / 8.0).exp();
                v.push(noise * decay * 20.0);
            }
        }
        "stationary_ar" => {
            let mut state = 7u64;
            let mut a = 0.0f64;
            let mut b = 0.0f64;
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

fn analyse(id: String, samples: &[f64]) -> Result<Row> {
    let mut frames = 0usize;
    let mut flattened_wins = 0usize;
    let mut raw_wins = 0usize;
    let mut gain_sum = 0f64;
    let mut finite = true;
    let mut at = 0usize;
    while at + FRAME <= samples.len() {
        let frame = &samples[at..at + FRAME];
        match envelope_flattened_tns(frame, ORDER, RADIUS) {
            Some(o)
                if o.raw_estimated.is_finite()
                    && o.flattened_estimated.is_finite()
                    && o.raw_estimated >= 0.0
                    && o.flattened_estimated >= 0.0 =>
            {
                frames += 1;
                if o.flattened_estimated < o.raw_estimated {
                    flattened_wins += 1;
                } else if o.raw_estimated < o.flattened_estimated {
                    raw_wins += 1;
                }
                let g = o.gain();
                if g.is_finite() {
                    gain_sum += g.min(1000.0);
                }
            }
            Some(_) => finite = false,
            // A degenerate frame (all-zero spectrum, silent input) has no
            // predictor to estimate; it is skipped, not a correctness failure.
            None => {}
        }
        at += HOP;
    }
    let mean_gain_ppm = if frames == 0 {
        0
    } else {
        ((gain_sum / frames as f64) * 1_000_000.0) as u64
    };
    Ok(Row {
        id,
        frames,
        flattened_wins,
        raw_wins,
        mean_gain_ppm,
        finite,
    })
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.finite;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.frames as u64);
    common::push_u64(projection, row.flattened_wins as u64);
    common::push_u64(projection, row.raw_wins as u64);
    common::push_u64(projection, row.mean_gain_ppm);
}

fn row_json(row: &Row) -> serde_json::Value {
    serde_json::json!({
        "id": row.id,
        "frames": row.frames,
        "flattened_wins": row.flattened_wins,
        "raw_wins": row.raw_wins,
        "mean_gain_ppm": row.mean_gain_ppm,
        "finite": row.finite,
    })
}

/// Run the court; writes `receipts/envelope-flattened-tns/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.envelope_flattened_tns.v1");

    let mut rows = Vec::new();
    let mut all_exact = true;
    for name in ["impulse_train", "am_tone", "castanet", "stationary_ar"] {
        let Some(samples) = synthetic(name) else {
            continue;
        };
        let row = analyse(name.to_string(), &samples)?;
        let mut gates = true;
        push_row(&mut projection, &row, &mut gates);
        all_exact &= gates;
        rows.push(row);
    }
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
        if n == 0 {
            continue;
        }
        let samples: Vec<f64> = fx.samples[..n].iter().map(|&s| f64::from(s)).collect();
        let row = analyse(name.to_string(), &samples)?;
        let mut gates = true;
        push_row(&mut projection, &row, &mut gates);
        all_exact &= gates;
        rows.push(row);
    }

    let total_frames: u64 = rows.iter().map(|r| r.frames as u64).sum();
    let total_flat_wins: u64 = rows.iter().map(|r| r.flattened_wins as u64).sum();
    let total_raw_wins: u64 = rows.iter().map(|r| r.raw_wins as u64).sum();
    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else {
        Verdict::Supported
    };

    let rows_json: serde_json::Value = rows.iter().map(row_json).collect();
    common::finish(
        "envelope-flattened-tns",
        receipts_root,
        ENVELOPE_FLATTENED_TNS_SHA256,
        &projection,
        verdict,
        format!(
            "envelope-flattened spectral predictor estimation over {n} fixtures / {tf} frames: \
             the flattened estimator beats the raw estimator on {fw} frames and loses on {rw}, \
             all computations finite; exploratory lossy estimator court, no quality claim",
            n = rows.len(),
            tf = total_frames,
            fw = total_flat_wins,
            rw = total_raw_wins,
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "fixtures": rows.len(),
                }),
            ),
            ("fixtures", rows_json),
            (
                "method",
                serde_json::json!({
                    "estimator": "an order-4 least-squares causal predictor over the spectrum, \
                                  applied causally to shape the fine structure",
                    "mechanism": "smooth |X[k]| into an envelope A[k], flatten Y[k] = X[k] / A[k], \
                                  estimate the predictor from Y, and measure it on the flattened \
                                  target where the comparison is meaningful",
                    "contrast": "the raw estimator spends its few degrees of freedom fitting the \
                                 gross envelope the scalefactors already carry",
                    "scope": "exploratory lossy spectral estimator; VOLE has no lossy transform \
                              codec, so this is not a production path",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the predictor is estimated and measured on a DCT-II analysis transform, not a \
                     real MDCT quantizer loop",
                    "no reconstruction quality, pre-echo or bitrate is measured: this court gates \
                     estimator well-formedness and reports prediction energy",
                ]),
            ),
        ],
    )
}

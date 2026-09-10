//! Long-term (pitch) prediction fitting (Exp2, priority `4`).
//!
//! A short-term sparse predictor is fitted first; the long-term stage is then
//! fitted over its residual `E_short`. Lag proposals come from a deterministic
//! autocorrelation-peak scan of `E_short`; gains are least-squares and are
//! quantized to Q12. Only the measured canonical winner is returned.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::arithmetic::quantize_weight;
use crate::learned::ltp::LongTermPredictor;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::train::sparse::fit_sparse_object;
use crate::learned::train::{TrainBudget, TrainStats};

/// Long-term tap counts searched (a 3-tap and a 5-tap variant).
pub const LTP_TAP_COUNTS: [usize; 2] = [3, 5];

/// Candidate pitch lags from autocorrelation peaks of the short-term residual.
fn lag_candidates(e: &[i32], min_lag: usize, max_lag: usize) -> Vec<usize> {
    let n = e.len();
    let hi = max_lag.min(n.saturating_sub(1));
    let lo = min_lag.max(1);
    if lo >= hi {
        return Vec::new();
    }
    let mean = e.iter().map(|&v| f64::from(v)).sum::<f64>() / n.max(1) as f64;
    let energy: f64 = e
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        .max(1.0);
    let mut r = vec![0f64; hi + 1];
    for lag in lo..=hi {
        let mut acc = 0f64;
        for t in lag..n {
            acc += (f64::from(e[t]) - mean) * (f64::from(e[t - lag]) - mean);
        }
        r[lag] = acc / energy;
    }
    let mut peaks: Vec<usize> = Vec::new();
    for lag in (lo + 1)..hi {
        if r[lag] > r[lag - 1] && r[lag] >= r[lag + 1] && r[lag] > 0.05 {
            peaks.push(lag);
        }
    }
    peaks.sort_by(|&a, &b| r[b].partial_cmp(&r[a]).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(16);
    peaks.sort_unstable();
    peaks
}

#[allow(clippy::needless_range_loop)]
fn fit_gains(e: &[i32], lag: usize, taps: usize) -> Option<Vec<f64>> {
    let n = e.len();
    if lag + 1 > n || taps > lag {
        return None;
    }
    // Features are e[t - (lag - j)] for j in 0..taps.
    let nv = taps;
    let mut a = vec![vec![0f64; nv]; nv];
    let mut b = vec![0f64; nv];
    for t in lag..n {
        for i in 0..taps {
            let fi = f64::from(e[t - (lag - i)]);
            for j in 0..=i {
                a[i][j] += fi * f64::from(e[t - (lag - j)]);
            }
            b[i] += fi * f64::from(e[t]);
        }
    }
    for i in 0..nv {
        for j in 0..i {
            a[j][i] = a[i][j];
        }
    }
    let scale = (0..nv).map(|i| a[i][i].abs()).fold(0f64, f64::max).max(1.0);
    for i in 0..nv {
        a[i][i] += 1e-6 + 1e-9 * scale;
    }
    solve(&a, &b)
}

#[allow(clippy::needless_range_loop)]
fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut rr = r.clone();
            rr.push(b[i]);
            rr
        })
        .collect();
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col][col].abs();
        for r in col + 1..n {
            if m[r][col].abs() > best {
                best = m[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        let d = m[col][col];
        for r in col + 1..n {
            let f = m[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for cc in col..=n {
                m[r][cc] -= f * m[col][cc];
            }
        }
    }
    let mut x = vec![0f64; n];
    for i in (0..n).rev() {
        let mut s = m[i][n];
        for j in i + 1..n {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(x)
}

/// Fit a long-term predictor object over `source`.
pub fn fit_ltp_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let n = frames as usize;
    if source.len() != n {
        return Err(Error::malformed("long-term fit geometry mismatch"));
    }
    let block = None;
    let (short_obj, short_stats) =
        fit_sparse_object(source, frames, sample_rate_hz, 8, block, budget)?;
    stats.merge(&short_stats);
    let LearnedModel::SparseLinear(short) = &short_obj.model else {
        return Err(Error::internal("sparse fit did not produce a sparse model"));
    };
    let short = short.clone();
    let h_short = short.hypothesis_all_from_source(source, n)?;
    let e: Vec<i32> = source
        .iter()
        .zip(h_short.iter())
        .map(|(&x, &h)| (i64::from(x) - i64::from(h)) as i32)
        .collect();
    let max_lag = (n / 2).clamp(2, 4096);
    let candidates = lag_candidates(&e, 2, max_lag);
    let mut best: Option<(LongTermPredictor, u64)> = None;
    for &lag in &candidates {
        for &taps in &LTP_TAP_COUNTS {
            stats.candidates += 1;
            let Some(gains) = fit_gains(&e, lag, taps) else {
                continue;
            };
            let p = LongTermPredictor {
                channels: 1,
                short: short.clone(),
                lag: lag as u16,
                gains: gains.iter().map(|&g| quantize_weight(g)).collect(),
                block_frames: block,
            };
            if p.validate().is_err() {
                continue;
            }
            let o = match LearnedObject::from_intrinsic_exp2(
                LearnedModel::LongTerm(p.clone()),
                1,
                frames,
                sample_rate_hz,
                Vec::new(),
                source,
            ) {
                Ok(o) => o,
                Err(_) => {
                    stats.rejected += 1;
                    continue;
                }
            };
            if !o.verify(source) {
                stats.rejected += 1;
                continue;
            }
            let bytes = crate::learned::accounting::LearnedCost::of(&o)
                .map(|c| c.complete_bytes)
                .unwrap_or(u64::MAX);
            if best.as_ref().is_none_or(|(_, bb)| bytes < *bb) {
                best = Some((p, bytes));
            }
        }
    }
    let (predictor, _) =
        best.ok_or_else(|| Error::internal("long-term fit produced no candidate"))?;
    let object = LearnedObject::from_intrinsic_exp2(
        LearnedModel::LongTerm(predictor),
        1,
        frames,
        sample_rate_hz,
        Vec::new(),
        source,
    )?;
    stats.quantization_attempts += 1;
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((object, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltp_fit_closes_and_finds_the_period() {
        let mut x = vec![0i32; 64];
        for t in 64..3000 {
            let lp = x[t - 23] as i64;
            let v = (x[t - 1] as i64 * 2) / 5 + (lp * 55) / 100 + ((t % 11) as i64 - 5) * 2;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let budget = TrainBudget::default();
        let (o, _) = fit_ltp_object(&x, 3000, 48_000, &budget).unwrap();
        assert!(o.verify(&x));
        let LearnedModel::LongTerm(p) = &o.model else {
            panic!("expected long term");
        };
        assert!(p.lag.abs_diff(23) <= 1, "lag = {}", p.lag);
    }
}

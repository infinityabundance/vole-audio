//! Sparse linear fitting by deterministic forward selection (Exp2, priority `3`).
//!
//! The fit is *proposal* machinery (training is non-normative). It selects lags
//! greedily by how much they reduce the modelling residual, then the canonical
//! quantized predictor is measured by its real stored bytes, and only the
//! measured winner is returned. No automatic differentiation is used.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::arithmetic::{quantize_bias, quantize_weight};
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::profile::LearnedProfile;
use crate::learned::sparse::SparseLinearPredictor;
use crate::learned::train::{TrainBudget, TrainStats};

/// The fixed low-order tail of the candidate lag dictionary.
pub const BASE_LAG_LIMIT: u16 = 32;

/// Build a deterministic candidate lag dictionary bounded by `max_lag`.
pub fn candidate_lags(source: &[i32], max_lag: u16) -> Vec<u16> {
    let n = source.len();
    let cap = usize::from(max_lag).min(n.saturating_sub(1)).max(1);
    let mut set = vec![false; cap + 1];
    let base = usize::from(BASE_LAG_LIMIT).min(cap);
    set[1..=base].fill(true);
    // Autocorrelation peaks (local maxima of the normalized autocorrelation).
    if n > 8 {
        let mean = source.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let energy: f64 = source
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            .max(1.0);
        let mut r = vec![0f64; cap + 1];
        for lag in 1..=cap {
            let mut acc = 0f64;
            for t in lag..n {
                acc += (f64::from(source[t]) - mean) * (f64::from(source[t - lag]) - mean);
            }
            r[lag] = acc / energy;
        }
        for lag in 2..cap {
            if r[lag] > r[lag - 1] && r[lag] > r[lag + 1] && r[lag] > 0.1 {
                set[lag] = true;
            }
        }
    }
    (1..=cap).filter(|&l| set[l]).map(|l| l as u16).collect()
}

/// Ridge fit on an explicit lag set (training-only float math).
#[allow(clippy::needless_range_loop)]
fn fit_lags(source: &[i32], lags: &[u16], lambda: f64) -> (Vec<f64>, f64) {
    let n = source.len();
    let k = lags.len();
    let nv = k + 1;
    let mut a = vec![vec![0f64; nv]; nv];
    let mut b = vec![0f64; nv];
    let mut max_lag = 0usize;
    for &l in lags {
        max_lag = max_lag.max(usize::from(l));
    }
    for t in max_lag..n {
        for (i, &l) in lags.iter().enumerate() {
            let f = f64::from(source[t - usize::from(l)]);
            for j in 0..=i {
                a[i][j] += f * f64::from(source[t - usize::from(lags[j])]);
            }
            a[k][i] += f;
            b[i] += f * f64::from(source[t]);
        }
        b[k] += f64::from(source[t]);
        a[k][k] += 1.0;
    }
    for i in 0..nv {
        for j in 0..i {
            a[j][i] = a[i][j];
        }
    }
    let scale = (0..nv).map(|i| a[i][i].abs()).fold(0f64, f64::max).max(1.0);
    for i in 0..k {
        a[i][i] += lambda + 1e-9 * scale;
    }
    a[k][k] += 1e-12 * scale;
    let sol = solve(&a, &b).unwrap_or_else(|| vec![0.0; nv]);
    let weights = sol[..k].to_vec();
    (weights, sol[k])
}

/// Residual sum of squares of a fitted model over `source`.
fn sse(source: &[i32], lags: &[u16], weights: &[f64], bias: f64) -> f64 {
    let n = source.len();
    let max_lag = lags.iter().map(|&l| usize::from(l)).max().unwrap_or(0);
    let mut acc = 0f64;
    for t in max_lag..n {
        let mut pred = bias;
        for (i, &l) in lags.iter().enumerate() {
            pred += weights[i] * f64::from(source[t - usize::from(l)]);
        }
        let d = f64::from(source[t]) - pred;
        acc += d * d;
    }
    acc
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

fn build_predictor(
    lags: &[u16],
    weights: &[f64],
    bias: f64,
    block_frames: Option<u32>,
) -> SparseLinearPredictor {
    SparseLinearPredictor {
        channels: 1,
        lags: lags.to_vec(),
        weights: weights.iter().map(|&w| quantize_weight(w)).collect(),
        bias: vec![quantize_bias(bias)],
        block_frames,
        coeff_encoding: crate::learned::sparse::COEFF_DELTA_VARINT,
    }
}

/// Fit a sparse linear object by forward selection over the lag dictionary,
/// returning the cheapest canonical candidate and its statistics.
pub fn fit_sparse_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    max_lags: u16,
    block_frames: Option<u32>,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    let sw = Stopwatch::start();
    let mut stats = TrainStats {
        candidates: 0,
        ..Default::default()
    };
    let n = frames as usize;
    if source.len() != n {
        return Err(Error::malformed("sparse fit geometry mismatch"));
    }
    let dictionary = candidate_lags(source, budget.max_taps as u16);
    if dictionary.is_empty() {
        return Err(Error::internal("sparse lag dictionary is empty"));
    }
    let max_lags = usize::from(max_lags).min(dictionary.len()).max(1);
    let mut selected: Vec<u16> = Vec::new();
    let mut remaining: Vec<u16> = dictionary;
    let mut best_sse = f64::INFINITY;
    for _ in 0..max_lags {
        let mut best_lag: Option<u16> = None;
        let mut best_candidate = best_sse;
        let mut best_fit: Option<(Vec<f64>, f64)> = None;
        for (idx, &lag) in remaining.iter().enumerate() {
            stats.candidates += 1;
            let mut trial = selected.clone();
            trial.push(lag);
            trial.sort_unstable();
            let (w, b) = fit_lags(source, &trial, budget.ridge_lambda);
            let s = sse(source, &trial, &w, b);
            if s + 1e-9 < best_candidate || (best_lag.is_none() && best_fit.is_none()) {
                best_candidate = s;
                best_lag = Some(lag);
                best_fit = Some((w, b));
                let _ = idx;
            }
        }
        let Some(lag) = best_lag else { break };
        if best_candidate >= best_sse {
            break;
        }
        best_sse = best_candidate;
        selected.push(lag);
        selected.sort_unstable();
        remaining.retain(|&l| l != lag);
        if remaining.is_empty() {
            break;
        }
        // Quantize-and-measure the current prefix immediately (proposal only).
    }
    if selected.is_empty() {
        return Err(Error::internal("sparse selection produced no lags"));
    }
    // Measure every prefix of the selected order by real canonical bytes.
    let mut best: Option<(SparseLinearPredictor, u64)> = None;
    for k in 1..=selected.len() {
        let mut prefix = selected[..k].to_vec();
        prefix.sort_unstable();
        let (w, b) = fit_lags(source, &prefix, budget.ridge_lambda);
        let p = build_predictor(&prefix, &w, b, block_frames);
        if p.validate().is_err() {
            continue;
        }
        let o = match LearnedObject::from_intrinsic_exp2(
            LearnedModel::SparseLinear(p.clone()),
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
    let (predictor, _) = best.ok_or_else(|| Error::internal("sparse fit produced no candidate"))?;
    // Optimizer v2: refine the quantized coefficients against the *actual*
    // canonical complete size (including the best v2 residual codec) with a
    // deterministic multiscale beam. The seed is always retained, so this can
    // never enlarge the object.
    stats.candidates += 1;
    let refined = refine_sparse_coefficients(
        &predictor,
        source,
        frames,
        sample_rate_hz,
        budget.max_iterations.min(512),
    );
    let predictor = match refined {
        Some((p, _)) => p,
        None => predictor,
    };
    let object = LearnedObject::from_intrinsic_exp2(
        LearnedModel::SparseLinear(predictor),
        1,
        frames,
        sample_rate_hz,
        Vec::new(),
        source,
    )?;
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    stats.quantization_attempts += 1;
    let _ = LearnedProfile::Exp2;
    Ok((object, stats))
}

/// Refine a sparse predictor's quantized coefficients against the actual
/// canonical complete size using the Exp2 multiscale beam optimizer. Returns
/// the candidate and its measured bytes, or `None` when nothing closes.
fn refine_sparse_coefficients(
    base: &SparseLinearPredictor,
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    max_evaluations: u64,
) -> Option<(SparseLinearPredictor, u64)> {
    let measure = |weights: &[i16]| -> u64 {
        let mut p = base.clone();
        p.weights = weights.to_vec();
        if p.validate().is_err() {
            return u64::MAX;
        }
        match LearnedObject::from_intrinsic_exp2(
            LearnedModel::SparseLinear(p),
            1,
            frames,
            sample_rate_hz,
            Vec::new(),
            source,
        )
        .and_then(|o| crate::learned::accounting::LearnedCost::of(&o))
        {
            Ok(c) => c.complete_bytes,
            Err(_) => u64::MAX,
        }
    };
    let seed = base.weights.clone();
    let base_bytes = measure(&seed);
    if base_bytes == u64::MAX {
        return None;
    }
    let result = crate::learned::train::optimizer2::beam_coordinate_descent(
        &seed,
        4,
        max_evaluations.max(1),
        |w| measure(w) as f64,
    );
    let final_bytes = measure(&result.best);
    if final_bytes <= base_bytes {
        let mut p = base.clone();
        p.weights = result.best;
        Some((p, final_bytes))
    } else {
        Some((base.clone(), base_bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_selection_recovers_a_sparse_ar_model() {
        // X[t] = 0.6 X[t-3] + 0.3 X[t-11] + drive.
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut x = vec![0i32; 12];
        for t in 12..4096 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let drive = (s >> 40) as i64 % 2001 - 1000;
            let v = (x[t - 3] as i64 * 6) / 10 + (x[t - 11] as i64 * 3) / 10 + drive;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let budget = TrainBudget::default();
        let (o, stats) = fit_sparse_object(&x, 4096, 48_000, 4, None, &budget).unwrap();
        assert!(o.verify(&x));
        assert!(stats.candidates > 0);
        let LearnedModel::SparseLinear(p) = &o.model else {
            panic!("expected sparse linear");
        };
        assert!(p.lags.contains(&3), "lags = {:?}", p.lags);
        assert!(p.lags.contains(&11), "lags = {:?}", p.lags);
    }

    #[test]
    fn candidate_dictionary_is_sorted_and_bounded() {
        let x: Vec<i32> = (0..2048).map(|i| ((i * 131) % 997) - 498).collect();
        let lags = candidate_lags(&x, 256);
        assert!(lags.windows(2).all(|w| w[0] < w[1]));
        assert!(lags.iter().all(|&l| (1..=256u16).contains(&l)));
    }
}

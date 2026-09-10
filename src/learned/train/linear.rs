//! Ridge/least-squares fitting for the linear finite-field family (`O.17`, `O.18`).
//!
//! The fit is ordinary regularized least squares on the causal tap window,
//! solved through the normal equations with partial-pivot Gaussian elimination.
//! It uses floating point (training is disposable); the returned predictor is
//! always the **quantized canonical** form, and closure is verified against the
//! quantized evaluator, never against the float model.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::object::LearnedObject;
use crate::learned::quantize::compile_linear;
use crate::learned::train::{TrainBudget, TrainStats};

/// Real-valued normal-equation solution.
pub struct LinearFit {
    /// `K · C · C` weights in evaluator indexing.
    pub weights: Vec<f64>,
    /// `C` biases.
    pub bias: Vec<f64>,
}

/// Fit a causal FIR by ridge regression over at most `analysis_frames`.
///
/// Block-local predictors reset their history at block boundaries, so the
/// design matrix encodes exactly the history the canonical evaluator will see.
#[allow(clippy::needless_range_loop)]
pub fn fit_ridge(
    source: &[i32],
    channels: u8,
    frames: usize,
    taps: u16,
    block_frames: Option<u32>,
    analysis_frames: usize,
    lambda: f64,
) -> Result<LinearFit> {
    let c = usize::from(channels);
    if c == 0 || source.len() != frames * c {
        return Err(Error::malformed("ridge fit geometry mismatch"));
    }
    let k = usize::from(taps);
    if k == 0 || u32::from(taps) > crate::limits::MAX_LEARNED_TAPS {
        return Err(Error::limit("ridge fit tap count exceeds the bound"));
    }
    let n_feat = k * c;
    let n_var = n_feat + 1; // + bias
    let use_frames = frames.min(analysis_frames.max(1));

    let mut a = vec![vec![0f64; n_var]; n_var];
    let mut b = vec![vec![0f64; c]; n_var];
    let mut row = vec![0f64; n_var];

    for t in 0..use_frames {
        let block_start = match block_frames {
            Some(bs) => {
                let bs = bs as usize;
                (t / bs) * bs
            }
            None => 0,
        };
        for kk in 1..=k {
            let idx = t as i64 - kk as i64;
            let feat = (kk - 1) * c;
            for i in 0..c {
                row[feat + i] = if idx >= block_start as i64 && idx >= 0 {
                    f64::from(source[idx as usize * c + i])
                } else {
                    0.0
                };
            }
        }
        row[n_feat] = 1.0;
        for j in 0..n_var {
            let fj = row[j];
            if fj == 0.0 {
                continue;
            }
            for o in 0..c {
                b[j][o] += fj * f64::from(source[t * c + o]);
            }
            for jj in 0..=j {
                a[j][jj] += fj * row[jj];
            }
        }
    }
    // Symmetric completion.
    for j in 0..n_var {
        for jj in 0..j {
            a[jj][j] = a[j][jj];
        }
    }
    // Ridge plus a relative jitter, so a degenerate feature column (an
    // all-zero or near-collinear tap window) stays numerically solvable. The
    // jitter is scaled by the matrix trace and is training-only.
    let scale = (0..n_var)
        .map(|j| a[j][j].abs())
        .fold(0.0f64, f64::max)
        .max(1.0);
    for j in 0..n_feat {
        a[j][j] += lambda + 1e-9 * scale;
    }
    a[n_feat][n_feat] += 1e-12 * scale;

    let mut weights = vec![0f64; k * c * c];
    let mut bias = vec![0f64; c];
    for o in 0..c {
        let sol = solve(&a, &b_column(&b, o, n_var))
            .ok_or_else(|| Error::internal("ridge normal equations are singular"))?;
        for kk in 1..=k {
            for i in 0..c {
                weights[((kk - 1) * c + o) * c + i] = sol[(kk - 1) * c + i];
            }
        }
        bias[o] = sol[n_feat];
    }
    Ok(LinearFit { weights, bias })
}

fn b_column(b: &[Vec<f64>], o: usize, n: usize) -> Vec<f64> {
    (0..n).map(|j| b[j][o]).collect()
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, row) in a.iter().enumerate() {
        let mut r = row.clone();
        r.push(b[i]);
        m.push(r);
    }
    for col in 0..n {
        // Pivot.
        let mut piv = col;
        let mut best = m[col][col].abs();
        for r in col + 1..n {
            let v = m[r][col].abs();
            if v > best {
                best = v;
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

/// Fit and compile a canonical linear object, verifying exact closure.
#[allow(clippy::too_many_arguments)]
pub fn fit_linear_object(
    source: &[i32],
    channels: u8,
    frames: u64,
    sample_rate_hz: u32,
    taps: u16,
    block_frames: Option<u32>,
    analysis_frames: usize,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    let sw = Stopwatch::start();
    let mut stats = TrainStats {
        candidates: 1,
        ..Default::default()
    };
    let fit = fit_ridge(
        source,
        channels,
        frames as usize,
        taps,
        block_frames,
        analysis_frames,
        budget.ridge_lambda,
    )?;
    stats.quantization_attempts += 1;
    let predictor = compile_linear(channels, taps, block_frames, &fit.weights, &fit.bias);
    let object = LearnedObject::from_intrinsic(
        crate::learned::model::LearnedModel::Linear(predictor),
        channels,
        frames,
        sample_rate_hz,
        Vec::new(),
        source,
    )?;
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((object, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::model::LearnedModel;

    #[test]
    fn ridge_recovers_a_known_fir_and_closes_exactly() {
        // An excited AR(2): X[t] = 0.5 X[t-1] + 0.25 X[t-2] + small drive.
        // X[0] = X[1] = 0 keeps the zero-history boundary consistent with the
        // canonical evaluator, so the fit is not dominated by a transient.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut x = vec![0i32, 0];
        for t in 2..4096 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let drive = (s >> 40) as i64 % 2001 - 1000;
            let v = i64::from(x[t - 1]) / 2 + i64::from(x[t - 2]) / 4 + drive;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let budget = TrainBudget::default();
        let (o, stats) = fit_linear_object(&x, 1, 4096, 48_000, 2, None, 4096, &budget).unwrap();
        assert!(stats.candidates >= 1);
        assert!(o.verify(&x));
        let LearnedModel::Linear(p) = &o.model else {
            panic!("expected linear");
        };
        // The fitted taps are close to 0.5 and 0.25 (within quantization and
        // the finite drive).
        let w0 = f64::from(p.weights[0]) / 4096.0;
        let w1 = f64::from(p.weights[1]) / 4096.0;
        assert!((w0 - 0.5).abs() < 0.05, "w0 = {w0}");
        assert!((w1 - 0.25).abs() < 0.05, "w1 = {w1}");
    }

    #[test]
    fn block_local_fit_closes_exactly() {
        let x: Vec<i32> = (0..600).map(|i| ((i * 331) % 9001) - 4500).collect();
        let budget = TrainBudget::default();
        let (o, _) = fit_linear_object(&x, 1, 600, 48_000, 3, Some(128), 600, &budget).unwrap();
        assert!(o.verify(&x));
    }
}

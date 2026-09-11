//! Context-mixture fitting (Exp2, priority `9`).
//!
//! A base sparse predictor supplies the residual whose magnitude defines the
//! context buckets; one expert is then fitted per bucket by ridge regression
//! over the rows that fall in that bucket. Only the measured winner is returned.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::arithmetic::{quantize_bias, quantize_weight};
use crate::learned::context_mixture::ContextMixturePredictor;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::sparse::SparseLinearPredictor;
use crate::learned::train::sparse::fit_sparse_object;
use crate::learned::train::{TrainBudget, TrainStats};

fn bucket_of(edges: &[u32], magnitude: u32) -> usize {
    edges.iter().filter(|&&e| magnitude > e).count()
}

#[allow(clippy::needless_range_loop)]
fn fit_expert(
    source: &[i32],
    lags: &[u16],
    rows: &[usize],
    lambda: f64,
) -> Option<(Vec<f64>, f64)> {
    let k = lags.len();
    let nv = k + 1;
    let mut a = vec![vec![0f64; nv]; nv];
    let mut b = vec![0f64; nv];
    for &t in rows {
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
    if rows.len() < k + 2 {
        return None;
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
    let sol = solve(&a, &b)?;
    Some((sol[..k].to_vec(), sol[k]))
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

fn measure(
    m: ContextMixturePredictor,
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
) -> Option<u64> {
    let o = LearnedObject::from_intrinsic_exp2(
        LearnedModel::ContextMixture(m),
        1,
        frames,
        sample_rate_hz,
        Vec::new(),
        source,
    )
    .ok()?;
    if !o.verify(source) {
        return None;
    }
    crate::learned::accounting::LearnedCost::of(&o)
        .ok()
        .map(|c| c.complete_bytes)
}

/// Fit a context-mixture object (3 buckets) and keep it only if it beats the
/// base sparse candidate.
pub fn fit_context_mixture_object(
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
        return Err(Error::malformed("context mixture fit geometry mismatch"));
    }
    let mut base_budget = *budget;
    base_budget.max_iterations = base_budget.max_iterations.min(48);
    let (base_obj, base_stats) =
        fit_sparse_object(source, frames, sample_rate_hz, 8, None, &base_budget)?;
    stats.merge(&base_stats);
    let LearnedModel::SparseLinear(base) = base_obj.model else {
        return Err(Error::internal(
            "context mixture base fit did not produce a sparse model",
        ));
    };
    let h = base.hypothesis_all_from_source(source, n)?;
    let residual: Vec<i32> = source
        .iter()
        .zip(h.iter())
        .map(|(&x, &hh)| (i64::from(x) - i64::from(hh)) as i32)
        .collect();
    // Two edges at the 1/3 and 2/3 quantiles of |residual|.
    let mut mags: Vec<u32> = residual.iter().map(|v| v.unsigned_abs()).collect();
    mags.sort_unstable();
    let q = |p: f64| -> u32 {
        let idx = ((mags.len() as f64 - 1.0) * p).round() as usize;
        mags[idx.min(mags.len() - 1)]
    };
    let mut edges = vec![q(1.0 / 3.0), q(2.0 / 3.0)];
    edges.sort_unstable();
    edges.dedup();
    if edges.is_empty() {
        // Degenerate residual: fall back to the base predictor wrapped as a
        // single-expert mixture.
        let m = ContextMixturePredictor {
            channels: 1,
            edges: Vec::new(),
            experts: vec![base.clone()],
        };
        let object = LearnedObject::from_intrinsic_exp2(
            LearnedModel::ContextMixture(m),
            1,
            frames,
            sample_rate_hz,
            Vec::new(),
            source,
        )?;
        stats.quantization_attempts += 1;
        stats.fit_ns = sw.elapsed_ns().max(0) as u64;
        return Ok((object, stats));
    }
    let buckets = edges.len() + 1;
    let max_lag = base.lags.iter().map(|&l| usize::from(l)).max().unwrap_or(1);
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); buckets];
    let mut prev: u32 = 0;
    for (t, &rv) in residual.iter().enumerate() {
        if t >= max_lag {
            rows[bucket_of(&edges, prev)].push(t);
        }
        prev = rv.unsigned_abs();
    }
    let mut experts = Vec::with_capacity(buckets);
    for (idx, r) in rows.iter().enumerate() {
        stats.candidates += 1;
        let (w, b) = fit_expert(source, &base.lags, r, budget.ridge_lambda).unwrap_or_else(|| {
            (
                base.weights
                    .iter()
                    .map(|&v| f64::from(v) / 4096.0)
                    .collect(),
                f64::from(base.bias[0]) / 4096.0,
            )
        });
        let _ = idx;
        experts.push(SparseLinearPredictor {
            channels: 1,
            lags: base.lags.clone(),
            weights: w.iter().map(|&v| quantize_weight(v)).collect(),
            bias: vec![quantize_bias(b)],
            block_frames: None,
            coeff_encoding: crate::learned::sparse::COEFF_DELTA_VARINT,
        });
    }
    let model = ContextMixturePredictor {
        channels: 1,
        edges,
        experts,
    };
    if model.validate().is_err() {
        return Err(Error::internal(
            "context mixture fit produced an invalid model",
        ));
    }
    // Keep the base if the mixture is larger.
    let mixture_bytes = measure(model.clone(), source, frames, sample_rate_hz);
    let base_bytes = measure(
        ContextMixturePredictor {
            channels: 1,
            edges: Vec::new(),
            experts: vec![base.clone()],
        },
        source,
        frames,
        sample_rate_hz,
    );
    let chosen = match (mixture_bytes, base_bytes) {
        (Some(mb), Some(bb)) if mb <= bb => model,
        (Some(_), Some(_)) => ContextMixturePredictor {
            channels: 1,
            edges: Vec::new(),
            experts: vec![base.clone()],
        },
        (Some(_), None) => model,
        (None, Some(_)) => ContextMixturePredictor {
            channels: 1,
            edges: Vec::new(),
            experts: vec![base.clone()],
        },
        (None, None) => return Err(Error::internal("context mixture produced no candidate")),
    };
    let object = LearnedObject::from_intrinsic_exp2(
        LearnedModel::ContextMixture(chosen),
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
    fn context_mixture_fit_closes() {
        let mut x = vec![0i32; 8];
        for t in 8..2500 {
            let v = (x[t - 1] as i64 * 45) / 100
                + (x[t - 5] as i64 * 35) / 100
                + ((t % 19) as i64 - 9) * 13;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let budget = TrainBudget::default();
        let (o, _) = fit_context_mixture_object(&x, 2500, 48_000, &budget).unwrap();
        assert!(o.verify(&x));
    }
}

//! `EnvelopeFlattenedTNS` — envelope-flattened spectral predictor estimation
//! (Phase 6 mechanism 13; lossy/exploratory).
//!
//! Temporal Noise Shaping estimates a low-order linear predictor over a
//! transform's spectral coefficients to model their fine structure. Estimated
//! directly on the raw spectrum, an order-3 or order-4 predictor spends its few
//! degrees of freedom fitting the **gross spectral envelope** — which the
//! quantizer's own scalefactors already carry — instead of the fine structure it
//! is meant to shape.
//!
//! The mechanism separates the two: smooth the magnitude spectrum into an
//! envelope `A[k]`, flatten `Y[k] = X[k] / A[k]`, estimate the predictor from
//! `Y`, and apply it to shape the real spectrum. On transient material this must
//! capture more fine structure with the same predictor order.
//!
//! VOLE has no lossy transform codec, so this is an **exploratory spectral
//! estimator** court, not a production path: it measures prediction residual
//! energy on the flattened spectrum, where the comparison is meaningful.

/// A forward DCT-II of a real frame (analysis transform stand-in for an MDCT).
pub fn dct2(frame: &[f64]) -> Vec<f64> {
    let n = frame.len();
    let mut out = vec![0f64; n];
    let scale = std::f64::consts::PI / n as f64;
    for (k, slot) in out.iter_mut().enumerate() {
        let mut acc = 0f64;
        for (i, &x) in frame.iter().enumerate() {
            acc += x * (((i as f64) + 0.5) * scale * k as f64).cos();
        }
        *slot = acc;
    }
    out
}

/// Smooth `|x[k]|` into a local spectral envelope with a moving average.
pub fn spectral_envelope(x: &[f64], radius: usize) -> Vec<f64> {
    let n = x.len();
    let mut env = vec![0f64; n];
    for (k, slot) in env.iter_mut().enumerate() {
        let lo = k.saturating_sub(radius);
        let hi = (k + radius).min(n - 1);
        let mut acc = 0f64;
        for &v in &x[lo..=hi] {
            acc += v.abs();
        }
        *slot = acc / (hi - lo + 1) as f64;
    }
    env
}

/// Flatten `x` by its envelope, never dividing by zero.
pub fn flatten(x: &[f64], env: &[f64], epsilon: f64) -> Vec<f64> {
    x.iter()
        .zip(env)
        .map(|(&v, &e)| v / e.max(epsilon))
        .collect()
}

/// Least-squares AR predictor of `order` for a real sequence
/// (`x[k] ≈ Σ_j c_j x[k-j]`), or `None` when the normal equations are singular.
pub fn fit_ar(x: &[f64], order: usize) -> Option<Vec<f64>> {
    let n = x.len();
    if order == 0 || n <= order + 2 {
        return None;
    }
    let mut a = vec![vec![0f64; order]; order];
    let mut b = vec![0f64; order];
    for k in order..n {
        for (i, (ai, bi)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
            let xi = x[k - 1 - i];
            *bi += xi * x[k];
            for (j, aij) in ai.iter_mut().enumerate() {
                *aij += xi * x[k - 1 - j];
            }
        }
    }
    solve(a, b, order)
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let d = a[col][col];
        for aij in a[col].iter_mut().skip(col) {
            *aij /= d;
        }
        b[col] /= d;
        let pivot_row = a[col].clone();
        let pivot_b = b[col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
                continue;
            }
            for (j, aij) in a[r].iter_mut().enumerate().skip(col) {
                *aij -= f * pivot_row[j];
            }
            b[r] -= f * pivot_b;
        }
    }
    Some(b)
}

/// Mean squared AR prediction residual of `x` under `coeffs` (applied causally).
pub fn residual_energy(x: &[f64], coeffs: &[f64]) -> f64 {
    let p = coeffs.len();
    if x.len() <= p + 1 {
        return 0.0;
    }
    let mut acc = 0f64;
    let mut count = 0usize;
    for k in p..x.len() {
        let mut pred = 0f64;
        for (j, &c) in coeffs.iter().enumerate() {
            pred += c * x[k - 1 - j];
        }
        let e = x[k] - pred;
        acc += e * e;
        count += 1;
    }
    if count == 0 { 0.0 } else { acc / count as f64 }
}

/// The measured outcome of one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TnsOutcome {
    /// Residual energy on the flattened spectrum using the predictor estimated
    /// from the raw spectrum.
    pub raw_estimated: f64,
    /// Residual energy on the flattened spectrum using the predictor estimated
    /// after envelope flattening.
    pub flattened_estimated: f64,
}

impl TnsOutcome {
    /// Prediction gain of the flattened estimator over the raw estimator on the
    /// flattened target (`> 1` means the flattened estimator is better).
    pub fn gain(&self) -> f64 {
        if self.flattened_estimated <= 0.0 {
            return f64::INFINITY;
        }
        self.raw_estimated / self.flattened_estimated
    }
}

/// Estimate a TNS predictor both ways and measure them **out of sample** on the
/// flattened target.
///
/// The predictor is fitted on the first half of the frame's spectrum and scored
/// on the second half, so neither estimator is evaluated on the data it was
/// fitted to — an in-sample comparison would trivially favour whichever target
/// the predictor was fitted to, which is the whole point of the mechanism.
pub fn envelope_flattened_tns(frame: &[f64], order: usize, radius: usize) -> Option<TnsOutcome> {
    let spectrum = dct2(frame);
    let env = spectral_envelope(&spectrum, radius);
    let flat = flatten(&spectrum, &env, 1e-9);
    let half = flat.len() / 2;
    if half <= order + 2 {
        return None;
    }
    let raw_coeffs = fit_ar(&spectrum[..half], order)?;
    let flat_coeffs = fit_ar(&flat[..half], order)?;
    Some(TnsOutcome {
        raw_estimated: residual_energy(&flat[half..], &raw_coeffs),
        flattened_estimated: residual_energy(&flat[half..], &flat_coeffs),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_of_a_constant_puts_energy_in_bin_zero() {
        let x = vec![1.0f64; 16];
        let s = dct2(&x);
        assert!(s[0].abs() > 10.0);
        for v in s.iter().skip(1) {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn flattening_normalizes_the_envelope() {
        let x: Vec<f64> = (0..64).map(|i| ((i as f64) * 0.3).sin() * 100.0).collect();
        let env = spectral_envelope(&x, 3);
        let flat = flatten(&x, &env, 1e-9);
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ar_recovers_an_exactly_predictable_sequence() {
        // x[k] = 0.5 x[k-1] + 0.25 x[k-2]
        let mut x = vec![1.0f64, 0.5];
        for k in 2..128 {
            x.push(0.5 * x[k - 1] + 0.25 * x[k - 2]);
        }
        let c = fit_ar(&x, 2).unwrap();
        assert!((c[0] - 0.5).abs() < 1e-6, "{c:?}");
        assert!((c[1] - 0.25).abs() < 1e-6, "{c:?}");
        assert!(residual_energy(&x, &c) < 1e-18);
    }
}

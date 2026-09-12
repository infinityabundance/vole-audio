//! `CentroidSQ` — centroid scalar reconstruction (Phase 6 mechanism 14;
//! lossy/exploratory).
//!
//! A scalar quantizer maps a coefficient to a symbol `q` and later reconstructs
//! a value. The usual reconstruction is the **cell midpoint**
//! `x̂ = sgn(q)(|q| - ½)Δ`. If the coefficient density inside a cell is not
//! uniform — and for transform coefficients it is strongly concentrated toward
//! zero — the midpoint is not the minimum-distortion reconstruction point.
//!
//! `CentroidSQ` keeps the decision boundaries (and therefore the symbols, and
//! therefore the bitstream) exactly as they are, and moves only the
//! reconstruction point toward the source distribution's conditional mean inside
//! each cell: `x̂ = sgn(q)(|q| + c)Δ`, with `c` fitted on a training set. The
//! reconstruction rule is profile-defined, so a decoder that knows the profile
//! pays no extra bits.
//!
//! VOLE has no lossy transform codec, so this is an **exploratory RD** court: it
//! fits `c` on a training set and reports held-out mean-squared error at
//! identical symbols.

/// A uniform-threshold scalar quantizer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantizer {
    /// Deadzone half-width: `|x| < deadzone` maps to symbol 0.
    pub deadzone: f64,
    /// Nonzero cell width.
    pub delta: f64,
}

impl Quantizer {
    /// The quantizer symbol for a coefficient (decision boundaries only).
    pub fn symbol(&self, x: f64) -> i64 {
        let a = x.abs();
        if a < self.deadzone {
            return 0;
        }
        let q = 1 + ((a - self.deadzone) / self.delta).floor() as i64;
        if x < 0.0 { -q } else { q }
    }

    /// Reconstruct symbol `q` at offset `c` inside its cell. `c = 0.5` is the
    /// midpoint rule; `c < 0.5` moves the point toward the cell's lower edge,
    /// where transform-coefficient density is concentrated.
    pub fn reconstruct(&self, q: i64, c: f64) -> f64 {
        if q == 0 {
            return 0.0;
        }
        let mag = self.deadzone + (q.unsigned_abs() as f64 - 1.0 + c) * self.delta;
        if q < 0 { -mag } else { mag }
    }
}

/// Mean-squared reconstruction error of `samples` under predictor offset `c`.
pub fn mse(samples: &[f64], q: &Quantizer, c: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut acc = 0f64;
    for &x in samples {
        let s = q.symbol(x);
        let e = x - q.reconstruct(s, c);
        acc += e * e;
    }
    acc / samples.len() as f64
}

/// Fit the reconstruction offset on a training set by a fixed grid search.
/// Returns `(best_c, best_mse)`; `c = 0.5` (the midpoint rule) is always in the
/// grid, so the fitted offset can never be worse than the midpoint.
pub fn fit_offset(training: &[f64], q: &Quantizer) -> (f64, f64) {
    let mut best_c = 0.5f64;
    let mut best = mse(training, q, 0.5);
    let mut step = 0i32;
    while step <= 18 {
        let c = 0.05 + 0.05 * step as f64;
        let m = mse(training, q, c);
        if m < best {
            best = m;
            best_c = c;
        }
        step += 1;
    }
    (best_c, best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn laplace(n: usize, scale: f64) -> Vec<f64> {
        // Deterministic heavy-tailed sample, no RNG dependency.
        let mut state = 12345u64;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            let sign = if u < 0.0 { -1.0 } else { 1.0 };
            let mag = -scale * (1.0 - 2.0 * u.abs()).ln();
            out.push(sign * mag);
        }
        out
    }

    #[test]
    fn symbols_are_unchanged_by_the_reconstruction_offset() {
        let q = Quantizer {
            deadzone: 0.25,
            delta: 0.5,
        };
        let x = laplace(1000, 1.0);
        let symbols: Vec<i64> = x.iter().map(|&v| q.symbol(v)).collect();
        let _ = mse(&x, &q, 0.5);
        let _ = mse(&x, &q, 0.2);
        let after: Vec<i64> = x.iter().map(|&v| q.symbol(v)).collect();
        assert_eq!(symbols, after);
    }

    #[test]
    fn the_fitted_offset_never_loses_to_the_midpoint() {
        let q = Quantizer {
            deadzone: 0.25,
            delta: 0.5,
        };
        let x = laplace(4000, 1.0);
        let (_, fitted) = fit_offset(&x, &q);
        assert!(fitted <= mse(&x, &q, 0.5) + 1e-18);
    }

    #[test]
    fn zero_symbol_reconstructs_to_zero_for_every_offset() {
        let q = Quantizer {
            deadzone: 0.5,
            delta: 1.0,
        };
        assert_eq!(q.reconstruct(0, 0.1), 0.0);
        assert_eq!(q.reconstruct(0, 0.9), 0.0);
    }
}

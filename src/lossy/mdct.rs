//! Critically-sampled MDCT with a sine window and 50% overlap.
//!
//! `Mdct::new(n)` builds a transform with `n` output coefficients from a
//! `2n`-sample windowed input; successive frames advance by `n` samples and the
//! analysis and synthesis windows satisfy the Princen–Bradley TDAC condition, so
//! perfect reconstruction holds for unmodified coefficients.
//!
//! The matrix is precomputed and the transform is a fixed deterministic matrix
//! multiply. That is `O(n²)` per frame, which is the right trade at the frame
//! sizes this codec uses (`n ≤ 512`): it keeps the transform bit-exactly
//! reproducible across hosts, and the alternative fast algorithms would add a
//! second implementation to prove correct.

/// A precomputed MDCT analysis/synthesis pair.
#[derive(Debug, Clone)]
pub struct Mdct {
    n: usize,
    window: Vec<f64>,
    /// `cos[k * 2n + m]`.
    cos: Vec<f64>,
}

impl Mdct {
    /// Build the transform for `n` coefficients (`2n` window samples).
    pub fn new(n: usize) -> Mdct {
        let two_n = 2 * n;
        let mut window = vec![0.0f64; two_n];
        for (m, w) in window.iter_mut().enumerate() {
            *w = (std::f64::consts::PI / two_n as f64 * (m as f64 + 0.5)).sin();
        }
        let mut cos = vec![0.0f64; n * two_n];
        for k in 0..n {
            for m in 0..two_n {
                cos[k * two_n + m] = (std::f64::consts::PI / n as f64
                    * (m as f64 + 0.5 + n as f64 / 2.0)
                    * (k as f64 + 0.5))
                    .cos();
            }
        }
        Mdct { n, window, cos }
    }

    /// Number of coefficients (half the window length).
    pub fn n(&self) -> usize {
        self.n
    }

    /// Analysis: `n` coefficients from a `2n`-sample frame.
    pub fn forward(&self, frame: &[f64]) -> Vec<f64> {
        debug_assert_eq!(frame.len(), 2 * self.n);
        let n = self.n;
        let two_n = 2 * n;
        let mut out = vec![0.0f64; n];
        for (k, slot) in out.iter_mut().enumerate() {
            let row = &self.cos[k * two_n..k * two_n + two_n];
            let mut acc = 0.0f64;
            for m in 0..two_n {
                acc += self.window[m] * frame[m] * row[m];
            }
            *slot = acc;
        }
        out
    }

    /// Synthesis into a fresh buffer, **without** the window and without
    /// overlap-add. `inverse_add` applies the window and accumulates; this is
    /// the unwindowed synthesis used to reason about distortion.
    pub fn synthesis(&self, coeffs: &[f64]) -> Vec<f64> {
        let two_n = 2 * self.n;
        let scale = 2.0 / self.n as f64;
        let mut out = vec![0.0f64; two_n];
        for (m, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, &c) in coeffs.iter().enumerate() {
                acc += c * self.cos[k * two_n + m];
            }
            *slot = scale * acc;
        }
        out
    }

    /// The analysis window (`2n` samples).
    pub fn window(&self) -> &[f64] {
        &self.window
    }

    /// Synthesis with overlap-add into `out` starting at `offset`.
    pub fn inverse_add(&self, coeffs: &[f64], out: &mut [f64], offset: usize) {
        let n = self.n;
        let two_n = 2 * n;
        let scale = 2.0 / n as f64;
        for m in 0..two_n {
            let mut acc = 0.0f64;
            for (k, &c) in coeffs.iter().enumerate() {
                acc += c * self.cos[k * two_n + m];
            }
            out[offset + m] += self.window[m] * scale * acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mdct_is_perfect_reconstruction() {
        let n = 64;
        let mdct = Mdct::new(n);
        let len = 1000usize;
        let x: Vec<f64> = (0..len).map(|i| (i as f64 * 0.31).sin() * 1000.0).collect();
        let hop = n;
        let frames = (len + hop).div_ceil(hop);
        let mut out = vec![0.0f64; frames * hop + hop];
        for f in 0..frames {
            let start = f * hop - hop;
            let mut frame = vec![0.0f64; 2 * n];
            for (m, slot) in frame.iter_mut().enumerate() {
                let g = start as i64 + m as i64;
                if g >= 0 && (g as usize) < len {
                    *slot = x[g as usize];
                }
            }
            let c = mdct.forward(&frame);
            mdct.inverse_add(&c, &mut out, f * hop);
        }
        let mut worst = 0.0f64;
        for i in 0..len {
            worst = worst.max((out[i + hop] - x[i]).abs());
        }
        assert!(worst < 1e-6, "MDCT reconstruction error {worst}");
    }
}

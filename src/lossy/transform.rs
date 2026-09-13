//! MDCT transform engine: psychoacoustic band allocation, dead-zone scalar
//! quantization, and VOLE-native entropy coding of the quantized coefficients.
//!
//! ```text
//! frame ──► MDCT ──► masking thresholds ──► per-band steps
//!                                              │
//!                              dead-zone scalar quantizer
//!                                              │
//!                        delta-coded band shape + entropy-coded symbols
//! ```
//!
//! The exponent stream is split into a slowly-varying **shape** (the band step
//! profile in `log2` units) and a single **gain** (the rate scalar). The shape
//! is predicted from the previous frame's shape both temporally and spectrally,
//! so only its change is transmitted; the gain is delta-coded separately. This
//! keeps the model description cheap even when the rate control moves the gain
//! sharply between frames.
//!
//! ## Distortion units
//!
//! [`TransformCandidate::distortion`] is the **time-domain** squared
//! reconstruction error the frame contributes after overlap-add, so it is
//! directly comparable with the predictive engine's error inside the
//! rate–distortion search.
//!
//! The forward transform is `c[k] = Σ_m w[m] x[m] cos[m,k]` and the synthesis is
//! `y[m] = (2/N) Σ_k c[k] cos[m,k] = w[m] x[m]`; the decimated overlap-add then
//! accumulates `w[m] y[m]`. A coefficient error `e[k]` therefore contributes a
//! time-domain error `w[m]·(2/N)·Σ_k e[k] cos[m,k]`, whose expected energy over
//! the frame is `(2/N)·Σ_k e[k]²` because `Σ_m w²[m] ≈ N` and
//! `Σ_k cos²[m,k] = N/2`. That is the constant the code applies, and the
//! `distortion_matches_time_domain_error` test checks it against a true
//! overlap-add reconstruction.

use crate::learned::residual_codec2::{ResidualCodecV2, decode_encoding_v3, encode_best_subset};
use crate::lossy::mdct::Mdct;
use crate::lossy::predict::{MAX_SYMBOL, quantizer};
use crate::lossy::psy::{Bands, Masking};

/// Time-domain error energy per unit of coefficient-domain error energy.
pub const fn distortion_scale(n: usize) -> f64 {
    2.0 / n as f64
}

/// Largest gain (in `log2` step units) that still leaves every band's own peak
/// coefficient representable. `exps[b] = shape[b] + gain` must stay below the
/// band peak's exponent, or the band's dominant coefficient falls inside the
/// dead zone and the band is quantised to nothing.
///
/// This is the **analysis-only** statement of the constraint. Applying it as a
/// single global bound is too coarse to ship — one quiet band then pins the
/// gain for the whole channel — which is why the codec still carries a single
/// global gain and this bound is recorded rather than enforced. A correct fix
/// needs per-band rate allocation, which is the first item of the 7B remainder.
pub fn gain_cap(step0: &[f64], coeffs: &[f64], bands: &Bands) -> i32 {
    let bc = bands.band_count;
    let mut peak = vec![0.0f64; bc];
    for (k, &x) in coeffs.iter().enumerate() {
        let b = bands.of_bin[k] as usize;
        peak[b] = peak[b].max(x.abs());
    }
    let mut cap = i32::MAX;
    for b in 0..bc {
        if peak[b] <= 0.0 {
            continue;
        }
        let shape = step0[b].log2().round().clamp(-50.0, 40.0) as i32;
        let allowed = peak[b].log2().floor() as i32 - 1 - shape;
        cap = cap.min(allowed);
    }
    cap
}

/// One evaluated transform frame.
#[derive(Debug, Clone)]
pub struct TransformCandidate {
    /// Complete frame record bytes.
    pub record: Vec<u8>,
    /// Absorbed band shape (`log2` step profile) after this frame.
    pub shape: Vec<i32>,
    /// Absorbed global gain in `log2` step units.
    pub gain: i32,
    /// Time-domain squared reconstruction error (see module docs).
    pub distortion: f64,
}

/// Per-channel transform state.
#[derive(Debug, Clone)]
pub struct TransformEngine {
    bands: Bands,
    mdct: Mdct,
    shape_prev: Vec<i32>,
    gain_prev: i32,
}

impl TransformEngine {
    /// Build an engine for a band layout and transform.
    pub fn new(bands: Bands, mdct: Mdct) -> TransformEngine {
        let bc = bands.band_count;
        TransformEngine {
            bands,
            mdct,
            shape_prev: vec![0i32; bc],
            gain_prev: 0,
        }
    }

    /// The band layout.
    pub fn bands(&self) -> &Bands {
        &self.bands
    }

    /// The transform.
    pub fn mdct(&self) -> &Mdct {
        &self.mdct
    }

    /// Masking-derived reference step per band: the step that would place
    /// quantization noise exactly at the band's allowed power.
    pub fn step0(&self, coeffs: &[f64]) -> Vec<f64> {
        let masking = Masking::analyse(coeffs, &self.bands);
        let bc = self.bands.band_count;
        let mut step0 = vec![0.0f64; bc];
        for (b, slot) in step0.iter_mut().enumerate() {
            let n_b = f64::from(self.bands.bins_per_band[b].max(1));
            *slot = (12.0 * masking.threshold_power[b] / n_b).sqrt().max(1e-6);
        }
        step0
    }

    /// Evaluate one frame at a candidate global gain (in `log2` step units).
    pub fn evaluate(&self, coeffs: &[f64], step0: &[f64], gain: i32) -> TransformCandidate {
        self.evaluate_with(coeffs, step0, gain, &ResidualCodecV2::ALL_V3)
    }

    /// Evaluate with an explicit entropy-codec subset. The encoder searches with
    /// a cheap subset and re-encodes the chosen gain with the full family, so
    /// the subset only has to order the candidates.
    pub fn evaluate_with(
        &self,
        coeffs: &[f64],
        step0: &[f64],
        gain: i32,
        codecs: &[ResidualCodecV2],
    ) -> TransformCandidate {
        let bc = self.bands.band_count;
        let mut shape = vec![0i32; bc];
        for (b, slot) in shape.iter_mut().enumerate() {
            *slot = step0[b].log2().round().clamp(-50.0, 40.0) as i32;
        }
        let mut exps = shape.clone();
        for slot in exps.iter_mut() {
            *slot = (*slot + gain).clamp(-40, 70);
        }
        // 2-D prediction: temporal (previous frame's shape) followed by spectral
        // (previous band's residual). Both terms are decoder-visible.
        let mut deltas: Vec<i32> = Vec::with_capacity(bc + 1);
        deltas.push(gain - self.gain_prev);
        let mut prev_resid = 0i32;
        for (b, &s) in shape.iter().enumerate() {
            let resid = s - self.shape_prev[b];
            deltas.push(resid - prev_resid);
            prev_resid = resid;
        }

        let mut q = quantizer(1.0);
        let mut flat = Vec::with_capacity(coeffs.len());
        let mut distortion = 0.0f64;
        for (k, &x) in coeffs.iter().enumerate() {
            let b = self.bands.of_bin[k] as usize;
            q.delta = 2f64.powi(exps[b]);
            q.deadzone = 0.5 * q.delta;
            // Clamp *before* measuring: the transmitted symbol is the clamped
            // one, so the reported distortion must describe that symbol or the
            // rate–distortion search is comparing two different quantisers.
            let sym = q.symbol(x).clamp(-MAX_SYMBOL, MAX_SYMBOL);
            let e = x - q.reconstruct(sym, 0.33);
            distortion += e * e;
            flat.push(sym as i32);
        }
        distortion *= distortion_scale(coeffs.len());

        let enc_exp = encode_best_subset(&deltas, codecs);
        let enc_coeff = encode_best_subset(&flat, codecs);
        let mut record = Vec::with_capacity(enc_exp.bytes.len() + enc_coeff.bytes.len() + 8);
        record.extend_from_slice(&(enc_exp.bytes.len() as u16).to_le_bytes());
        record.extend_from_slice(&enc_exp.bytes);
        record.extend_from_slice(&(enc_coeff.bytes.len() as u32).to_le_bytes());
        record.extend_from_slice(&enc_coeff.bytes);
        TransformCandidate {
            record,
            shape,
            gain,
            distortion,
        }
    }

    /// Absorb a committed frame into the cross-frame prediction state.
    pub fn commit(&mut self, c: &TransformCandidate) {
        self.shape_prev.clone_from(&c.shape);
        self.gain_prev = c.gain;
    }

    /// Decode one frame record, returning the reconstructed coefficient frame.
    /// `shape_prev`/`gain_prev` are the decoder's own previous values.
    pub fn decode(
        bands: &Bands,
        n: usize,
        record: &[u8],
        shape_prev: &[i32],
        gain_prev: i32,
    ) -> crate::error::Result<(Vec<f64>, Vec<i32>, i32)> {
        let bc = bands.band_count;
        let exp_len = u16::from_le_bytes(
            record
                .get(0..2)
                .ok_or_else(|| crate::error::Error::malformed("lossy frame is truncated"))?
                .try_into()
                .unwrap(),
        ) as usize;
        let exp = record.get(2..2 + exp_len).ok_or_else(|| {
            crate::error::Error::malformed("lossy frame exponent stream truncated")
        })?;
        let rest = &record[2 + exp_len..];
        let coeff_len = u32::from_le_bytes(
            rest.get(0..4)
                .ok_or_else(|| crate::error::Error::malformed("lossy frame is truncated"))?
                .try_into()
                .unwrap(),
        ) as usize;
        let coeff = rest.get(4..4 + coeff_len).ok_or_else(|| {
            crate::error::Error::malformed("lossy frame coefficient stream truncated")
        })?;

        let deltas = decode_encoding_v3(exp, bc + 1)?.decode(bc + 1)?;
        let gain = gain_prev + deltas[0];
        let mut shape = vec![0i32; bc];
        let mut prev_resid = 0i32;
        for (b, slot) in shape.iter_mut().enumerate() {
            prev_resid += deltas[b + 1];
            *slot = shape_prev[b] + prev_resid;
        }
        let decoded = decode_encoding_v3(coeff, n)?.decode(n)?;
        let mut coeffs = vec![0.0f64; n];
        for (k, slot) in coeffs.iter_mut().enumerate() {
            let b = bands.of_bin[k] as usize;
            let exp = (shape[b] + gain).clamp(-40, 70);
            *slot = quantizer(2f64.powi(exp)).reconstruct(i64::from(decoded[k]), 0.33);
        }
        Ok((coeffs, shape, gain))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reported distortion must equal the true overlap-added time-domain
    /// error energy of the frame, or the rate–distortion search compares two
    /// engines in different units.
    #[test]
    fn distortion_matches_time_domain_error() {
        let n = 64usize;
        let mdct = Mdct::new(n);
        let bands = Bands::new(16_000, n, 1000.0);
        let engine = TransformEngine::new(bands.clone(), mdct.clone());
        // A deterministic pseudo-random coefficient frame.
        let mut state = 0x2468u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let x: Vec<f64> = (0..2 * n).map(|_| rnd() * 1000.0).collect();
        let coeffs = mdct.forward(&x);
        let step0 = engine.step0(&coeffs);
        let cand = engine.evaluate(&coeffs, &step0, 0);

        // Rebuild the quantized coefficients the same way the decoder does.
        let shape: Vec<i32> = step0
            .iter()
            .map(|s| s.log2().round().clamp(-50.0, 40.0) as i32)
            .collect();
        let mut q = quantizer(1.0);
        let mut quant = vec![0.0f64; n];
        for (k, slot) in quant.iter_mut().enumerate() {
            let b = bands.of_bin[k] as usize;
            let exp = (shape[b] + cand.gain).clamp(-40, 70);
            q.delta = 2f64.powi(exp);
            q.deadzone = 0.5 * q.delta;
            let sym = q.symbol(coeffs[k]);
            *slot = q.reconstruct(sym, 0.33);
        }
        // True time-domain contribution: the window times the unwindowed
        // synthesis of the coefficient error.
        let e: Vec<f64> = coeffs
            .iter()
            .zip(quant.iter())
            .map(|(a, b)| a - b)
            .collect();
        let synth = mdct.synthesis(&e);
        let mut err = 0.0f64;
        for (m, &s) in synth.iter().enumerate() {
            let v = mdct.window()[m] * s;
            err += v * v;
        }
        let ratio = cand.distortion / err;
        assert!(
            (ratio - 1.0).abs() < 0.25,
            "distortion {} vs time-domain {} (ratio {ratio})",
            cand.distortion,
            err
        );
    }
}

//! Linear-prediction engine with transmitted per-frame parameters:
//! short-term LPC (quantized reflection coefficients) plus a long-term pitch
//! predictor.
//!
//! ```text
//! analysis:   autocorrelation(frame) → Levinson–Durbin → a[1..P], k[1..P]
//!             open-loop LPC residual → pitch search → lag, g_p
//!             k quantized uniformly in asin(k)  (stability preserving)
//!
//! synthesis:  pred[n] = Σ a_j·x̂h[n-j] + g_p·e[n-lag]
//!             r [n]   = x[n] - pred[n]
//!             r̂ [n]   = Q_Δ(r[n])                  (dead-zone scalar)
//!             x̂h[n]   = pred[n] + r̂[n] ;  e[n] = r̂[n]
//! ```
//!
//! Both stages the decoder needs are in the bitstream as *parameters* and
//! everything else is decoder-derived, so the loop has no hidden state. The
//! synthesis filter is guaranteed stable because every quantized reflection
//! coefficient stays strictly inside `(-1, 1)`.
//!
//! The short-term predictor removes the formant envelope; the long-term
//! predictor removes the periodic excitation. Together they are what lets a
//! predictive codec beat a plain transform on voiced speech at low rates.

use crate::learned::centroid_sq::Quantizer;

/// `Δ = 2^(gain/2)`, so one gain unit is half a bit of step size.
pub fn step_of(gain: i32) -> f64 {
    2f64.powf(f64::from(gain) * 0.5)
}

/// Inverse of [`step_of`], rounded to the nearest gain unit.
pub fn gain_of(step: f64) -> i32 {
    (2.0 * (step.max(1e-6)).log2()).round() as i32
}

/// Largest residual symbol magnitude the entropy coders will ever see. A
/// runaway DPCM frame can otherwise emit an astronomically large symbol that
/// makes the range coders pathological while carrying no real information. The
/// clamp is applied at the encoder and the transmitted symbol is already
/// clamped, so the decoder reconstructs the identical value.
pub const MAX_SYMBOL: i64 = 1 << 20;

/// The dead-zone quantizer used for prediction residuals.
pub fn quantizer(step: f64) -> Quantizer {
    Quantizer {
        deadzone: 0.5 * step,
        delta: step,
    }
}

/// Centroid reconstruction offset for residual cells.
pub const RECONSTRUCTION_OFFSET: f64 = 0.33;

/// Levels per quantized reflection coefficient (5 bits).
pub const REFLECTION_LEVELS: i32 = 32;
/// Levels for the quantized long-term gain (5 bits).
pub const LTPG_LEVELS: i32 = 32;
/// Lower edge of the long-term gain range.
pub const LTPG_MIN: f64 = -1.0;
/// Upper edge of the long-term gain range.
pub const LTPG_MAX: f64 = 1.5;

/// Analysis result for one frame.
#[derive(Debug, Clone)]
pub struct FrameParams {
    /// Analysed reflection coefficients.
    pub k: Vec<f64>,
    /// Pitch lag in samples (0 when unvoiced).
    pub lag: i32,
    /// Long-term gain.
    pub ltpg: f64,
    /// Open-loop residual RMS after both predictors (rate estimate only).
    pub residual_rms: f64,
}

/// Quantize a reflection coefficient in the arcsine domain, which keeps the
/// dequantized value strictly inside `(-1, 1)` and spends resolution near the
/// unit circle where the filter is most sensitive.
pub fn quantize_reflection(k: f64) -> i32 {
    let t = (k.clamp(-0.999_9, 0.999_9).asin() / (std::f64::consts::FRAC_PI_2) + 1.0) * 0.5;
    (t * f64::from(REFLECTION_LEVELS - 1))
        .round()
        .clamp(0.0, f64::from(REFLECTION_LEVELS - 1)) as i32
}

/// Inverse of [`quantize_reflection`]. The angle is held just inside `pi/2` so
/// the reconstructed coefficient is strictly less than one in magnitude, which
/// is what keeps the synthesis filter minimum phase.
pub fn dequantize_reflection(q: i32) -> f64 {
    let t = q as f64 / f64::from(REFLECTION_LEVELS - 1);
    ((t * 2.0 - 1.0) * (std::f64::consts::FRAC_PI_2 - 1.0e-4)).sin()
}

/// Quantize the long-term gain.
pub fn quantize_ltpg(g: f64) -> i32 {
    let t = (g.clamp(LTPG_MIN, LTPG_MAX) - LTPG_MIN) / (LTPG_MAX - LTPG_MIN);
    (t * f64::from(LTPG_LEVELS - 1)).round() as i32
}

/// Inverse of [`quantize_ltpg`].
pub fn dequantize_ltpg(q: i32) -> f64 {
    LTPG_MIN + (LTPG_MAX - LTPG_MIN) * q as f64 / f64::from(LTPG_LEVELS - 1)
}

/// Autocorrelation LPC analysis. Returns the direct-form prediction weights
/// (`x̂ = Σ a_j x[n-j]`) and the reflection coefficients.
pub fn lpc_reflection(x: &[f64], order: usize, sample_rate_hz: u32) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n <= order || order == 0 {
        return (vec![0.0; order], vec![0.0; order]);
    }
    // Triangular (Bartlett) lag window reduces the edge bias of a short frame.
    let mut r = vec![0.0f64; order + 1];
    for (lag, slot) in r.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for i in lag..n {
            acc += x[i] * x[i - lag];
        }
        *slot = acc / n as f64;
    }
    r[0] = r[0] * 1.0001 + 1e-9;
    // Gaussian lag window of ~60 Hz bandwidth: mild enough to preserve the
    // higher-order coefficients while still damping the frame-edge bias.
    for (lag, slot) in r.iter_mut().enumerate().skip(1) {
        let w = (-0.5
            * (2.0 * std::f64::consts::PI * 60.0 * lag as f64 / f64::from(sample_rate_hz)).powi(2))
        .exp();
        *slot *= w;
    }
    let mut a = vec![0.0f64; order + 1];
    let mut a_prev = vec![0.0f64; order + 1];
    let mut k = vec![0.0f64; order + 1];
    a[0] = 1.0;
    let mut err = r[0];
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc += a_prev[j] * r[i - j];
        }
        let ki = if err.abs() < 1e-12 { 0.0 } else { -acc / err };
        let ki = ki.clamp(-0.999_9, 0.999_9);
        k[i] = ki;
        a[0] = 1.0;
        a[i] = ki;
        for j in 1..i {
            a[j] = a_prev[j] + ki * a_prev[i - j];
        }
        err *= 1.0 - ki * ki;
        if err <= 1e-12 {
            for j in (i + 1)..=order {
                a[j] = 0.0;
                k[j] = 0.0;
            }
            break;
        }
        a_prev.copy_from_slice(&a);
    }
    let weights: Vec<f64> = (1..=order).map(|j| -a[j]).collect();
    (weights, k[1..=order].to_vec())
}

/// Convert reflection coefficients to direct-form prediction weights.
pub fn reflection_to_weights(k: &[f64]) -> Vec<f64> {
    let order = k.len();
    let mut a = vec![0.0f64; order + 1];
    let mut a_prev = vec![0.0f64; order + 1];
    a[0] = 1.0;
    for i in 1..=order {
        let ki = k[i - 1].clamp(-0.999_9, 0.999_9);
        a[0] = 1.0;
        a[i] = ki;
        for j in 1..i {
            a[j] = a_prev[j] + ki * a_prev[i - j];
        }
        a_prev.copy_from_slice(&a);
    }
    (1..=order).map(|j| -a[j]).collect()
}

/// Analyse one frame of `full[start..start+n]` with `full[..start]` available as
/// history. `min_lag`/`max_lag` bound the pitch search and `lag_hint` biases it
/// toward the previous frame's lag so the tracker follows a voice instead of
/// jumping between octave-related candidates.
#[allow(clippy::too_many_arguments)]
pub fn analyse_frame(
    full: &[f64],
    start: usize,
    n: usize,
    order: usize,
    min_lag: usize,
    max_lag: usize,
    lag_hint: i32,
    sample_rate_hz: u32,
) -> FrameParams {
    // Short-term analysis uses a 2×n span centred on the frame: a 20 ms frame
    // alone gives a noisy autocorrelation for order 16, and a noisy filter is
    // one whose quantized parameters change every frame.
    let a_lo = start.saturating_sub(n / 2);
    let a_hi = (start + n + n / 2).min(full.len());
    let (weights, k) = if a_hi > a_lo + order + 1 {
        lpc_reflection(&full[a_lo..a_hi], order, sample_rate_hz)
    } else {
        (vec![0.0; order], vec![0.0; order])
    };

    // Open-loop LPC residual over the frame plus one pitch period of history.
    let lo = start.saturating_sub(max_lag).max(a_lo);
    let mut resid = Vec::with_capacity(start + n - lo);
    for i in lo..(start + n).min(full.len()) {
        let mut pred = 0.0f64;
        for (j, &w) in weights.iter().enumerate() {
            let idx = i as i64 - 1 - j as i64;
            if idx >= lo as i64 {
                pred += w * full[idx as usize];
            }
        }
        resid.push(full[i] - pred);
    }
    let frame_off = start - lo;

    // Pitch search on the residual, biased toward the previous lag.
    let mut best_lag = 0i32;
    let mut best_gain = 0.0f64;
    let mut best_score = 0.0f64;
    let mut best_energy = 0.0f64;
    let upper = max_lag.min(resid.len().saturating_sub(1));
    for lag in min_lag..=upper {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in frame_off..resid.len() {
            if i < lag {
                continue;
            }
            num += resid[i] * resid[i - lag];
            den += resid[i - lag] * resid[i - lag];
        }
        if den <= 1e-12 {
            continue;
        }
        // Continuity bias: a small penalty per sample of lag deviation keeps the
        // tracker on one pitch track when several lags correlate comparably.
        let bias = if lag_hint > 0 {
            let d = (lag as f64 - f64::from(lag_hint)).abs();
            0.02 * d / f64::from(lag_hint).max(1.0) * num.abs() / den.max(1e-12).sqrt()
        } else {
            0.0
        };
        let score = num.abs() / den.sqrt() - bias;
        if score > best_score {
            best_score = score;
            best_lag = lag as i32;
            best_gain = (num / den).clamp(LTPG_MIN, LTPG_MAX);
            best_energy = den;
        }
    }
    if best_gain < 0.1 || best_energy <= 1e-9 {
        best_lag = 0;
        best_gain = 0.0;
    }

    // Residual RMS after both predictors (rate control only).
    let mut energy = 0.0f64;
    let mut count = 0usize;
    for i in frame_off..resid.len() {
        let mut v = resid[i];
        if best_lag > 0 && i >= best_lag as usize {
            v -= best_gain * resid[i - best_lag as usize];
        }
        energy += v * v;
        count += 1;
    }
    let residual_rms = if count > 0 {
        (energy / count as f64).sqrt()
    } else {
        0.0
    };
    FrameParams {
        k,
        lag: best_lag,
        ltpg: best_gain,
        residual_rms,
    }
}

/// Number of sub-blocks per frame over which the synthesis filter is
/// interpolated from the previous frame's coefficients to this frame's. LPC
/// parameters quantized once per frame and applied abruptly produce a filter
/// discontinuity every frame, which is audible as roughness even when the
/// residual error is small.
pub const LPC_SUBBLOCKS: usize = 4;

/// Linearly interpolate two reflection-coefficient vectors and convert to
/// direct-form prediction weights.
fn interpolated_weights(k_prev: &[i32], k_cur: &[i32], t: f64) -> Vec<f64> {
    let k: Vec<f64> = k_prev
        .iter()
        .zip(k_cur.iter())
        .map(|(&a, &b)| {
            let ka = dequantize_reflection(a);
            let kb = dequantize_reflection(b);
            ka + t * (kb - ka)
        })
        .collect();
    reflection_to_weights(&k)
}

/// A committed prediction frame.
#[derive(Debug, Clone)]
pub struct PredictCandidate {
    pub record: Vec<u8>,
    pub distortion: f64,
}

/// One channel of forward-LPC + pitch prediction.
#[derive(Debug, Clone)]
pub struct PredictEngine {
    order: usize,
    min_lag: usize,
    max_lag: usize,
    /// Reconstructed sample history, most recent first.
    hist: Vec<f64>,
    /// Reconstructed excitation (quantized residual), most recent first.
    excit: Vec<f64>,
    k_prev: Vec<i32>,
    lag_prev: i32,
    ltpg_prev: i32,
    gain_prev: i32,
}

impl PredictEngine {
    /// A fresh engine. `min_lag`/`max_lag` bound the pitch search in samples.
    pub fn new(order: usize, min_lag: usize, max_lag: usize) -> PredictEngine {
        PredictEngine {
            order,
            min_lag,
            max_lag,
            hist: vec![0.0f64; order],
            excit: vec![0.0f64; max_lag],
            k_prev: vec![0i32; order],
            lag_prev: 0,
            ltpg_prev: 0,
            gain_prev: 0,
        }
    }

    /// Pitch-search bounds.
    pub fn lag_bounds(&self) -> (usize, usize) {
        (self.min_lag, self.max_lag)
    }

    /// Residual gain code absorbed by the last frame.
    pub fn last_gain(&self) -> i32 {
        self.gain_prev
    }

    /// Quantized reflection coefficients absorbed by the last frame.
    pub fn k_prev(&self) -> &[i32] {
        &self.k_prev
    }

    /// Previous quantized pitch lag.
    pub fn lag_prev(&self) -> i32 {
        self.lag_prev
    }

    /// Previous quantized long-term gain.
    pub fn ltpg_prev(&self) -> i32 {
        self.ltpg_prev
    }

    /// Evaluate a frame at a candidate residual gain without mutating state.
    pub fn evaluate(
        &self,
        x: &[f64],
        params: &FrameParams,
        gain: i32,
        codecs: &[crate::learned::residual_codec2::ResidualCodecV2],
    ) -> PredictCandidate {
        let mut probe = self.clone();
        probe.run(x, params, gain, codecs).0
    }

    /// Commit a frame: run the loop for real and update the state. Returns the
    /// record and the reconstruction the decoder will reproduce, which the
    /// encoder uses to keep its own state honest.
    pub fn commit(
        &mut self,
        x: &[f64],
        params: &FrameParams,
        gain: i32,
    ) -> (PredictCandidate, Vec<f64>) {
        self.run(
            x,
            params,
            gain,
            &crate::learned::residual_codec2::ResidualCodecV2::ALL_V3,
        )
    }

    /// Commit with an explicit entropy-codec subset (used during the search, so
    /// the subset does not define the final size).
    pub fn commit_with(
        &mut self,
        x: &[f64],
        params: &FrameParams,
        gain: i32,
        codecs: &[crate::learned::residual_codec2::ResidualCodecV2],
    ) -> (PredictCandidate, Vec<f64>) {
        self.run(x, params, gain, codecs)
    }

    fn run(
        &mut self,
        x: &[f64],
        params: &FrameParams,
        gain: i32,
        codecs: &[crate::learned::residual_codec2::ResidualCodecV2],
    ) -> (PredictCandidate, Vec<f64>) {
        let q = quantizer(step_of(gain));
        let k_q: Vec<i32> = params.k.iter().map(|&v| quantize_reflection(v)).collect();
        let mut weights = interpolated_weights(&self.k_prev, &k_q, 1.0 / LPC_SUBBLOCKS as f64);
        // An unvoiced frame keeps the previous lag so the delta costs nothing;
        // the gain is ~0 so the stale reference contributes nothing.
        let lag_q = if params.lag <= 0 {
            self.lag_prev
        } else {
            params.lag.clamp(self.min_lag as i32, self.max_lag as i32)
        };
        let lag = if lag_q <= 0 {
            0
        } else {
            (lag_q as usize).clamp(self.min_lag, self.max_lag)
        };
        let ltpg_q = quantize_ltpg(params.ltpg);
        let ltpg = dequantize_ltpg(ltpg_q);
        let mut vector: Vec<i32> = Vec::with_capacity(self.order + 4 + x.len());
        for (i, &v) in k_q.iter().enumerate() {
            vector.push(v - self.k_prev[i]);
        }
        vector.push(lag_q - self.lag_prev);
        vector.push(ltpg_q - self.ltpg_prev);
        vector.push(gain - self.gain_prev);

        let mut distortion = 0.0f64;
        let mut recon = Vec::with_capacity(x.len());
        let sub = x.len().div_ceil(LPC_SUBBLOCKS).max(1);
        for (n, &v) in x.iter().enumerate() {
            if n % sub == 0 {
                let t = ((n / sub) + 1) as f64 / LPC_SUBBLOCKS as f64;
                weights = interpolated_weights(&self.k_prev, &k_q, t);
            }
            let mut pred = 0.0f64;
            for (j, &w) in weights.iter().enumerate() {
                pred += w * self.hist[j];
            }
            if lag > 0 && lag <= self.excit.len() {
                pred += ltpg * self.excit[lag - 1];
            }
            let r = v - pred;
            let sym = q.symbol(r).clamp(-MAX_SYMBOL, MAX_SYMBOL);
            let err = q.reconstruct(sym, RECONSTRUCTION_OFFSET);
            let xh = pred + err;
            let d = v - xh;
            distortion += d * d;
            recon.push(xh);
            vector.push(sym as i32);
            for j in (1..self.order).rev() {
                self.hist[j] = self.hist[j - 1];
            }
            self.hist[0] = xh;
            for j in (1..self.excit.len()).rev() {
                self.excit[j] = self.excit[j - 1];
            }
            self.excit[0] = err;
        }
        let enc = crate::learned::residual_codec2::encode_best_subset(&vector, codecs);
        let mut record = Vec::with_capacity(enc.bytes.len() + 2);
        record.extend_from_slice(&(enc.bytes.len() as u16).to_le_bytes());
        record.extend_from_slice(&enc.bytes);
        self.k_prev = k_q;
        self.lag_prev = lag_q;
        self.ltpg_prev = ltpg_q;
        self.gain_prev = gain;
        (PredictCandidate { record, distortion }, recon)
    }

    /// Decoder-side reconstruction from transmitted parameter deltas and symbols.
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct(
        &mut self,
        delta_k: &[i32],
        delta_lag: i32,
        delta_ltpg: i32,
        gain: i32,
        symbols: &[i32],
    ) -> Vec<f64> {
        let q = quantizer(step_of(gain));
        let k_q: Vec<i32> = (0..self.order)
            .map(|i| self.k_prev[i] + delta_k[i])
            .collect();
        let lag_q = self.lag_prev + delta_lag;
        let ltpg_q = self.ltpg_prev + delta_ltpg;
        let lag = if lag_q <= 0 {
            0
        } else {
            (lag_q as usize).clamp(self.min_lag, self.max_lag)
        };
        let ltpg = dequantize_ltpg(ltpg_q.clamp(0, LTPG_LEVELS - 1));
        let mut out = Vec::with_capacity(symbols.len());
        let sub = symbols.len().div_ceil(LPC_SUBBLOCKS).max(1);
        let mut weights = interpolated_weights(&self.k_prev, &k_q, 1.0 / LPC_SUBBLOCKS as f64);
        for (written, &sym) in symbols.iter().enumerate() {
            if written.is_multiple_of(sub) {
                let t = ((written / sub) + 1) as f64 / LPC_SUBBLOCKS as f64;
                weights = interpolated_weights(&self.k_prev, &k_q, t);
            }
            let mut pred = 0.0f64;
            for (j, &w) in weights.iter().enumerate() {
                pred += w * self.hist[j];
            }
            if lag > 0 && lag <= self.excit.len() {
                pred += ltpg * self.excit[lag - 1];
            }
            let err = q.reconstruct(i64::from(sym), RECONSTRUCTION_OFFSET);
            let xh = pred + err;
            out.push(xh);
            for j in (1..self.order).rev() {
                self.hist[j] = self.hist[j - 1];
            }
            self.hist[0] = xh;
            for j in (1..self.excit.len()).rev() {
                self.excit[j] = self.excit[j - 1];
            }
            self.excit[0] = err;
        }
        self.k_prev = k_q;
        self.lag_prev = lag_q;
        self.ltpg_prev = ltpg_q;
        self.gain_prev = gain;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_step_round_trips() {
        for gain in -20..40 {
            let s = step_of(gain);
            assert_eq!(gain_of(s), gain, "gain {gain}");
        }
    }

    #[test]
    fn reflection_quantization_is_stable() {
        for i in -1000..=1000 {
            let k = i as f64 / 1000.0;
            let q = quantize_reflection(k);
            let back = dequantize_reflection(q);
            assert!((0..REFLECTION_LEVELS).contains(&q));
            assert!(back.abs() < 1.0, "k={k} back={back}");
        }
    }

    #[test]
    fn lpc_finds_an_ar_process() {
        // Well-conditioned AR(2): x[n] = 1.2 x[n-1] - 0.5 x[n-2] + e
        let mut state = 0x1234u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let mut x = vec![0.0f64; 8000];
        for n in 2..x.len() {
            x[n] = 1.2 * x[n - 1] - 0.5 * x[n - 2] + rnd();
        }
        let (w, k) = lpc_reflection(&x[..4000], 8, 16_000);
        // The reflection coefficients must be a stable set.
        for &ki in &k {
            assert!(ki.abs() < 1.0, "unstable reflection {ki}");
        }
        // The predictor must remove most of the signal.
        let (mut sig, mut res) = (0.0f64, 0.0f64);
        for n in 200..x.len() {
            let mut pred = 0.0f64;
            for (j, &wj) in w.iter().enumerate() {
                pred += wj * x[n - 1 - j];
            }
            let e = x[n] - pred;
            res += e * e;
            sig += x[n] * x[n];
        }
        let gain = 10.0 * (sig / res).log10();
        assert!(gain > 6.0, "LPC prediction gain {gain} dB");
    }

    #[test]
    fn pitch_search_finds_a_known_period() {
        // 100 Hz at 16 kHz = 160-sample period.
        let mut x = vec![0.0f64; 4000];
        for (i, slot) in x.iter_mut().enumerate() {
            let t = i as f64;
            *slot = 8000.0 * (2.0 * std::f64::consts::PI * t / 160.0).sin()
                + 2000.0 * (2.0 * std::f64::consts::PI * t / 40.0).sin();
        }
        let p = analyse_frame(&x, 2000, 320, 12, 32, 320, 0, 16_000);
        assert!(p.lag > 0, "lag {}", p.lag);
        assert!((p.lag - 160).abs() <= 1, "lag {}", p.lag);
        assert!(p.ltpg > 0.5, "gain {}", p.ltpg);
    }

    #[test]
    fn multi_frame_encoder_matches_decoder() {
        // A pitched, formant-shaped signal over several frames: the state carried
        // between frames must match exactly or the decoder drifts.
        let n = 320usize;
        let frames = 5usize;
        let len = n * frames;
        let mut x = vec![0.0f64; len + 480];
        for (i, slot) in x.iter_mut().enumerate() {
            let t = i as f64;
            *slot = 5000.0 * (2.0 * std::f64::consts::PI * t / 96.0).sin()
                + 2000.0 * (2.0 * std::f64::consts::PI * t * 0.11).sin();
        }
        let (min_lag, max_lag) = (32usize, 320usize);
        let mut enc = PredictEngine::new(12, min_lag, max_lag);
        let mut stream = Vec::new();
        let mut recon_enc: Vec<f64> = Vec::new();
        let mut lag_hint = 0i32;
        for f in 0..frames {
            let p = analyse_frame(&x, f * n, n, 12, min_lag, max_lag, lag_hint, 16_000);
            lag_hint = p.lag;
            let (cand, recon) = enc.commit(&x[f * n..(f + 1) * n], &p, 10);
            stream.extend_from_slice(&cand.record);
            recon_enc.extend_from_slice(&recon);
        }
        let mut dec = PredictEngine::new(12, min_lag, max_lag);
        let mut recon_dec: Vec<f64> = Vec::new();
        let mut pos = 0usize;
        for _ in 0..frames {
            let len_b = u16::from_le_bytes(stream[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let vec = crate::learned::residual_codec2::decode_encoding_v3(
                &stream[pos..pos + len_b],
                12 + 3 + n,
            )
            .unwrap()
            .decode(12 + 3 + n)
            .unwrap();
            pos += len_b;
            let gain = dec.last_gain() + vec[14];
            let r = dec.reconstruct(&vec[0..12], vec[12], vec[13], gain, &vec[15..]);
            recon_dec.extend_from_slice(&r);
        }
        for (i, (a, b)) in recon_enc.iter().zip(recon_dec.iter()).enumerate() {
            assert!((a - b).abs() < 1e-9, "sample {i}: {a} vs {b}");
        }
    }

    #[test]
    fn closed_loop_matches_decoder() {
        // A synthetic two-formant signal with a pitch period, so both predictors
        // are exercised.
        let mut x = vec![0.0f64; 640];
        for (i, slot) in x.iter_mut().enumerate() {
            let t = i as f64;
            *slot = 6000.0 * (2.0 * std::f64::consts::PI * t / 160.0).sin()
                + 2500.0 * (2.0 * std::f64::consts::PI * t * 0.07).sin();
        }
        let p = analyse_frame(&x, 0, x.len(), 12, 32, 320, 0, 16_000);
        let gain = 12;

        let mut enc = PredictEngine::new(12, 32, 320);
        let (cand, recon) = enc.commit(&x, &p, gain);

        // Decode the record exactly as the far end does.
        let enc_len = u16::from_le_bytes(cand.record[0..2].try_into().unwrap()) as usize;
        let vec = crate::learned::residual_codec2::decode_encoding_v3(
            &cand.record[2..2 + enc_len],
            12 + 3 + x.len(),
        )
        .unwrap()
        .decode(12 + 3 + x.len())
        .unwrap();
        let mut dec = PredictEngine::new(12, 32, 320);
        let decoded = dec.reconstruct(&vec[0..12], vec[12], vec[13], vec[14], &vec[15..]);

        assert_eq!(decoded.len(), recon.len());
        for (a, b) in decoded.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }
}

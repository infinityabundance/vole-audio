//! Voice-frame prediction for `vole.audio.stream.voice.exp1` (Phase 7C).
//!
//! The predictor is deliberately built only from machinery that already exists
//! in the crate, adapted to a *causal, bounded-frame* contract:
//!
//! ```text
//! short term   : bounded estimator competition
//!                autocorrelation+Tukey -> Levinson–Durbin
//!                Burg
//!                covariance least squares (stepped down to reflection form)
//!                order       ∈ {8, 10, 12, 14, 16}
//!                coefficient : uniform Q quantiser in the REFLECTION domain
//!                              width ∈ {5,6,7,8} bits, shift = width − 1
//! long term    : lag ∈ [32, 288] samples, gain ∈ [0, 1.5] (32 codes)
//!                reference = the RECONSTRUCTED EXCITATION ring
//! residual     : dead-zone scalar quantiser, centroid reconstruction
//! reconstruction: x̂[n] = P_w(x̂ history) + g·e[n-lag] + e[n]
//! ```
//!
//! Three properties are contract, not preference:
//!
//! * **Stability by construction.** Every quantised reflection coefficient is
//!   clamped to `|k| ≤ 1 − 2^-shift`, so the synthesis filter is minimum phase
//!   for *every* bit pattern a decoder can receive. No decoder-side stability
//!   check can fail, which is what a lossy streaming profile needs.
//! * **The long-term reference is the reconstructed *excitation*, never the
//!   reconstructed output.** Predicting from the output puts the long-term gain
//!   inside the feedback path of the short-term filter, and the cascade can
//!   diverge for perfectly legal parameters — a positive-feedback loop with gain
//!   above one. The excitation ring is bounded by the quantiser step, so the
//!   long-term contribution is bounded by construction. This is also what makes
//!   packet-loss concealment possible (§7C.5): repeating the excitation ring is
//!   exactly pitch-period repetition.
//! * **One shared synthesis loop.** [`synthesize`] is the encoder's inner loop
//!   and the decoder's whole job. There is no second implementation to drift.
//!
//! The entire decoder state is two bounded rings of reconstructed values, so a
//! lost packet can never desynchronise a later one: parameters are absolute, and
//! the contamination decays with the short-term memory (≤ order samples) plus
//! the excitation ring.

use crate::learned::train::lpc as tlpc;
use crate::lossy::predict as lp;

/// Short-term orders competed per frame (7C.2 bound: ≤ 16).
pub const ORDER_LADDER: [usize; 5] = [8, 10, 12, 14, 16];
/// Coefficient widths in bits (reflection domain). `shift = width - 1`.
pub const WIDTH_LADDER: [u8; 4] = [5, 6, 7, 8];
/// Number of long-term gain codes.
pub const LTPG_LEVELS: i32 = 32;
/// Largest long-term gain code.
pub const LTPG_MAX_CODE: i32 = LTPG_LEVELS - 1;
/// Smallest pitch lag searched, in samples (500 Hz at 16 kHz).
pub const MIN_LAG: usize = 32;
/// Largest pitch lag searched, in samples (55 Hz at 16 kHz).
pub const MAX_LAG: usize = 288;
/// Ring size of the reconstructed history and excitation (power of two).
pub const HIST_SLOTS: usize = 512;
/// Chronological depth of history the analysis may read.
pub const HIST_LEN: usize = MAX_LAG + 16;
/// Lower residual gain bound searched by the closed loop.
pub const GAIN_MIN: i32 = -4;
/// Upper residual gain bound searched by the closed loop.
pub const GAIN_MAX: i32 = 44;
/// Samples per residual-gain subframe. A single quantiser step per 20 ms frame
/// is the coarse-quantiser overload trap the standards avoid: AMR-WB, EVS and
/// SILK all carry one gain per 5 ms. 80 samples = 5 ms at 16 kHz.
pub const RESIDUAL_SUB_LEN: usize = 80;
/// Saturation applied to every reconstructed sample. A resonant synthesis
/// filter is minimum phase but can still ring far up on sustained excitation;
/// saturating keeps the ring finite **identically** on both sides, because both
/// run this same loop.
pub const SATURATION: f64 = 2.0e9;

/// Numerator bandwidth-expansion factor of the perceptual weighting filter.
pub const WEIGHT_GAMMA1: f64 = 0.9;
/// Denominator bandwidth-expansion factor of the perceptual weighting filter.
pub const WEIGHT_GAMMA2: f64 = 0.6;

/// Whether the closed loop minimises the *perceptually weighted* error.
///
/// The mechanism is implemented and tested ([`weighted_error_energy`]), and with
/// `γ₁ = γ₂ = 1` it reduces exactly to plain MSE, so this switch is the only
/// difference between the two objectives. It is **off** because it is measured to
/// lower the profile's declared metric: enabling it changes which model and
/// excitation are chosen toward perceptually better but MSE-worse reconstructions
/// (320×1 @24k falls 11.07 → 9.32 dB and @32k 15.52 → 13.34 dB on the court's
/// cases), and almost doubles the distortion-evaluation cost, pushing encode p99
/// to 6.8 ms against the 5 ms constitution. It should be re-opened together with
/// the court's ViSQOL column, which is the metric it is actually for.
pub const WEIGHTING: bool = false;

/// The `(γ₁, γ₂)` pair the closed loop uses.
pub fn weight_gammas() -> (f64, f64) {
    if WEIGHTING {
        (WEIGHT_GAMMA1, WEIGHT_GAMMA2)
    } else {
        (1.0, 1.0)
    }
}

/// Perceptually weighted error energy of a reconstruction.
///
/// Plain MSE spends bits on error the ear never hears (formant peaks) and starves
/// the spectral valleys where it does. CELP therefore minimises the error through
/// the weighting filter
///
/// ```text
/// W(z) = A(z/γ₁) / A(z/γ₂),   A(z) = 1 − Σ_j w_j z^-j
/// ```
/// which de-emphasises the formants. The filter is encoder-only: the decoder is
/// unchanged and no bit is spent on it. Implementing it as
/// `(1 − Σ w_j γ₁^j z^-j)·E = (1 − Σ w_j γ₂^j z^-j)·E_w` gives the recursion
/// `e_w[n] = e[n] − Σ_j w_j γ₁^j e[n−j] + Σ_j w_j γ₂^j e_w[n−j]`.
///
/// The filter memory starts at zero for each call, so every candidate for the same
/// span is scored on identical footing.
pub fn weighted_error_energy(
    target: &[f64],
    out: &[f64],
    w: &[f64],
    gamma1: f64,
    gamma2: f64,
) -> f64 {
    let order = w.len();
    // `γ₁ = γ₂` makes `W(z) = 1`, so the weighted error is the plain error. Taking
    // that branch keeps the disabled switch free rather than paying the filter.
    if order == 0 || (gamma1 - gamma2).abs() < 1e-12 {
        return target
            .iter()
            .zip(out.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
    }
    let mut gamma1_pow = vec![0.0f64; order];
    let mut gamma2_pow = vec![0.0f64; order];
    let (mut a1, mut a2) = (gamma1, gamma2);
    for j in 0..order {
        gamma1_pow[j] = a1;
        gamma2_pow[j] = a2;
        a1 *= gamma1;
        a2 *= gamma2;
    }
    let mut e_hist = vec![0.0f64; order];
    let mut ew_hist = vec![0.0f64; order];
    let mut acc = 0.0f64;
    let n = target.len().min(out.len());
    for i in 0..n {
        let e = target[i] - out[i];
        let mut num = e;
        for (j, &wj) in w.iter().enumerate() {
            num -= wj * gamma1_pow[j] * e_hist[j];
        }
        let mut ew = num;
        for (j, &wj) in w.iter().enumerate() {
            ew += wj * gamma2_pow[j] * ew_hist[j];
        }
        acc += ew * ew;
        if order > 1 {
            e_hist.copy_within(0..order - 1, 1);
            ew_hist.copy_within(0..order - 1, 1);
        }
        e_hist[0] = e;
        ew_hist[0] = ew;
    }
    acc
}

/// Reflection coefficients of a `width`-bit quantiser have `shift = width - 1`.
pub fn shift_of(width: u8) -> u8 {
    width.saturating_sub(1)
}

/// One competed frame model (everything the decoder needs but the symbols).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameModel {
    /// 0 autocorrelation+Levinson, 1 Burg, 2 covariance least squares.
    pub estimator: u8,
    /// Short-term order.
    pub order: usize,
    /// Coefficient width in bits.
    pub width: u8,
    /// Pitch lag in samples; 0 means "no long-term predictor".
    pub lag: i32,
    /// Quantised long-term gain.
    pub ltpg_q: i32,
}

/// The `estimator` value reserved for the vector-quantised spectral path. The
/// scalar estimators are 0..=2, so 3 is free and needs no extra mode bit; when
/// it is set, `order` is [`crate::voice::vq::VQ_ORDER`] and `width` is
/// [`crate::voice::vq::VQ_INTERNAL_WIDTH`] (an internal synthesis width, not a
/// wire field).
pub const VQ_ESTIMATOR: u8 = 3;

impl FrameModel {
    /// True when the spectrum is transmitted as a vector-quantiser index.
    pub fn is_vq(&self) -> bool {
        self.estimator == VQ_ESTIMATOR
    }

    /// Bytes the model description occupies in a frame record.
    pub fn description_bytes(&self) -> usize {
        let k_bytes = if self.is_vq() {
            crate::voice::vq::VQ_INDEX_BYTES
        } else {
            (self.order * usize::from(self.width)).div_ceil(8)
        };
        1 + 1 + if self.lag > 0 { 3 } else { 0 } + k_bytes
    }
}

/// The bounded decoder state: reconstructed output history (for the short-term
/// filter) and reconstructed excitation (for the long-term predictor).
#[derive(Debug, Clone)]
pub struct VoiceState {
    hist: [f64; HIST_SLOTS],
    excit: [f64; HIST_SLOTS],
    head: usize,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceState {
    /// A zeroed state (a cold decoder, and the encoder's start of stream).
    pub fn new() -> VoiceState {
        VoiceState {
            hist: [0.0; HIST_SLOTS],
            excit: [0.0; HIST_SLOTS],
            head: 0,
        }
    }

    /// Wipe the state. Used by the court when a decoder is started cold.
    pub fn reset(&mut self) {
        self.hist = [0.0; HIST_SLOTS];
        self.excit = [0.0; HIST_SLOTS];
        self.head = 0;
    }

    /// The `j`-th most recent reconstructed sample.
    #[inline]
    pub fn at(&self, j: usize) -> f64 {
        self.hist[(self.head.wrapping_sub(j)) & (HIST_SLOTS - 1)]
    }

    /// The `j`-th most recent reconstructed excitation value.
    #[inline]
    pub fn excitation(&self, j: usize) -> f64 {
        self.excit[(self.head.wrapping_sub(j)) & (HIST_SLOTS - 1)]
    }

    /// Push one reconstructed sample and the excitation that produced it.
    #[inline]
    pub fn push(&mut self, x: f64, e: f64) {
        self.head = (self.head + 1) & (HIST_SLOTS - 1);
        self.hist[self.head] = x;
        self.excit[self.head] = e;
    }

    /// The `n` most recent reconstructed output samples, oldest first.
    pub fn chronological(&self, n: usize) -> Vec<f64> {
        let n = n.min(HIST_SLOTS);
        (0..n).map(|k| self.at(n - 1 - k)).collect()
    }

    /// The `n` most recent excitation values, oldest first.
    pub fn excitation_chronological(&self, n: usize) -> Vec<f64> {
        let n = n.min(HIST_SLOTS);
        (0..n).map(|k| self.excitation(n - 1 - k)).collect()
    }
}

/// A closed-loop evaluation of one model at one residual gain.
#[derive(Debug, Clone)]
pub struct Synthesis {
    /// Reconstructed samples the decoder will reproduce.
    pub samples: Vec<f64>,
    /// Squared error against the source frame.
    pub distortion: f64,
    /// Quantised residual symbols to transmit.
    pub symbols: Vec<i32>,
}

/// Convert a quantised reflection vector back to synthesis weights.
pub fn weights_of(k_q: &[i32], width: u8) -> Vec<f64> {
    let scale = 1.0 / f64::from(1u32 << shift_of(width));
    let k: Vec<f64> = k_q.iter().map(|&q| f64::from(q) * scale).collect();
    lp::reflection_to_weights(&k)
}

fn long_term_gain(ltpg_q: i32) -> f64 {
    lp::dequantize_ltpg(ltpg_q.clamp(0, LTPG_MAX_CODE))
}

fn effective_lag(lag: i32) -> usize {
    if lag <= 0 {
        0
    } else {
        (lag as usize).clamp(MIN_LAG, MAX_LAG)
    }
}

/// Synthesis from an explicit excitation stream with a **per-subframe** long-term
/// predictor.
///
/// This is the single reconstruction primitive and the CELP loop. The adaptive
/// term is the adaptive codebook: `gₐ(s)·u[n−lag(s)]` where `u` is the *total*
/// excitation ring (previous adaptive plus fixed contributions), so the predictor
/// accumulates a periodic waveform across frames instead of reading only the
/// fixed part. `sub_len` fixes which subframe's `(lag, gain)` applies.
pub fn synthesize_celp(
    state: &mut VoiceState,
    k_q: &[i32],
    width: u8,
    sub_len: usize,
    pitch: &[(i32, i32)],
    fixed: &[f64],
) -> Vec<f64> {
    let w = weights_of(k_q, width);
    let sub_len = sub_len.max(1);
    let mut out = Vec::with_capacity(fixed.len());
    for (n, &e) in fixed.iter().enumerate() {
        let (lag_i, ltpg_q) = pitch.get(n / sub_len).copied().unwrap_or((0, 0));
        let ltpg = long_term_gain(ltpg_q);
        let lag = effective_lag(lag_i);
        let mut st = 0.0f64;
        for (j, &wj) in w.iter().enumerate() {
            st += wj * state.at(j);
        }
        let adaptive = if lag > 0 {
            ltpg * state.excitation(lag - 1)
        } else {
            0.0
        };
        let u = (adaptive + e).clamp(-SATURATION, SATURATION);
        let xh = (st + u).clamp(-SATURATION, SATURATION);
        state.push(xh, u);
        out.push(xh);
    }
    out
}

/// The uniform-pitch special case: one `(lag, gain)` for the whole span. The
/// scalar path and concealment use this.
pub fn synthesize_excitation(
    state: &mut VoiceState,
    k_q: &[i32],
    width: u8,
    lag: i32,
    ltpg_q: i32,
    excitation: &[f64],
) -> Vec<f64> {
    let pitch = [(lag, ltpg_q)];
    synthesize_celp(
        state,
        k_q,
        width,
        excitation.len().max(1),
        &pitch,
        excitation,
    )
}

/// The residual gain code in force at sample `n`.
#[inline]
fn gain_at(gains: &[i32], sub_len: usize, n: usize) -> i32 {
    if gains.is_empty() {
        return GAIN_MAX;
    }
    gains[(n / sub_len.max(1)).min(gains.len() - 1)]
}

/// Subframe count for a frame of `n` samples at the residual-gain geometry.
pub fn gain_subframes(n: usize) -> usize {
    n.div_ceil(RESIDUAL_SUB_LEN).max(1)
}

/// Bits of the per-subframe gain block: one 4-bit delta per subframe after the
/// first. The first gain rides in the frame's own gain byte.
pub fn gain_block_bits(n: usize) -> usize {
    gain_subframes(n).saturating_sub(1) * 4
}

/// Bytes of the per-subframe gain block.
pub fn gain_block_bytes(n: usize) -> usize {
    gain_block_bits(n).div_ceil(8)
}

/// Largest magnitude a chained 4-bit gain delta can carry.
pub const GAIN_DELTA_LIMIT: i32 = 7;

/// Pack the subframe gains after the first as chained 4-bit signed deltas,
/// MSB-first. The encoder must reconstruct from the *quantised* deltas, so a
/// value that cannot be represented is clamped here and again by the decoder.
pub fn encode_gain_deltas(gains: &[i32]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0u32;
    for pair in gains.windows(2) {
        let d = (pair[1] - pair[0]).clamp(-GAIN_DELTA_LIMIT - 1, GAIN_DELTA_LIMIT);
        acc = (acc << 4) | ((d as u32) & 0xF);
        bits += 4;
        if bits == 8 {
            out.push(acc as u8);
            acc = 0;
            bits = 0;
        }
    }
    if bits > 0 {
        out.push((acc << (8 - bits)) as u8);
    }
    out
}

/// Inverse of [`encode_gain_deltas`], returning exactly `nsub` gains.
pub fn decode_gain_deltas(base: i32, bytes: &[u8], nsub: usize) -> Vec<i32> {
    let mut gains = Vec::with_capacity(nsub);
    gains.push(base);
    let mut pos = 0usize;
    for _ in 1..nsub {
        let byte = bytes.get(pos >> 3).copied().unwrap_or(0);
        let nibble = (byte >> (4 - (pos & 4))) & 0xF;
        pos += 4;
        let d = if nibble >= 8 {
            i32::from(nibble) - 16
        } else {
            i32::from(nibble)
        };
        let prev = *gains.last().unwrap();
        gains.push((prev + d).clamp(GAIN_MIN, GAIN_MAX));
    }
    gains
}

/// Reconstruction from a symbol stream whose quantiser step may change per
/// subframe.
///
/// The step at sample `n` is `step_of(gains[n / sub_len])` (clamped to the last
/// entry). `sub_len == usize::MAX` collapses to the single-gain case, which is
/// what [`synthesize`] passes. Encoder and decoder run this identical loop, so a
/// per-subframe gain cannot desynchronise them.
#[allow(clippy::too_many_arguments)]
pub fn synthesize_gains(
    state: &mut VoiceState,
    k_q: &[i32],
    width: u8,
    sub_len: usize,
    gains: &[i32],
    lag: i32,
    ltpg_q: i32,
    symbols: &[i32],
) -> Vec<f64> {
    let excitation: Vec<f64> = symbols
        .iter()
        .enumerate()
        .map(|(n, &s)| {
            let q = lp::quantizer(lp::step_of(gain_at(gains, sub_len, n)));
            q.reconstruct(i64::from(s), lp::RECONSTRUCTION_OFFSET)
        })
        .collect();
    synthesize_excitation(state, k_q, width, lag, ltpg_q, &excitation)
}

/// The shared synthesis loop used by the encoder's probe and the decoder.
#[allow(clippy::too_many_arguments)]
pub fn synthesize(
    state: &mut VoiceState,
    k_q: &[i32],
    width: u8,
    lag: i32,
    ltpg_q: i32,
    gain: i32,
    symbols: &[i32],
) -> Vec<f64> {
    synthesize_gains(state, k_q, width, usize::MAX, &[gain], lag, ltpg_q, symbols)
}

/// Quantise a reflection vector, guaranteeing `|k| < 1` for every code.
pub fn quantise_k(k: &[f64], width: u8) -> Vec<i32> {
    tlpc::quantize_reflections(k, shift_of(width))
}

/// Clamp a fixed-width reflection code vector to the strictly-inside range.
///
/// A two's-complement `width`-bit field admits `-2^(width-1)`, which dequantises
/// to `k = -1` — a marginally stable filter that would ring without bound. The
/// encoder never emits it, but a hostile or corrupt packet could, so the decoder
/// clamps before the coefficients ever reach a filter.
pub fn clamp_k_codes(k_q: &[i32], width: u8) -> Vec<i32> {
    let limit = (1i32 << shift_of(width)) - 1;
    k_q.iter().map(|&q| q.clamp(-limit, limit)).collect()
}

/// The chronological analysis block: reconstructed history followed by the
/// frame's source samples. Estimators see exactly this.
pub fn analysis_block(state: &VoiceState, frame: &[i32]) -> Vec<i32> {
    let mut block: Vec<i32> = state
        .chronological(HIST_LEN)
        .into_iter()
        .map(|v| v.round().clamp(-2.0e9, 2.0e9) as i32)
        .collect();
    block.extend_from_slice(frame);
    block
}

/// Step a direct-form **predictor** coefficient vector down to reflection
/// coefficients. `None` when the filter is not minimum phase (an unstable
/// least-squares fit), in which case the candidate is rejected rather than
/// transmitted.
///
/// The recursion is the exact inverse of
/// `c_j^(p) = c_j^(p-1) − k_p·c_{p−j}^(p-1)`.
pub fn predictor_to_reflections(c: &[f64]) -> Option<Vec<f64>> {
    let mut a = c.to_vec();
    let p = a.len();
    let mut k = vec![0.0f64; p];
    for order in (1..=p).rev() {
        let kp = a[order - 1];
        if !kp.is_finite() || kp.abs() >= 0.999_9 {
            return None;
        }
        k[order - 1] = kp;
        let den = 1.0 - kp * kp;
        if den <= 1e-12 {
            return None;
        }
        // The recursion reads the *current* level, so it must not see values
        // rewritten during this iteration.
        let prev = a.clone();
        for j in 1..order {
            a[j - 1] = (prev[j - 1] + kp * prev[order - 1 - j]) / den;
        }
    }
    Some(k)
}

/// Full-order reflection vectors from each of the three estimators, in the
/// lossy-engine sign convention that [`lp::reflection_to_weights`] expects.
fn estimator_reflections(block: &[i32], order: usize) -> [Option<Vec<f64>>; 3] {
    let alpha = 0.25;
    let r = tlpc::autocorrelation(block, order, alpha);
    let lev = {
        let k = tlpc::levinson_reflections(&r, order);
        if k.len() == order {
            Some(k.into_iter().map(|v| -v).collect::<Vec<f64>>())
        } else {
            None
        }
    };
    let burg = {
        let k = tlpc::burg_reflections(block, order);
        if k.len() == order {
            Some(k.into_iter().map(|v| -v).collect::<Vec<f64>>())
        } else {
            None
        }
    };
    let lsq = tlpc::lsq_coefficients(block, order)
        .and_then(|c| predictor_to_reflections(&c))
        .map(|k| k.into_iter().map(|v| -v).collect::<Vec<f64>>());
    [lev, burg, lsq]
}

/// Open-loop residual energy of `frame` under a reflection vector, over the
/// frame only, using the reconstructed history as left context.
fn open_loop_energy(state: &VoiceState, frame: &[f64], k: &[f64], width: u8) -> f64 {
    let k_q = quantise_k(k, width);
    let w = weights_of(&k_q, width);
    let mut energy = 0.0f64;
    for (n, &x) in frame.iter().enumerate() {
        let mut pred = 0.0f64;
        for (j, &wj) in w.iter().enumerate() {
            let t = n as i64 - 1 - j as i64;
            let v = if t >= 0 {
                frame[t as usize]
            } else {
                state.at((-t - 1) as usize)
            };
            pred += wj * v;
        }
        let r = x - pred;
        energy += r * r;
    }
    energy
}

/// Open-loop pitch search on the short-term residual, over the owned history.
///
/// The residual is computed on `state`'s history *and* the frame, so lags up to
/// [`MAX_LAG`] are reachable even with a 160-sample frame. `lag_hint` biases the
/// search toward the previous frame's lag so the tracker follows a voice instead
/// of jumping between octave multiples of the true period.
fn pitch_search(
    state: &VoiceState,
    frame: &[f64],
    k: &[f64],
    width: u8,
    lag_hint: i32,
) -> (i32, f64) {
    let k_q = quantise_k(k, width);
    let w = weights_of(&k_q, width);
    let n = frame.len();
    let ctx = HIST_LEN;
    let hist = state.chronological(ctx);
    let mut resid = Vec::with_capacity(ctx + n);
    for (i, &h) in hist.iter().enumerate() {
        let mut pred = 0.0f64;
        for (j, &wj) in w.iter().enumerate() {
            if i > j {
                pred += wj * hist[i - 1 - j];
            }
        }
        resid.push(h - pred);
    }
    for (i, &x) in frame.iter().enumerate() {
        let mut pred = 0.0f64;
        for (j, &wj) in w.iter().enumerate() {
            let t = i as i64 - 1 - j as i64;
            let v = if t >= 0 {
                frame[t as usize]
            } else {
                hist[(ctx as i64 + t) as usize]
            };
            pred += wj * v;
        }
        resid.push(x - pred);
    }
    let base = ctx;
    let mut energy = 0.0f64;
    for &v in &resid[base..] {
        energy += v * v;
    }
    let mut best = (0i32, 0.0f64, 0.0f64);
    let upper = MAX_LAG.min(base);
    for lag in MIN_LAG..=upper {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in base..resid.len() {
            let s = resid[i - lag];
            num += resid[i] * s;
            den += s * s;
        }
        if den <= 1e-12 {
            continue;
        }
        // Continuity bias: a small penalty per lag of deviation keeps the
        // tracker on one pitch track when several lags correlate comparably
        // (which happens for every multiple of the true period).
        let bias = if lag_hint > 0 {
            let d = (lag as f64 - f64::from(lag_hint)).abs();
            0.05 * d / f64::from(lag_hint.max(1)) * num.abs() / den.max(1e-12).sqrt()
        } else {
            0.0
        };
        let score = num.abs() / den.sqrt() - bias;
        if score > best.2 {
            best = (lag as i32, (num / den).clamp(0.0, 1.5), score);
        }
    }
    // The long-term predictor must clearly beat the short-term residual alone:
    // score²/energy is the fraction of residual energy it explains.
    if best.0 == 0 || best.1 < 0.15 || best.2 * best.2 < 0.05 * energy {
        (0, 0.0)
    } else {
        (best.0, best.1)
    }
}

/// One analysed candidate before the closed-loop gain search.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The model proposal.
    pub model: FrameModel,
    /// Quantised reflection vector at the proposal width.
    pub k_q: Vec<i32>,
    /// Unquantised reflection vector this candidate came from, at `model.order`.
    /// The spectral vector quantiser needs it: converting the coarse width-6
    /// codes would feed the codebook a spectrum the encoder never saw.
    pub k_raw: Vec<f64>,
    /// Open-loop residual energy (ranking only, never a decision).
    pub energy: f64,
}

/// Bounded analysis: propose the candidate ladder for one frame.
///
/// Ordering is by open-loop residual energy, which is a *proposal* mechanism
/// only — every accept/reject decision downstream is made on complete physical
/// bytes and closed-loop distortion.
pub fn analyse(state: &VoiceState, frame: &[i32], lag_hint: i32) -> Vec<Candidate> {
    let f: Vec<f64> = frame.iter().map(|&v| f64::from(v)).collect();
    let max_order = *ORDER_LADDER.last().unwrap();
    let block = analysis_block(state, frame);
    let est = estimator_reflections(&block, max_order);
    let mut out: Vec<Candidate> = Vec::new();
    for (e, refl) in est.iter().enumerate() {
        let Some(k) = refl else { continue };
        for &order in &ORDER_LADDER {
            let sub = &k[..order];
            // Rank at the mid width: the width sweep is a coding decision, not
            // a prediction-quality decision.
            let energy = open_loop_energy(state, &f, sub, 6);
            let (lag, ltpg) = pitch_search(state, &f, sub, 6, lag_hint);
            out.push(Candidate {
                model: FrameModel {
                    estimator: e as u8,
                    order,
                    width: 6,
                    lag,
                    ltpg_q: if lag > 0 { lp::quantize_ltpg(ltpg) } else { 0 },
                },
                k_q: quantise_k(sub, 6),
                k_raw: sub.to_vec(),
                energy,
            });
        }
    }
    out
}

/// Closed-loop evaluation of one candidate at one residual gain.
///
/// The state is *cloned*, exactly as the decoder would run it, so the returned
/// distortion is the distortion of the audio the decoder will produce.
pub fn close_loop(
    state: &VoiceState,
    frame: &[f64],
    model: &FrameModel,
    k_q: &[i32],
    gain: i32,
) -> Synthesis {
    close_loop_gains(state, frame, model, k_q, usize::MAX, &[gain])
}

/// Closed-loop evaluation with a quantiser step that may change per subframe.
///
/// This is the gain-normalised excitation search the standards use to escape the
/// coarse-quantiser overload of a single frame-wide step: the step tracks the
/// residual level within 5 ms instead of being set by the loudest moment of the
/// whole frame. `sub_len == usize::MAX` is the uniform-gain case.
pub fn close_loop_gains(
    state: &VoiceState,
    frame: &[f64],
    model: &FrameModel,
    k_q: &[i32],
    sub_len: usize,
    gains: &[i32],
) -> Synthesis {
    let mut probe = state.clone();
    let mut symbols = Vec::with_capacity(frame.len());
    let mut samples = Vec::with_capacity(frame.len());
    let w = weights_of(k_q, model.width);
    let ltpg = long_term_gain(model.ltpg_q);
    let lag = effective_lag(model.lag);
    for (n, &x) in frame.iter().enumerate() {
        let q = lp::quantizer(lp::step_of(gain_at(gains, sub_len, n)));
        let mut st = 0.0f64;
        for (j, &wj) in w.iter().enumerate() {
            st += wj * probe.at(j);
        }
        let adaptive = if lag > 0 {
            ltpg * probe.excitation(lag - 1)
        } else {
            0.0
        };
        let target = x - st - adaptive;
        let sym = q.symbol(target).clamp(-lp::MAX_SYMBOL, lp::MAX_SYMBOL);
        // The distortion must describe the *transmitted* symbol, or the rate
        // search would be comparing two different quantisers.
        let fixed = q.reconstruct(sym, lp::RECONSTRUCTION_OFFSET);
        let u = (adaptive + fixed).clamp(-SATURATION, SATURATION);
        let xh = (st + u).clamp(-SATURATION, SATURATION);
        probe.push(xh, u);
        symbols.push(sym as i32);
        samples.push(xh);
    }
    // The distortion is the *perceptually weighted* error energy (plain MSE when
    // weighting is off), and it is encoder-only.
    let (g1, g2) = weight_gammas();
    let distortion = weighted_error_energy(frame, &samples, &w, g1, g2);
    Synthesis {
        samples,
        distortion,
        symbols,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_quantisation_keeps_every_code_stable() {
        for width in WIDTH_LADDER {
            let shift = shift_of(width);
            let limit = (1i64 << shift) - 1;
            for &q in &[-limit, limit] {
                let k = f64::from(q as i32) / f64::from(1u32 << shift);
                assert!(
                    k.abs() < 1.0,
                    "width {width} code {q} gives |k| = {}",
                    k.abs()
                );
            }
            assert_eq!(shift + 1, width);
        }
    }

    #[test]
    fn every_wire_code_yields_a_strictly_stable_filter() {
        for width in WIDTH_LADDER {
            let bits = usize::from(width);
            let lo = -(1i32 << (bits - 1));
            let hi = (1i32 << (bits - 1)) - 1;
            let scale = 1.0 / f64::from(1u32 << shift_of(width));
            for code in lo..=hi {
                let clamped = clamp_k_codes(&[code; 4], width);
                let c0 = f64::from(clamped[0]) * scale;
                assert!(
                    c0.abs() < 1.0,
                    "width {width} code {code} gives |k| = {}",
                    c0.abs()
                );
                if code == lo {
                    // The unclamped extreme really is the marginally stable one,
                    // so the clamp is doing work rather than being decorative.
                    assert!((f64::from(code) * scale).abs() == 1.0);
                }
            }
        }
    }

    #[test]
    fn predictor_to_reflections_inverts_the_levinson_step_up() {
        let k = [0.62f64, -0.41, 0.28, -0.13, 0.07];
        let mut a = vec![0.0f64; k.len() + 1];
        for i in 1..=k.len() {
            let ki = k[i - 1];
            let old = a.clone();
            for j in 1..i {
                a[j] = old[j] - ki * old[i - j];
            }
            a[i] = ki;
        }
        let predictors: Vec<f64> = (1..=k.len()).map(|j| a[j]).collect();
        let back = predictor_to_reflections(&predictors).expect("stable filter");
        for (got, want) in back.iter().zip(k.iter()) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    #[test]
    fn predictor_to_reflections_rejects_an_unstable_filter() {
        assert!(predictor_to_reflections(&[1.5]).is_none());
        assert!(predictor_to_reflections(&[0.5, 1.2]).is_none());
    }

    #[test]
    fn encoder_and_decoder_synthesis_agree_exactly() {
        let mut enc = VoiceState::new();
        let mut dec = VoiceState::new();
        let k_q = quantise_k(&[0.5, -0.25, 0.125, 0.0625], 6);
        for n in 0..8 {
            enc.push(f64::from(n - 4), 0.25);
            dec.push(f64::from(n - 4), 0.25);
        }
        let symbols: Vec<i32> = (0..160).map(|i| (i % 7) - 3).collect();
        let a = synthesize(&mut enc, &k_q, 6, 73, 20, 6, &symbols);
        let b = synthesize(&mut dec, &k_q, 6, 73, 20, 6, &symbols);
        assert_eq!(a, b);
        for j in 0..HIST_LEN {
            assert_eq!(enc.at(j), dec.at(j));
            assert_eq!(enc.excitation(j), dec.excitation(j));
        }
    }

    #[test]
    fn close_loop_matches_shared_synthesis_exactly() {
        let mut s = VoiceState::new();
        for n in 0..HIST_LEN {
            s.push(
                ((n as f64) * 0.37).sin() * 3000.0,
                ((n as f64) * 0.11).cos() * 40.0,
            );
        }
        let k_q = quantise_k(&[0.44, -0.19, 0.09, 0.03, -0.02, 0.01, 0.0, 0.0], 6);
        let model = FrameModel {
            estimator: 0,
            order: 8,
            width: 6,
            lag: 71,
            ltpg_q: 18,
        };
        let frame: Vec<f64> = (0..160)
            .map(|n| ((n as f64) * 0.21).sin() * 5000.0)
            .collect();
        let synth = close_loop(&s, &frame, &model, &k_q, 8);

        let mut dec = s.clone();
        let out = synthesize(&mut dec, &k_q, 6, 71, 18, 8, &synth.symbols);
        assert_eq!(out, synth.samples);
        // The distortion is the perceptually weighted error energy, so it must be
        // reproduced by the shared weighting filter, not by a plain MSE sum.
        let w = weights_of(&k_q, 6);
        let (g1, g2) = weight_gammas();
        let e = weighted_error_energy(&frame, &synth.samples, &w, g1, g2);
        assert!((e - synth.distortion).abs() < 1e-6);
    }

    #[test]
    fn synthesis_saturates_instead_of_diverging() {
        let mut state = VoiceState::new();
        for i in 0..HIST_SLOTS {
            state.push(2.0e8 * f64::from(1 + (i as i32 % 2)), 1.0e6);
        }
        let k_q = quantise_k(&[0.95, -0.9, 0.85, -0.8], 6);
        let excitation = vec![1_000_000.0f64; 400];
        let out = synthesize_excitation(&mut state, &k_q, 6, 0, 0, &excitation);
        for &v in &out {
            assert!(v.is_finite() && v.abs() <= SATURATION);
        }
    }

    #[test]
    fn a_maximum_long_term_gain_cannot_diverge() {
        // Predicting from the reconstructed *output* would let this loop grow;
        // predicting from the bounded excitation ring cannot.
        let mut state = VoiceState::new();
        for _ in 0..HIST_LEN {
            state.push(30_000.0, 30_000.0);
        }
        // A near-unity reflection set with the long-term predictor at full gain.
        let k_q = quantise_k(&[0.94, -0.88, 0.82, -0.76, 0.7, -0.64, 0.58, -0.5], 6);
        let excitation: Vec<f64> = (0..800).map(|_| 4000.0).collect();
        let out = synthesize_excitation(&mut state, &k_q, 6, 80, LTPG_MAX_CODE, &excitation);
        let peak = out.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(peak.is_finite(), "the loop must not diverge");
        assert!(
            peak <= 4.0e8,
            "the loop grew to {peak}, far beyond the excitation scale"
        );
    }

    #[test]
    fn voice_state_rings_read_back_in_order() {
        let mut s = VoiceState::new();
        for i in 1..=40 {
            s.push(f64::from(i), f64::from(-i));
        }
        assert_eq!(s.at(0), 40.0);
        assert_eq!(s.at(39), 1.0);
        assert_eq!(s.chronological(4), vec![37.0, 38.0, 39.0, 40.0]);
        assert_eq!(s.excitation(0), -40.0);
        assert_eq!(
            s.excitation_chronological(4),
            vec![-37.0, -38.0, -39.0, -40.0]
        );
    }

    #[test]
    fn analysis_proposes_a_candidate_for_every_live_estimator() {
        let mut s = VoiceState::new();
        for n in 0..HIST_LEN {
            let t = n as f64;
            s.push(8000.0 * (t * 0.11).sin() + 200.0 * (t * 1.7).sin(), 0.0);
        }
        let frame: Vec<i32> = (0..320)
            .map(|n| (8000.0 * ((n as f64) * 0.13).sin()) as i32)
            .collect();
        let cands = analyse(&s, &frame, 0);
        assert!(!cands.is_empty());
        for c in &cands {
            assert!(ORDER_LADDER.contains(&c.model.order));
            assert_eq!(c.k_q.len(), c.model.order);
        }
        assert!(cands.iter().any(|c| c.model.estimator == 0));
        assert!(cands.iter().any(|c| c.model.estimator == 1));
    }

    #[test]
    fn the_pitch_tracker_prefers_the_hint_over_an_octave_multiple() {
        // A perfectly periodic residual correlates equally at 80, 160 and 240;
        // the continuity bias must keep the tracker at 80.
        let mut s = VoiceState::new();
        let period = 80usize;
        for n in 0..HIST_LEN {
            let v = if n % period == 0 { 8000.0 } else { 0.0 };
            s.push(v, v);
        }
        let frame: Vec<f64> = (0..320)
            .map(|n| if n % period == 0 { 8000.0 } else { 0.0 })
            .collect();
        let k = [0.0f64; 8];
        let (lag, _) = pitch_search(&s, &frame, &k, 6, 80);
        assert_eq!(lag, period as i32);
    }
}

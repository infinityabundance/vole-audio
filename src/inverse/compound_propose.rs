//! **Blind inverse proposal of Compound composition graphs** (Phase 7A.1).
//!
//! The `compound` court demonstrates that hand-authored graphs are compact. This
//! module closes the other half of the loop: given **only** observed samples, a
//! sample rate and an extent, it must *discover* a useful deterministic
//! composition graph. It never receives fixture identity, the source graph,
//! hidden oscillator parameters, construction metadata or an expected answer.
//!
//! ## Analysis has zero authority
//!
//! Everything in this module is a *proposal*: autocorrelation fundamental
//! estimation, harmonic projection, envelope/onset fitting and echo detection
//! run in floating point and may be wrong. The emitted [`CompoundGraph`] is a
//! deterministic integer object, and the caller is responsible for verifying it
//! exactly (`materialize` + an exact residual). A different graph plus a residual
//! that reconstructs the input and costs fewer bytes is always the winner — no
//! graph identity is ever required.
//!
//! ## Bounded and deterministic
//!
//! The search is a fixed, bounded portfolio: no randomness, no tuning tables,
//! no fixture dispatch. Every family is derived from closed-form statistics of
//! the input, and the candidate count is bounded by [`CompoundBudget`].

use crate::compound::{CompoundGraph, CompoundNode, CompoundOp, MAX_COMPOUND_ARITY};
use crate::error::{Error, Result};

/// Default highest fundamental searched (Hz).
pub const DEFAULT_MIN_FREQ_HZ: u32 = 40;
/// Default highest fundamental searched (Hz).
pub const DEFAULT_MAX_FREQ_HZ: u32 = 4_000;

/// Hard ceiling on independent tones in one decomposition.
const MAX_TONES: usize = 16;

/// Bounded blind-proposal budget. Every field is an explicit ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompoundBudget {
    /// Hard ceiling on proposed graphs.
    pub max_candidates: usize,
    /// Largest harmonic bank (number of oscillators) proposed, and the ceiling
    /// on independent tones in one decomposition.
    pub max_harmonics: usize,
    /// Largest analysis window (frames) used for projection.
    pub analysis_frames: usize,
    /// Lowest fundamental candidate (Hz).
    pub min_freq_hz: u32,
    /// Highest fundamental candidate (Hz).
    pub max_freq_hz: u32,
    /// Number of energy bands used for envelope fitting.
    pub envelope_bands: usize,
}

impl Default for CompoundBudget {
    fn default() -> Self {
        CompoundBudget {
            max_candidates: 10,
            max_harmonics: 24,
            analysis_frames: 8_192,
            min_freq_hz: DEFAULT_MIN_FREQ_HZ,
            max_freq_hz: DEFAULT_MAX_FREQ_HZ,
            envelope_bands: 64,
        }
    }
}

impl CompoundBudget {
    /// Validate the budget (a zero candidate ceiling is meaningless because the
    /// universal literal fallback is not this module's responsibility, but a
    /// zero ceiling here would silently propose nothing).
    pub fn validate(&self) -> Result<()> {
        if self.max_candidates == 0 {
            return Err(Error::malformed(
                "compound proposal budget max_candidates must be >= 1",
            ));
        }
        if self.min_freq_hz == 0 || self.min_freq_hz >= self.max_freq_hz {
            return Err(Error::malformed(
                "compound proposal frequency range is empty",
            ));
        }
        if self.analysis_frames < 64 {
            return Err(Error::malformed(
                "compound proposal analysis window is too short",
            ));
        }
        Ok(())
    }
}

/// Discovered composition family (evidence label only — never decoder state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompoundFamily {
    Silence,
    Constant,
    Oscillator,
    GainOscillator,
    EnvelopedOscillator,
    HarmonicBank,
    EnvelopedHarmonicBank,
    ToneDecomposition,
    EnvelopedToneDecomposition,
    EchoicBank,
}

impl CompoundFamily {
    /// Stable evidence label.
    pub const fn name(self) -> &'static str {
        match self {
            CompoundFamily::Silence => "silence",
            CompoundFamily::Constant => "constant",
            CompoundFamily::Oscillator => "oscillator",
            CompoundFamily::GainOscillator => "gain(oscillator)",
            CompoundFamily::EnvelopedOscillator => "envelope(oscillator)",
            CompoundFamily::HarmonicBank => "harmonic_bank",
            CompoundFamily::EnvelopedHarmonicBank => "envelope(harmonic_bank)",
            CompoundFamily::ToneDecomposition => "tone_decomposition",
            CompoundFamily::EnvelopedToneDecomposition => "envelope(tone_decomposition)",
            CompoundFamily::EchoicBank => "add(bank,delay(bank))",
        }
    }
}

/// One discovered candidate graph.
#[derive(Debug, Clone)]
pub struct CompoundProposal {
    pub family: CompoundFamily,
    pub label: String,
    pub graph: CompoundGraph,
}

/// Discover bounded deterministic composition candidates from an observed mono
/// window. `samples` is the canonical interleaved code sequence; `frames` is the
/// frame extent and `sample_rate_hz` the rate. Only channel 0 is analysed
/// (Compound is mono-first in this build) and the result is empty when the input
/// is multichannel or degenerate.
pub fn propose(
    samples: &[i32],
    channels: u8,
    sample_rate_hz: u32,
    frames: u64,
    budget: &CompoundBudget,
) -> Result<Vec<CompoundProposal>> {
    budget.validate()?;
    if channels != 1 {
        return Ok(Vec::new());
    }
    if frames == 0
        || sample_rate_hz == 0
        || frames > crate::limits::MAX_OBJECT_FRAMES
        || samples.len() != frames as usize
    {
        return Ok(Vec::new());
    }

    let mut out: Vec<CompoundProposal> = Vec::new();
    let x: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
    let n = frames as usize;

    // 1. Silence.
    if x.iter().all(|&v| v == 0.0) {
        out.push(CompoundProposal {
            family: CompoundFamily::Silence,
            label: "silence".to_string(),
            graph: graph_of(
                frames,
                sample_rate_hz,
                vec![node(CompoundOp::Silence, vec![])],
            ),
        });
        return Ok(out);
    }

    // 2. Constant (exact single level).
    let first = samples[0];
    let mut constant = true;
    for &s in samples.iter() {
        if s != first {
            constant = false;
            break;
        }
    }
    if constant {
        out.push(CompoundProposal {
            family: CompoundFamily::Constant,
            label: format!("constant(level={first})"),
            graph: graph_of(
                frames,
                sample_rate_hz,
                vec![node(CompoundOp::Constant { level: first }, vec![])],
            ),
        });
        return Ok(out);
    }

    // 3. Tonal analysis: fundamental, harmonics, envelope, echo.
    let fs = f64::from(sample_rate_hz);
    let rms = band_rms(&x, budget.envelope_bands);
    let peak_band = argmax(&rms);
    let tonal_slice = analysis_slice(n, peak_band, rms.len(), budget.analysis_frames);
    let window = &x[tonal_slice.clone()];

    let (f0, strength) = match estimate_fundamental(
        window,
        fs,
        f64::from(budget.min_freq_hz),
        f64::from(budget.max_freq_hz),
    ) {
        Some(v) => v,
        None => return Ok(out),
    };
    // Snap to the integer frequency the emitted oscillator will actually use, so
    // the analytic projection and the deterministic graph share one frequency
    // and the window-offset phase removal below is exact. The window estimate is
    // then refined over the full signal: a 1 Hz error is negligible over the
    // analysis window but dephases completely over a long extent.
    let f0 = refine_frequency(
        &x,
        fs,
        f0,
        f64::from(budget.min_freq_hz),
        f64::from(budget.max_freq_hz),
        sample_rate_hz,
    );

    let max_harmonic =
        ((f64::from(budget.max_freq_hz) / f0).floor() as usize).min(budget.max_harmonics);
    let harmonics = harmonic_amplitudes(window, fs, f0, max_harmonic.max(1), tonal_slice.start);

    let env = fit_envelope(&rms, n, rms.len(), frames);

    // Absolute-scale single oscillator (no gain node) when the amplitude is
    // large enough for `amp_q16` to resolve it.
    if let Some(g) = single_oscillator_direct(&harmonics, f0, frames, sample_rate_hz) {
        out.push(CompoundProposal {
            family: CompoundFamily::Oscillator,
            label: format!("oscillator(f0={f0:.2},strength={strength:.3})"),
            graph: g,
        });
    }

    // Probe-scale single oscillator (explicit Q16 gain) — the general case.
    if let Some(g) = tonal_graph(
        &harmonics[..1.min(harmonics.len())],
        f0,
        frames,
        sample_rate_hz,
    ) {
        out.push(CompoundProposal {
            family: CompoundFamily::GainOscillator,
            label: format!("gain(oscillator,f0={f0:.2},strength={strength:.3})"),
            graph: g,
        });
    }

    // Harmonic bank (probe scale + explicit gain).
    let bank = tonal_graph(&harmonics, f0, frames, sample_rate_hz);
    if let Some(g) = bank.clone() {
        let count = harmonics.iter().filter(|(a, _)| *a >= 64.0).count();
        out.push(CompoundProposal {
            family: CompoundFamily::HarmonicBank,
            label: format!("harmonic_bank(k={count})"),
            graph: g,
        });
    }

    // Enveloped variants: apply the fitted ADSR to the single tone and the bank.
    if let Some(e) = &env {
        if let Some(g) = tonal_graph(
            &harmonics[..1.min(harmonics.len())],
            f0,
            frames,
            sample_rate_hz,
        ) && let Some(eg) = with_envelope(g, e)
        {
            out.push(CompoundProposal {
                family: CompoundFamily::EnvelopedOscillator,
                label: format!(
                    "envelope(oscillator,a={},d={},s={},r={})",
                    e.0, e.1, e.2, e.3
                ),
                graph: eg,
            });
        }
        if let Some(g) = bank.clone()
            && let Some(eg) = with_envelope(g, e)
        {
            out.push(CompoundProposal {
                family: CompoundFamily::EnvelopedHarmonicBank,
                label: format!(
                    "envelope(harmonic_bank,a={},d={},s={},r={})",
                    e.0, e.1, e.2, e.3
                ),
                graph: eg,
            });
        }
    }

    // Echoic variant: a delayed copy of the bank, when a secondary
    // autocorrelation peak exists.
    if let Some(g) = bank.as_ref()
        && let Some(delay) = estimate_echo_delay(window, fs, f0)
        && let Some(eg) = with_echo(g.clone(), delay)
    {
        out.push(CompoundProposal {
            family: CompoundFamily::EchoicBank,
            label: format!("add(bank,delay(bank,{delay}))"),
            graph: eg,
        });
    }

    // General tonal decomposition: matching pursuit over independent tones.
    // Unlike the harmonic bank this makes no integer-harmonic assumption, so it
    // expresses polyphony, detuned pairs and non-harmonic mixtures.
    let tones = decompose_tones(
        window,
        fs,
        f64::from(budget.min_freq_hz),
        f64::from(budget.max_freq_hz),
        budget.max_harmonics.min(MAX_TONES),
    );
    let tone_graph_candidate = tone_graph(&tones, tonal_slice.start, fs, frames, sample_rate_hz);
    if let Some(g) = tone_graph_candidate.clone() {
        out.push(CompoundProposal {
            family: CompoundFamily::ToneDecomposition,
            label: format!("tone_decomposition(k={})", tones.len()),
            graph: g,
        });
    }
    if let Some(e) = &env
        && let Some(g) = tone_graph_candidate
        && let Some(eg) = with_envelope(g, e)
    {
        out.push(CompoundProposal {
            family: CompoundFamily::EnvelopedToneDecomposition,
            label: format!(
                "envelope(tone_decomposition,k={},a={},d={},s={},r={})",
                tones.len(),
                e.0,
                e.1,
                e.2,
                e.3
            ),
            graph: eg,
        });
    }

    let _ = window;
    if out.len() > budget.max_candidates {
        out.truncate(budget.max_candidates);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------------

fn graph_of(frames: u64, rate: u32, nodes: Vec<CompoundNode>) -> CompoundGraph {
    CompoundGraph {
        channels: 1,
        frames,
        sample_rate_hz: rate,
        nodes,
    }
}

fn node(op: CompoundOp, children: Vec<u16>) -> CompoundNode {
    CompoundNode { op, children }
}

/// Absolute-scale `amp_q16` (`code / 2^15`), `0` when unresolvable.
/// Convert a radian phase to the frozen 2^64 phase domain (one cycle = 2^64).
fn phase0_from_angle(theta: f64) -> u64 {
    let frac = (theta / (2.0 * std::f64::consts::PI)).rem_euclid(1.0);
    (frac * (u64::MAX as f64)) as u64
}

/// Absolute-scale `amp_q16` (`code / 2^15`), `0` when unresolvable.
fn amp_q16_direct(a_code: f64) -> i32 {
    let q = (a_code.abs() / 32_768.0).round();
    if !q.is_finite() || q < 1.0 {
        return 0;
    }
    q.min(65_535.0) as i32
}

/// A single absolute-scale oscillator (no gain node), when the fundamental is
/// strong enough for `amp_q16` to resolve it.
fn single_oscillator_direct(
    harmonics: &[(f64, f64)],
    f0: f64,
    frames: u64,
    rate: u32,
) -> Option<CompoundGraph> {
    let (amp, phase) = *harmonics.first()?;
    let q = amp_q16_direct(amp);
    if q == 0 {
        return None;
    }
    let freq = f0.round().clamp(1.0, f64::from(rate / 2)) as u32;
    let osc = node(
        CompoundOp::Oscillator {
            freq_hz: freq,
            amp_q16: q,
            phase0: phase0_from_angle(phase),
        },
        vec![],
    );
    Some(graph_of(frames, rate, vec![osc]))
}

/// Append one oscillator scaled to `amp` code units. The oscillator runs at
/// maximum `amp_q16` resolution and a per-tone `Gain` applies the amplitude, so
/// the tones can be summed without the intermediate saturating (`Add` then a
/// single shared `Gain` would clip a loud mixture to `i32::MAX`).
fn push_tone(
    nodes: &mut Vec<CompoundNode>,
    children: &mut Vec<u16>,
    amp: f64,
    freq: u32,
    phase0: u64,
) {
    if !amp.is_finite() || amp.abs() < 0.5 {
        return;
    }
    let (q, gain) = tone_gain_amp(amp);
    let osc = nodes.len() as u16;
    nodes.push(node(
        CompoundOp::Oscillator {
            freq_hz: freq,
            amp_q16: q,
            phase0,
        },
        vec![],
    ));
    // Always apply the per-tone gain: `Gain` is a Q16 multiply, so even a gain of
    // 1 is a 1/65536 scale, not unity.
    let g = nodes.len() as u16;
    nodes.push(node(CompoundOp::Gain { q16: gain }, vec![osc]));
    children.push(g);
}

/// Split an amplitude into an oscillator `amp_q16` and an optional `Gain` so the
/// rendered peak is as close to `amp` as the integer representation allows.
/// `amp_q16` alone resolves `0.5` code units up to `32767`; louder tones use a
/// power-of-two gain (step `gain/2`) while keeping full `amp_q16` resolution.
fn tone_gain_amp(amp: f64) -> (i32, i32) {
    let amp = amp.abs();
    if amp <= 32_767.5 {
        let q = (2.0 * amp).round().clamp(1.0, 65_535.0) as i32;
        return (q, 1);
    }
    let mut gain = 1.0f64;
    while 65_535.0 * gain / 2.0 < amp && gain < (1 << 24) as f64 {
        gain *= 2.0;
    }
    let q = (2.0 * amp / gain).round().clamp(1.0, 65_535.0) as i32;
    (q, gain as i32)
}

/// Adaptive-resolution tonal graph: one oscillator for a pure tone, an `Add`
/// bank for a multi-harmonic tone, each term scaled to its own amplitude before
/// the sum.
fn tonal_graph(harmonics: &[(f64, f64)], f0: f64, frames: u64, rate: u32) -> Option<CompoundGraph> {
    let mut nodes: Vec<CompoundNode> = Vec::new();
    let mut children: Vec<u16> = Vec::new();
    for (k, &(amp, phase)) in harmonics.iter().enumerate() {
        let freq = (f0 * (k + 1) as f64).round();
        if freq < 1.0 || freq > f64::from(rate / 2) {
            continue;
        }
        push_tone(
            &mut nodes,
            &mut children,
            amp,
            freq as u32,
            phase0_from_angle(phase),
        );
        if children.len() >= MAX_COMPOUND_ARITY as usize {
            break;
        }
    }
    if children.is_empty() {
        return None;
    }
    if children.len() > 1 {
        nodes.push(node(CompoundOp::Add, children));
    }
    Some(graph_of(frames, rate, nodes))
}

/// Envelope parameters: `(attack, decay, sustain_q16, release, t_off)`.
type FittedEnvelope = (u32, u32, i32, u32, Option<i64>);

/// Multiply the root by the frozen analytic ADSR.
fn with_envelope(mut g: CompoundGraph, env: &FittedEnvelope) -> Option<CompoundGraph> {
    let (a, d, s, r, t_off) = *env;
    crate::sampler::envelope::EnvelopeParams::new(a, d, s, r)?;
    let root = (g.nodes.len() - 1) as u16;
    g.nodes.push(node(
        CompoundOp::Envelope {
            attack_frames: a,
            decay_frames: d,
            sustain_q16: s,
            release_frames: r,
            t_on: 0,
            t_off,
        },
        vec![root],
    ));
    Some(g)
}

/// Add a delayed copy of the root.
fn with_echo(mut g: CompoundGraph, delay: u32) -> Option<CompoundGraph> {
    if delay == 0 {
        return None;
    }
    let root = (g.nodes.len() - 1) as u16;
    g.nodes.push(node(
        CompoundOp::Delay {
            frames: i64::from(delay),
        },
        vec![root],
    ));
    let delayed = (g.nodes.len() - 1) as u16;
    g.nodes.push(node(CompoundOp::Add, vec![root, delayed]));
    Some(g)
}

// ---------------------------------------------------------------------------
// Deterministic float analysis (proposal-only, zero authority)
// ---------------------------------------------------------------------------

/// Normalised autocorrelation fundamental estimate over `[min_hz, max_hz]`.
/// Returns `(f0_hz, strength)` with strength in `[0, 1]`.
///
/// A low-frequency signal's autocorrelation decreases monotonically from lag 0,
/// so the *global* maximum over the lag window sits at the smallest lag. The
/// estimator therefore follows the standard pitch-detection rule: find the
/// first local minimum at or after `lag_min`, then the next local maximum — that
/// is the fundamental period. Only that peak is refined.
fn estimate_fundamental(x: &[f64], fs: f64, min_hz: f64, max_hz: f64) -> Option<(f64, f64)> {
    if x.len() < 8 {
        return None;
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let y: Vec<f64> = x.iter().map(|&v| v - mean).collect();
    let lag_min = (fs / max_hz).floor().max(1.0) as usize;
    let lag_max = ((fs / min_hz).ceil() as usize).min(y.len().saturating_sub(2));
    if lag_max <= lag_min {
        return None;
    }
    let mut r = vec![0.0f64; lag_max + 2];
    for lag in 1..=lag_max + 1 {
        let count = y.len().saturating_sub(lag);
        if count == 0 {
            continue;
        }
        let mut num = 0.0f64;
        let mut den_a = 0.0f64;
        let mut den_b = 0.0f64;
        for t in 0..count {
            num += y[t] * y[t + lag];
            den_a += y[t] * y[t];
            den_b += y[t + lag] * y[t + lag];
        }
        let den = (den_a * den_b).sqrt();
        r[lag] = if den > f64::EPSILON { num / den } else { 0.0 };
    }
    // First local minimum at or after lag_min, then the next local maximum.
    let mut i = lag_min.max(1);
    while i < lag_max + 1 && !(r[i] <= r[i - 1] && r[i] <= r[i + 1]) {
        i += 1;
    }
    let mut best_lag = 0usize;
    while i < lag_max + 1 {
        if r[i] >= r[i - 1] && r[i] > r[i + 1] {
            best_lag = i;
            break;
        }
        i += 1;
    }
    if best_lag == 0 {
        // Fallback: strongest local maximum, else the global maximum.
        let mut best = f64::NEG_INFINITY;
        for lag in lag_min..=lag_max {
            if r[lag] >= r[lag - 1] && r[lag] >= r[lag + 1] && r[lag] > best {
                best = r[lag];
                best_lag = lag;
            }
        }
        if best_lag == 0 {
            best_lag = (lag_min..=lag_max)
                .max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap_or(std::cmp::Ordering::Equal))?;
        }
    }
    let refined = refine_peak(&y, best_lag);
    let f0 = fs / refined;
    Some((f0, r[best_lag].clamp(0.0, 1.0)))
}

fn refine_peak(y: &[f64], lag: usize) -> f64 {
    if lag == 0 || lag + 1 >= y.len() {
        return lag as f64;
    }
    let corr = |l: usize| -> f64 {
        let count = y.len().saturating_sub(l);
        if count == 0 {
            return 0.0;
        }
        let mut num = 0.0;
        for t in 0..count {
            num += y[t] * y[t + l];
        }
        num
    };
    let a = corr(lag - 1);
    let b = corr(lag);
    let c = corr(lag + 1);
    let denom = a - 2.0 * b + c;
    if denom.abs() < f64::EPSILON {
        return lag as f64;
    }
    let delta = 0.5 * (a - c) / denom;
    (lag as f64 + delta.clamp(-0.5, 0.5)).max(1.0)
}

/// Projection amplitude of a pure sinusoid at `f` over a bounded prefix of `x`.
fn project_amplitude(x: &[f64], fs: f64, f: f64) -> f64 {
    let cap = x.len().min(8_192);
    let y = &x[..cap];
    if y.is_empty() || f <= 0.0 || f >= fs / 2.0 {
        return 0.0;
    }
    let w = 2.0 * std::f64::consts::PI * f / fs;
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for (t, &v) in y.iter().enumerate() {
        let a = w * t as f64;
        s += v * a.sin();
        c += v * a.cos();
    }
    2.0 * (s * s + c * c).sqrt() / y.len() as f64
}

/// Refine a rounded frequency estimate by a bounded integer search; the best
/// single-sinusoid fit over the full signal is the one maximising projection
/// amplitude (equivalently minimising residual energy).
fn refine_frequency(x: &[f64], fs: f64, f0: f64, min_hz: f64, max_hz: f64, rate: u32) -> f64 {
    let nyquist = f64::from(rate / 2).max(1.0);
    let base = f0.round().clamp(1.0, nyquist);
    let mut best = base;
    let mut best_amp = project_amplitude(x, fs, base);
    let lo = (base - 2.0).max(min_hz.max(1.0));
    let hi = (base + 2.0).min(max_hz).min(nyquist);
    let mut cand = lo.floor();
    while cand <= hi.ceil() {
        let amp = project_amplitude(x, fs, cand);
        if amp > best_amp {
            best_amp = amp;
            best = cand;
        }
        cand += 1.0;
    }
    best
}

/// One complex-spectrum peak / matched tone in the analysis window's local time
/// base: `x[t] ≈ Σ amp·sin(2π·freq·t/fs + phase)`.
#[derive(Debug, Clone, Copy)]
struct Tone {
    freq: f64,
    amp: f64,
    phase: f64,
}

/// In-place iterative radix-2 Cooley–Tukey FFT (analysis-only, zero authority).
/// `re`/`im` must have a power-of-two length.
fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0usize;
        while i < n {
            let mut cur_r = 1.0f64;
            let mut cur_i = 0.0f64;
            for k in 0..half {
                let ur = re[i + k];
                let ui = im[i + k];
                let xr = re[i + k + half];
                let xi = im[i + k + half];
                let vr = xr * cur_r - xi * cur_i;
                let vi = xr * cur_i + xi * cur_r;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + half] = ur - vr;
                im[i + k + half] = ui - vi;
                let nwr = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = nwr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Hann-windowed, twice-zero-padded magnitude spectrum of `x` (bins `0..n/2`).
/// Returns `(magnitudes, n_fft, bin_hz_denominator_n)`.
fn spectrum(x: &[f64]) -> (Vec<f64>, usize) {
    let m = x.len();
    if m < 4 {
        return (Vec::new(), 0);
    }
    let base = m.next_power_of_two() * 2;
    let n = base.min(1 << 15);
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    let denom = (m - 1) as f64;
    for (i, &v) in x.iter().enumerate().take(n) {
        let win = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos();
        re[i] = v * win;
    }
    fft_in_place(&mut re, &mut im);
    let mut mags = Vec::with_capacity(n / 2);
    for k in 0..n / 2 {
        mags.push((re[k] * re[k] + im[k] * im[k]).sqrt());
    }
    (mags, n)
}

/// Projection `(amplitude, phase)` of `x` onto a sinusoid at `f`, over the local
/// time base `t = 0..x.len()`.
fn project_tone(x: &[f64], fs: f64, f: f64) -> (f64, f64) {
    let n = x.len();
    if n == 0 || f <= 0.0 || f >= fs / 2.0 {
        return (0.0, 0.0);
    }
    let w = 2.0 * std::f64::consts::PI * f / fs;
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for (t, &v) in x.iter().enumerate() {
        let a = w * t as f64;
        s += v * a.sin();
        c += v * a.cos();
    }
    let amp = 2.0 * (s * s + c * c).sqrt() / n as f64;
    (amp, c.atan2(s))
}

/// Golden-section maximiser over `[a, b]` (analysis-only).
fn golden_max<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64) -> f64 {
    const R: f64 = 0.618_033_988_749_894_9;
    if b <= a {
        return a;
    }
    let mut c = b - R * (b - a);
    let mut d = a + R * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..30 {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - R * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + R * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// Best single tone in `res`: FFT magnitude peaks seed a bounded local
/// refinement that maximises the projection amplitude.
fn best_frequency(res: &[f64], fs: f64, min_hz: f64, max_hz: f64) -> Option<Tone> {
    if res.len() < 16 || max_hz <= min_hz {
        return None;
    }
    let (mags, n) = spectrum(res);
    if mags.is_empty() {
        return None;
    }
    let bin_hz = fs / n as f64;
    let lo = ((min_hz / bin_hz).ceil() as usize).max(1);
    let hi = ((max_hz / bin_hz).floor() as usize).min(mags.len().saturating_sub(2));
    if hi <= lo {
        return None;
    }
    let mut peaks: Vec<(f64, usize)> = Vec::new();
    for k in lo..=hi {
        if mags[k] >= mags[k - 1] && mags[k] >= mags[k + 1] {
            peaks.push((mags[k], k));
        }
    }
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(6);
    let mut best: Option<Tone> = None;
    for (_, k) in peaks {
        let center = k as f64 * bin_hz;
        let a = (center - 1.5 * bin_hz).max(min_hz);
        let b = (center + 1.5 * bin_hz).min(max_hz);
        let f = golden_max(|f| project_tone(res, fs, f).0, a, b);
        let (amp, phase) = project_tone(res, fs, f);
        if best.as_ref().is_none_or(|t| amp > t.amp) {
            best = Some(Tone {
                freq: f,
                amp,
                phase,
            });
        }
    }
    best
}

/// Solve the joint least-squares fit of `x` onto the real basis
/// `{sin(w_j t), cos(w_j t)}` for the given frequencies, returning coefficients
/// `[a_0, b_0, a_1, b_1, ...]` with `x[t] ≈ Σ a_j sin(w_j t) + b_j cos(w_j t)`.
///
/// The Gram matrix is solved with a tiny ridge term so nearly collinear
/// frequencies (e.g. 180 Hz and 181 Hz inside one window) resolve to the
/// minimum-energy split instead of one atom absorbing the other.
fn solve_basis(x: &[f64], fs: f64, freqs: &[f64]) -> Vec<f64> {
    let k = freqs.len();
    let m = 2 * k;
    let n = x.len();
    if k == 0 {
        return Vec::new();
    }
    let w: Vec<f64> = freqs
        .iter()
        .map(|&f| 2.0 * std::f64::consts::PI * f / fs)
        .collect();
    let mut g = vec![0.0f64; m * m];
    let mut d = vec![0.0f64; m];
    let mut s = vec![0.0f64; k];
    let mut c = vec![0.0f64; k];
    for (t, &v) in x.iter().enumerate() {
        let tt = t as f64;
        for j in 0..k {
            let a = w[j] * tt;
            s[j] = a.sin();
            c[j] = a.cos();
        }
        for j in 0..k {
            d[2 * j] += v * s[j];
            d[2 * j + 1] += v * c[j];
            for l in 0..=j {
                let gss = g[2 * j * m + 2 * l] + s[j] * s[l];
                g[2 * j * m + 2 * l] = gss;
                let gsc = g[2 * j * m + 2 * l + 1] + s[j] * c[l];
                g[2 * j * m + 2 * l + 1] = gsc;
                let gcs = g[(2 * j + 1) * m + 2 * l] + c[j] * s[l];
                g[(2 * j + 1) * m + 2 * l] = gcs;
                let gcc = g[(2 * j + 1) * m + 2 * l + 1] + c[j] * c[l];
                g[(2 * j + 1) * m + 2 * l + 1] = gcc;
            }
        }
    }
    // Mirror the lower triangle.
    for j in 0..m {
        for l in 0..j {
            let v = g[j * m + l];
            g[l * m + j] = v;
        }
    }
    let ridge = 1e-6 * (n as f64 / 2.0).max(1.0);
    for j in 0..m {
        g[j * m + j] += ridge;
    }
    solve_linear(g, d, m)
}

/// Gaussian elimination with partial pivoting and a zero-size guard.
fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, m: usize) -> Vec<f64> {
    for col in 0..m {
        let mut piv = col;
        let mut best = a[col * m + col].abs();
        for r in col + 1..m {
            let v = a[r * m + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            continue;
        }
        if piv != col {
            for c in 0..m {
                a.swap(col * m + c, piv * m + c);
            }
            b.swap(col, piv);
        }
        let d = a[col * m + col];
        for r in col + 1..m {
            let f = a[r * m + col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..m {
                a[r * m + c] -= f * a[col * m + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0f64; m];
    for col in (0..m).rev() {
        let d = a[col * m + col];
        if d.abs() < 1e-15 {
            x[col] = 0.0;
            continue;
        }
        let mut s = b[col];
        for c in col + 1..m {
            s -= a[col * m + c] * x[c];
        }
        x[col] = s / d;
    }
    x
}

/// Synthesize the model `Σ a_j sin + b_j cos` from basis coefficients.
fn synthesize(coeffs: &[f64], fs: f64, freqs: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    for (j, &f) in freqs.iter().enumerate() {
        let a = coeffs[2 * j];
        let b = coeffs[2 * j + 1];
        let w = 2.0 * std::f64::consts::PI * f / fs;
        for (t, o) in out.iter_mut().enumerate() {
            let ang = w * t as f64;
            *o += a * ang.sin() + b * ang.cos();
        }
    }
    out
}

fn residual_energy(x: &[f64], model: &[f64]) -> f64 {
    x.iter()
        .zip(model.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum()
}

/// Decompose the analysis window into a bounded set of tones by orthogonal
/// matching pursuit **over the integer atom basis the graph can emit**, with a
/// joint least-squares re-solve at every step.
///
/// Working directly in the integer-frequency basis is essential: two tones one
/// hertz apart are nearly collinear inside a short window, so a continuous
/// optimum would merge them into one fractional frequency that the integer
/// oscillator cannot represent. Selecting integer atoms and re-solving together
/// lets the decomposition express the detuned pair exactly.
fn decompose_tones(
    window: &[f64],
    fs: f64,
    min_hz: f64,
    max_hz: f64,
    max_tones: usize,
) -> Vec<Tone> {
    if window.len() < 16 || max_tones == 0 {
        return Vec::new();
    }
    // Fit the raw window: over a finite window a sine/cosine pair naturally
    // carries the segment mean, so subtracting it first would corrupt the fit.
    let x: Vec<f64> = window.to_vec();
    let n = x.len();
    let initial_energy: f64 = x.iter().map(|v| v * v).sum();
    if initial_energy <= f64::EPSILON {
        return Vec::new();
    }
    let lo = min_hz.max(1.0);
    let hi = max_hz.min(fs / 2.0 - 1.0);

    let mut freqs: Vec<f64> = Vec::with_capacity(max_tones);
    let mut model = vec![0.0f64; n];
    let mut energy = initial_energy;
    let mut first_amp = 0.0f64;
    for _ in 0..max_tones {
        let r: Vec<f64> = x.iter().zip(model.iter()).map(|(&a, &b)| a - b).collect();
        let Some(t) = best_frequency(&r, fs, lo, hi) else {
            break;
        };
        if freqs.is_empty() {
            first_amp = t.amp;
        }
        // Choose the best unused integer frequency near the continuous estimate.
        let base = t.freq.round();
        let mut cand = None;
        let mut cand_amp = 0.0f64;
        for d in -2i32..=2 {
            let f = (base + f64::from(d)).clamp(lo, hi);
            if freqs.iter().any(|&u| (u - f).abs() < 0.5) {
                continue;
            }
            let (amp, _) = project_tone(&r, fs, f);
            if amp > cand_amp {
                cand_amp = amp;
                cand = Some(f);
            }
        }
        let Some(f) = cand else {
            break;
        };
        if cand_amp < (first_amp * 0.02).max(1.0) {
            break;
        }
        freqs.push(f);
        let trial_coeffs = solve_basis(&x, fs, &freqs);
        let trial_model = synthesize(&trial_coeffs, fs, &freqs, n);
        let next = residual_energy(&x, &trial_model);
        if next > energy * (1.0 - 1e-5) {
            freqs.pop();
            break;
        }
        model = trial_model;
        energy = next;
    }

    // Split refinement: greedy correlation cannot see a partial that is nearly
    // collinear with an already-selected atom (e.g. 180 Hz vs 181 Hz inside a
    // short window). Test adding each selected frequency's immediate integer
    // neighbours and keep a split whenever it genuinely reduces the joint LS
    // residual energy. Bounded by `max_tones` and by the number of selected
    // tones.
    let mut improved = true;
    while improved && freqs.len() < max_tones {
        improved = false;
        let mut best: Option<(Vec<f64>, f64)> = None;
        for k in 0..freqs.len() {
            for d in [-1.0f64, 1.0] {
                let f = freqs[k] + d;
                if f < lo || f > hi || freqs.iter().any(|&u| (u - f).abs() < 0.5) {
                    continue;
                }
                let mut trial = freqs.clone();
                trial.push(f);
                let c = solve_basis(&x, fs, &trial);
                let m = synthesize(&c, fs, &trial, n);
                let e = residual_energy(&x, &m);
                if e < energy * (1.0 - 1e-5) && best.as_ref().is_none_or(|(_, be)| e < *be) {
                    best = Some((trial, e));
                }
            }
        }
        if let Some((trial, e)) = best {
            freqs = trial;
            energy = e;
            improved = true;
        }
    }
    if freqs.is_empty() {
        return Vec::new();
    }
    let coeffs = solve_basis(&x, fs, &freqs);

    let floor = (first_amp * 0.01).max(0.5);
    let mut tones = Vec::with_capacity(freqs.len());
    for (j, &f) in freqs.iter().enumerate() {
        let a = coeffs[2 * j];
        let b = coeffs[2 * j + 1];
        let amp = (a * a + b * b).sqrt();
        if amp < floor {
            continue;
        }
        tones.push(Tone {
            freq: f,
            amp,
            phase: b.atan2(a),
        });
    }
    tones
}

/// Build an additive graph of independent oscillators from a tone list, each
/// scaled to its own amplitude before the sum. Absolute phase is recovered from
/// the window offset.
fn tone_graph(
    tones: &[Tone],
    offset: usize,
    fs: f64,
    frames: u64,
    rate: u32,
) -> Option<CompoundGraph> {
    let mut nodes: Vec<CompoundNode> = Vec::new();
    let mut children: Vec<u16> = Vec::new();
    for t in tones {
        let freq = t.freq.round().clamp(1.0, f64::from(rate / 2)) as u32;
        // The tone is fit in the window's local time base; convert to the
        // absolute phase the graph's oscillator uses from frame 0.
        let w = 2.0 * std::f64::consts::PI * t.freq / fs;
        let absolute = t.phase - w * offset as f64;
        push_tone(
            &mut nodes,
            &mut children,
            t.amp,
            freq,
            phase0_from_angle(absolute),
        );
        if children.len() >= MAX_COMPOUND_ARITY as usize {
            break;
        }
    }
    if children.is_empty() {
        return None;
    }
    if children.len() > 1 {
        nodes.push(node(CompoundOp::Add, children));
    }
    Some(graph_of(frames, rate, nodes))
}

/// Project the window onto `sin`/`cos` at each harmonic of `f0`; returns
/// `(amplitude, absolute_phase_radians)` per harmonic, starting at the
/// fundamental. `offset` is the absolute frame index of `x[0]`, so the returned
/// phase is valid for an oscillator whose phase is defined from frame 0.
fn harmonic_amplitudes(
    x: &[f64],
    fs: f64,
    f0: f64,
    max_k: usize,
    offset: usize,
) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(max_k);
    let n = x.len() as f64;
    if n == 0.0 {
        return out;
    }
    let mut peak = 0.0f64;
    let mut raw: Vec<(f64, f64)> = Vec::with_capacity(max_k);
    for k in 1..=max_k {
        let f = f0 * k as f64;
        if f >= fs / 2.0 {
            raw.push((0.0, 0.0));
            continue;
        }
        let w = 2.0 * std::f64::consts::PI * f / fs;
        let mut s = 0.0f64;
        let mut c = 0.0f64;
        for (t, &v) in x.iter().enumerate() {
            let a = w * t as f64;
            s += v * a.sin();
            c += v * a.cos();
        }
        s *= 2.0 / n;
        c *= 2.0 / n;
        let amp = (s * s + c * c).sqrt();
        // `x = A·sin(wt + φ) = A·cosφ·sin(wt) + A·sinφ·cos(wt)`, so the
        // `sin(wt)` projection is `A·cosφ` and the `cos(wt)` projection is
        // `A·sinφ`; hence `φ = atan2(c, s)`. The window starts at absolute frame
        // `offset`, so remove the phase accumulated before it to make `phase0`
        // absolute.
        let phase = c.atan2(s) - w * offset as f64;
        peak = peak.max(amp);
        raw.push((amp, phase));
    }
    if peak <= f64::EPSILON {
        return raw;
    }
    // Keep harmonics above a fixed fraction of the strongest, preserving index.
    let threshold = 0.02 * peak;
    for &(amp, phase) in &raw {
        if amp >= threshold {
            out.push((amp, phase));
        } else {
            out.push((0.0, 0.0));
        }
    }
    out
}

/// Fixed-width energy bands over the whole extent.
fn band_rms(x: &[f64], bands: usize) -> Vec<f64> {
    let n = x.len();
    let bands = bands.max(1).min(n.max(1));
    let mut out = Vec::with_capacity(bands);
    for b in 0..bands {
        let start = b * n / bands;
        let end = ((b + 1) * n / bands).max(start + 1).min(n);
        let mut acc = 0.0f64;
        for &v in &x[start..end] {
            acc += v * v;
        }
        out.push((acc / (end - start) as f64).sqrt());
    }
    out
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// A bounded analysis window around the peak-energy band.
fn analysis_slice(n: usize, peak_band: usize, bands: usize, cap: usize) -> std::ops::Range<usize> {
    let len = cap.min(n);
    let center = peak_band
        .checked_mul(n)
        .and_then(|x| x.checked_div(bands))
        .unwrap_or(0);
    let start = center.saturating_sub(len / 2).min(n - len);
    start..start + len
}

/// Fit a bounded ADSR to the band-energy contour.
fn fit_envelope(rms: &[f64], n: usize, bands: usize, frames: u64) -> Option<FittedEnvelope> {
    if bands < 4 || n == 0 {
        return None;
    }
    let peak = rms.iter().cloned().fold(0.0f64, f64::max);
    if peak <= f64::EPSILON {
        return None;
    }
    let floor = 0.02 * peak;
    let peak_band = argmax(rms);
    // Peak must not be at the very first or last band for an envelope to be
    // meaningful (otherwise there is nothing to shape).
    let starts_low = rms[0] < 0.5 * peak;
    let ends_low = rms[bands - 1] < 0.5 * peak;
    if !starts_low && !ends_low && peak_band != 0 {
        // Flat contour: an envelope would only add bytes.
        let mid = &rms[bands / 4..(3 * bands / 4).max(bands / 4 + 1)];
        let mid_max = mid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mid_min = mid.iter().cloned().fold(f64::INFINITY, f64::min);
        if mid_max <= f64::EPSILON || mid_min > 0.6 * mid_max {
            return None;
        }
    }
    let band_frames = (frames as usize / bands).max(1) as u32;
    let t_on = 0i64;
    let attack_frames = (peak_band as u32).saturating_mul(band_frames);
    // Sustain = level of the region after the peak, before any final drop.
    let mut off_band = bands - 1;
    while off_band > peak_band && rms[off_band] < 0.25 * peak {
        off_band -= 1;
    }
    let sustain_region = &rms[peak_band..=off_band];
    let sustain_level = if sustain_region.is_empty() {
        peak
    } else {
        let mut vals = sustain_region.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals[vals.len() / 2]
    };
    let sustain_q16 = ((sustain_level / peak) * 65_536.0).round() as i64;
    let sustain_q16 = sustain_q16.clamp(0, 65_536) as i32;
    let decay_frames = ((off_band.saturating_sub(peak_band)) as u32).saturating_mul(band_frames);
    let t_off = if off_band < bands - 1 {
        Some((off_band as u32).saturating_mul(band_frames) as i64 + i64::from(band_frames))
    } else {
        None
    };
    let release_frames = if let Some(off) = t_off {
        (frames.saturating_sub(off.max(0) as u64) as u32).min(band_frames.saturating_mul(4))
    } else {
        0
    };
    if attack_frames == 0 && decay_frames == 0 && release_frames == 0 && sustain_q16 >= 65_536 {
        return None;
    }
    if !starts_low && !ends_low && release_frames == 0 && sustain_q16 >= 65_536 {
        return None;
    }
    let _ = (floor, t_on);
    Some((
        attack_frames.max(1),
        decay_frames,
        sustain_q16,
        release_frames,
        t_off,
    ))
}

/// Secondary autocorrelation peak → a plausible echo delay. Bounded to a
/// `min(fs/8, 4096)`-sample lag range and a 2048-sample inner correlation so the
/// search stays cheap regardless of the case extent.
fn estimate_echo_delay(x: &[f64], fs: f64, f0: f64) -> Option<u32> {
    let period = fs / f0;
    let min_lag = (2.0 * period).ceil() as usize;
    let max_lag = ((fs * 0.125) as usize).min(4_096);
    if x.len() < min_lag + 4 || max_lag <= min_lag {
        return None;
    }
    let inner = x.len().min(2_048);
    let y = &x[..inner];
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let y: Vec<f64> = y.iter().map(|&v| v - mean).collect();
    let energy: f64 = y.iter().map(|v| v * v).sum();
    if energy <= f64::EPSILON {
        return None;
    }
    let mut best_lag = 0usize;
    let mut best = 0.1f64; // require a real peak
    for lag in min_lag..max_lag.min(y.len().saturating_sub(2)) {
        let count = y.len() - lag;
        let mut num = 0.0;
        for t in 0..count {
            num += y[t] * y[t + lag];
        }
        let r = num / energy;
        if r > best {
            best = r;
            best_lag = lag;
        }
    }
    if best_lag == 0 {
        return None;
    }
    // A lag at a whole multiple of the fundamental period is just the same tone
    // delayed, not an echo; proposing it would only double the signal.
    let phase = best_lag as f64 % period;
    if phase < 0.15 * period || phase > 0.85 * period {
        return None;
    }
    Some(best_lag as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_harmonic(f0: f64, rate: u32, frames: usize, amps: &[f64]) -> Vec<i32> {
        let mut out = vec![0i32; frames];
        for (k, &a) in amps.iter().enumerate() {
            let f = f0 * (k + 1) as f64;
            for (t, s) in out.iter_mut().enumerate() {
                let v = a * (2.0 * std::f64::consts::PI * f * t as f64 / f64::from(rate)).sin();
                *s = (*s as f64 + v).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
            }
        }
        out
    }

    #[test]
    fn tone_decomposition_discovers_polyphony() {
        let rate = 48_000u32;
        let frames = 8_192usize;
        let mut samples = vec![0i32; frames];
        for (f, a) in [(180.0f64, 16384.0f64), (181.0, 8192.0), (270.0, 8192.0)] {
            for (t, s) in samples.iter_mut().enumerate() {
                let v = a * (2.0 * std::f64::consts::PI * f * t as f64 / f64::from(rate)).sin();
                *s = (*s as f64 + v).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
            }
        }
        let budget = CompoundBudget::default();
        let props = propose(&samples, 1, rate, frames as u64, &budget).unwrap();
        assert!(!props.is_empty());
        let mean_abs = |g: &CompoundGraph| -> f64 {
            let m = g.materialize().unwrap();
            samples
                .iter()
                .zip(m.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .sum::<f64>()
                / frames as f64
        };
        let single = props
            .iter()
            .find(|p| p.family == CompoundFamily::GainOscillator)
            .map(|p| mean_abs(&p.graph))
            .expect("a single-oscillator probe");
        let decomposed = props
            .iter()
            .filter(|p| p.family == CompoundFamily::ToneDecomposition)
            .map(|p| mean_abs(&p.graph))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("a tone decomposition");
        // The decomposition must capture both the detuned pair and the third
        // tone: its residual should be a fraction of a single oscillator's.
        assert!(
            decomposed < single * 0.25,
            "tone decomposition residual {decomposed:.0} did not beat single oscillator {single:.0}"
        );
    }

    #[test]
    fn tone_decomposition_matches_deterministic_polyphony() {
        use crate::compound::{CompoundGraph, CompoundNode, CompoundOp};
        let rate = 48_000u32;
        let frames = 24_000u64;
        let osc = |f: u32, a: i32, p: u64| CompoundNode {
            op: CompoundOp::Oscillator {
                freq_hz: f,
                amp_q16: a,
                phase0: p,
            },
            children: vec![],
        };
        let g = CompoundGraph {
            channels: 1,
            frames,
            sample_rate_hz: rate,
            nodes: vec![
                osc(180, 2, 0),
                osc(181, 1, 1 << 40),
                osc(270, 1, 1 << 41),
                CompoundNode {
                    op: CompoundOp::Add,
                    children: vec![0, 1, 2],
                },
            ],
        };
        let full = g.materialize().unwrap();
        let samples: Vec<i32> = full[..8_192].to_vec();
        let budget = CompoundBudget::default();
        let props = propose(&samples, 1, rate, 8_192, &budget).unwrap();
        let best = props
            .iter()
            .map(|p| {
                let seed =
                    crate::inverse::seed::SeedObject::from_graph_target(p.graph.clone(), &samples)
                        .unwrap()
                        .unwrap();
                (seed.complete_bytes(), p.family)
            })
            .min_by_key(|(b, _)| *b)
            .unwrap();
        // The deterministic three-oscillator construction must be discovered and
        // closed to within the integer oscillator's representational floor: the
        // residual is a few code units, not tens of kilobytes. (~0.3 B/sample is
        // the entropy of that quantisation noise.)
        assert!(
            best.0 < 3_500,
            "best seed {} bytes (family {:?}) on a known 3-oscillator construction",
            best.0,
            best.1
        );
    }

    #[test]
    fn solve_basis_recovers_exact_tone_sums() {
        let rate = 48_000u32;
        let n = 8_192usize;
        let x: Vec<f64> = (0..n)
            .map(|t| {
                let tf = t as f64;
                let w = 2.0 * std::f64::consts::PI * tf / f64::from(rate);
                65_535.0 * (180.0 * w).sin()
                    + 32_768.0 * (181.0 * w + 0.5).sin()
                    + 32_768.0 * (270.0 * w + 1.3).sin()
            })
            .collect();
        let coeffs = solve_basis(&x, f64::from(rate), &[180.0, 181.0, 270.0]);
        let model = synthesize(&coeffs, f64::from(rate), &[180.0, 181.0, 270.0], n);
        let rms = (residual_energy(&x, &model) / n as f64).sqrt();
        assert!(rms < 1.0, "LS residual rms {rms} on an exact tone sum");
    }

    #[test]
    fn tone_decomposition_closes_a_pure_sine() {
        let fixture = crate::entropy::corpus::named("single-sine").unwrap();
        let ch = usize::from(fixture.channels);
        let rate = 48_000u32;
        let samples: Vec<i32> = fixture
            .samples
            .iter()
            .step_by(ch)
            .take(8_192)
            .copied()
            .collect();
        let budget = CompoundBudget::default();
        let props = propose(&samples, 1, rate, samples.len() as u64, &budget).unwrap();
        // The proposer's job is the *fit*: it must close the pure sine to within
        // the integer oscillator's representational floor (a 1-code-unit
        // rounding difference), leaving an almost-zero residual. How many bytes
        // that residual costs is the entropy layer's question, not the
        // proposer's.
        let best_fit = props
            .iter()
            .map(|p| {
                let m = p.graph.materialize().unwrap();
                samples
                    .iter()
                    .zip(m.iter())
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                    .sum::<f64>()
                    / samples.len() as f64
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_fit < 1.0,
            "proposer fit mean|r| = {best_fit} on a pure sine"
        );
    }

    #[test]
    fn discovers_a_pure_tone_fundamental() {
        let rate = 48_000u32;
        let frames = 24_000usize;
        let samples = synth_harmonic(220.0, rate, frames, &[8_000.0]);
        let budget = CompoundBudget::default();
        let props = propose(&samples, 1, rate, frames as u64, &budget).unwrap();
        assert!(!props.is_empty());
        // A single-oscillator or bank family must appear and be near 220 Hz.
        let mut found = false;
        for p in &props {
            let expected = p.graph.materialize().unwrap();
            // Residual is small relative to the tone.
            let err: f64 = samples
                .iter()
                .zip(expected.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .sum::<f64>()
                / frames as f64;
            if err < 200.0 {
                found = true;
            }
        }
        assert!(found, "no proposal approximated a pure 220 Hz tone");
        assert!(matches!(
            props[0].family,
            CompoundFamily::Oscillator
                | CompoundFamily::GainOscillator
                | CompoundFamily::HarmonicBank
        ));
    }

    #[test]
    fn detects_a_harmonic_bank() {
        let rate = 48_000u32;
        let frames = 24_000usize;
        let samples = synth_harmonic(300.0, rate, frames, &[8_000.0, 4_000.0, 2_000.0, 1_000.0]);
        let budget = CompoundBudget::default();
        let props = propose(&samples, 1, rate, frames as u64, &budget).unwrap();
        let bank = props
            .iter()
            .find(|p| p.family == CompoundFamily::HarmonicBank)
            .expect("a harmonic bank should be proposed");
        let expected = bank.graph.materialize().unwrap();
        let err: f64 = samples
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .sum::<f64>()
            / frames as f64;
        // A four-harmonic bank should be a good approximation (unit ~ 8000).
        assert!(err < 400.0, "harmonic bank residual too large: {err}");
    }

    #[test]
    fn silence_and_constant_are_exact() {
        let budget = CompoundBudget::default();
        let sil = vec![0i32; 1_000];
        let props = propose(&sil, 1, 48_000, 1_000, &budget).unwrap();
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].family, CompoundFamily::Silence);
        assert_eq!(props[0].graph.materialize().unwrap(), sil);

        let cst = vec![123i32; 1_000];
        let props = propose(&cst, 1, 48_000, 1_000, &budget).unwrap();
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].family, CompoundFamily::Constant);
        assert_eq!(props[0].graph.materialize().unwrap(), cst);
    }

    #[test]
    fn proposals_are_valid_graphs_and_deterministic() {
        let rate = 44_100u32;
        let frames = 12_000usize;
        let samples = synth_harmonic(440.0, rate, frames, &[6_000.0, 3_000.0, 1_500.0]);
        let budget = CompoundBudget::default();
        let a = propose(&samples, 1, rate, frames as u64, &budget).unwrap();
        let b = propose(&samples, 1, rate, frames as u64, &budget).unwrap();
        assert_eq!(a.len(), b.len());
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(p.label, q.label);
            assert_eq!(p.graph, q.graph);
            p.graph.validate().unwrap();
        }
        // Multichannel is out of domain for mono-first Compound.
        assert!(
            propose(&samples, 2, rate, (frames / 2) as u64, &budget)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn budget_is_enforced() {
        let rate = 48_000u32;
        let frames = 8_000usize;
        let samples = synth_harmonic(500.0, rate, frames, &[5_000.0, 2_000.0, 1_000.0]);
        let budget = CompoundBudget {
            max_candidates: 2,
            ..CompoundBudget::default()
        };
        let props = propose(&samples, 1, rate, frames as u64, &budget).unwrap();
        assert!(props.len() <= 2);
    }
}

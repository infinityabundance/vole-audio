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

/// Default lowest fundamental searched (Hz).
pub const DEFAULT_MIN_FREQ_HZ: u32 = 40;
/// Default highest fundamental searched (Hz).
pub const DEFAULT_MAX_FREQ_HZ: u32 = 4_000;

/// Bounded blind-proposal budget. Every field is an explicit ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompoundBudget {
    /// Hard ceiling on proposed graphs.
    pub max_candidates: usize,
    /// Largest harmonic bank (number of oscillators) proposed.
    pub max_harmonics: usize,
    /// Largest number of tonal layers produced by residual-guided refinement.
    pub max_layers: usize,
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
            max_layers: 3,
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
    EchoicBank,
    Layered,
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
            CompoundFamily::EchoicBank => "add(bank,delay(bank))",
            CompoundFamily::Layered => "layered",
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

    // Residual-guided iterative decomposition: subtract the current tonal
    // explanation, re-analyse the residual, and add another tonal layer, up to
    // `max_layers`. Each layer is an independent bank, so a polyphonic mixture
    // of non-harmonically-related tones is expressed as `Add(bank0, bank1, ...)`.
    if budget.max_layers >= 2
        && let Some(mut current) = bank.clone()
    {
        for layer in 2..=budget.max_layers {
            let Ok(base_signal) = current.materialize() else {
                break;
            };
            let residual: Vec<f64> = x
                .iter()
                .zip(base_signal.iter())
                .map(|(&a, &b)| a - f64::from(b))
                .collect();
            let r_slice = analysis_slice(n, peak_band, rms.len(), budget.analysis_frames);
            let r_window = &residual[r_slice.clone()];
            let Some((f1, s1)) = estimate_fundamental(
                r_window,
                fs,
                f64::from(budget.min_freq_hz),
                f64::from(budget.max_freq_hz),
            ) else {
                break;
            };
            if s1 <= 0.15 {
                break;
            }
            let f1 = refine_frequency(
                &residual,
                fs,
                f1,
                f64::from(budget.min_freq_hz),
                f64::from(budget.max_freq_hz),
                sample_rate_hz,
            );
            let max_k = ((f64::from(budget.max_freq_hz) / f1).floor() as usize)
                .min(budget.max_harmonics)
                .max(1);
            let r_offset = r_slice.start;
            let Some(g2) = tonal_graph(
                &harmonic_amplitudes(r_window, fs, f1, max_k, r_offset),
                f1,
                frames,
                sample_rate_hz,
            ) else {
                break;
            };
            let Some(next) = layered(current, g2) else {
                break;
            };
            current = next.clone();
            out.push(CompoundProposal {
                family: CompoundFamily::Layered,
                label: format!("layered(f0={f0:.2},f1={f1:.2},layers={layer})"),
                graph: next,
            });
        }
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

/// Code-domain units produced per `amp_q16` unit at unity gain is `2^15`; an
/// explicit root gain rescales a probe so the whole `amp_q16` range is usable.
/// The gain is chosen from the peak amplitude: it is the smallest power of two
/// (clamped to `[1, 65536]`) whose `amp_q16 = round(2a/gain)` stays in range.
fn choose_gain(peak: f64) -> i32 {
    if !peak.is_finite() || peak <= 0.0 {
        return 1;
    }
    let need = 2.0 * peak / 65_535.0;
    let mut gain = 1.0f64;
    while gain < need && gain < 65_536.0 {
        gain *= 2.0;
    }
    gain.min(65_536.0) as i32
}

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

/// Adaptive-gain tonal graph: one oscillator for a pure tone, an `Add` bank for a
/// multi-harmonic tone, always followed by a root gain chosen from the peak
/// amplitude so arbitrary code-domain levels are representable exactly.
fn tonal_graph(harmonics: &[(f64, f64)], f0: f64, frames: u64, rate: u32) -> Option<CompoundGraph> {
    let peak = harmonics.iter().map(|(a, _)| *a).fold(0.0f64, f64::max);
    if peak <= 0.0 {
        return None;
    }
    let gain = choose_gain(peak);
    let mut nodes: Vec<CompoundNode> = Vec::new();
    let mut children: Vec<u16> = Vec::new();
    for (k, &(amp, phase)) in harmonics.iter().enumerate() {
        let q = (2.0 * amp / f64::from(gain)).round().clamp(0.0, 65_535.0) as i32;
        if q == 0 {
            continue;
        }
        let freq = (f0 * (k + 1) as f64).round();
        if freq < 1.0 || freq > f64::from(rate / 2) {
            continue;
        }
        let idx = nodes.len() as u16;
        nodes.push(node(
            CompoundOp::Oscillator {
                freq_hz: freq as u32,
                amp_q16: q,
                phase0: phase0_from_angle(phase),
            },
            vec![],
        ));
        children.push(idx);
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
    let root = (nodes.len() - 1) as u16;
    nodes.push(node(CompoundOp::Gain { q16: gain }, vec![root]));
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

/// Add two independent graphs.
fn layered(a: CompoundGraph, b: CompoundGraph) -> Option<CompoundGraph> {
    // Re-index b's nodes after a's.
    let offset = u16::try_from(a.nodes.len()).ok()?;
    let total = a.nodes.len().checked_add(b.nodes.len())?;
    if total as u32 > crate::compound::MAX_COMPOUND_NODES {
        return None;
    }
    let b_last = (b.nodes.len() - 1) as u16;
    let mut nodes = a.nodes.clone();
    for n in b.nodes {
        let children = n
            .children
            .iter()
            .map(|&c| c.checked_add(offset))
            .collect::<Option<Vec<u16>>>()?;
        nodes.push(node(n.op, children));
    }
    let ra = (a.nodes.len() - 1) as u16;
    let rb = offset + b_last;
    nodes.push(node(CompoundOp::Add, vec![ra, rb]));
    let mut g = a;
    g.nodes = nodes;
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
    fn layered_decomposition_improves_polyphony() {
        let rate = 48_000u32;
        let frames = 24_000usize;
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
        let layered = props
            .iter()
            .filter(|p| p.family == CompoundFamily::Layered)
            .map(|p| mean_abs(&p.graph))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("a layered decomposition");
        assert!(
            layered < single * 0.8,
            "layered residual {layered:.0} did not beat single oscillator {single:.0}"
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

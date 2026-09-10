//! Deterministic flagship-corpus generation (Phase M).
//!
//! Every object in the flagship corpus is **generated**, never stored: the
//! manifest carries the generator specification and the canonical content hash,
//! and `corpus verify` regenerates the samples and proves the hash. Nothing is
//! tuned after results exist; changing a generator changes the manifest and
//! therefore the seal subject.
//!
//! Generation is integer-only (the frozen Q30 sine table from
//! `universe::phase`, splitmix64 for stochastic material), so an object is
//! bit-reproducible on any host.
//!
//! The stratification axes are deliberately **orthogonal**, because a single
//! "structured vs random" label cannot interpret a result: a signal can be
//! random in time yet strongly redundant across channels (the H.2
//! `random-control` lesson). Each object names representation, amplitude
//! occupancy, channel structure, temporal structure and entropy character.
//!
//! Negative controls are hostile by construction: full i32 occupancy,
//! independent per-channel seeds, no duplicated channels, no shared low-byte
//! structure, no short accidental periods and no DC bias.

use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;
use serde::{Deserialize, Serialize};

/// Sample rates the flagship corpus freezes (contract §48).
pub const RATES: [u32; 4] = [44_100, 48_000, 96_000, 192_000];

/// FLAC's own channel ceiling: objects above it are `B1` `NOT_APPLICABLE` by
/// format domain and must never enter a B1-vs-VOLE aggregate.
pub const B1_MAX_CHANNELS: u8 = 8;

// ---------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------

macro_rules! axis {
    ($name:ident { $( $variant:ident => $label:literal ),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub enum $name { $( #[serde(rename = $label)] $variant ),+ }
        impl $name {
            pub const fn as_str(self) -> &'static str {
                match self { $( $name::$variant => $label ),+ }
            }
        }
    };
}

axis!(Representation {
    Literal => "literal",
    ExactRepetition => "exact_repetition",
    Oscillator => "oscillator",
    Wavetable => "wavetable",
    Residual => "residual",
    Compound => "compound",
    Noise => "noise",
});

axis!(Amplitude {
    LowByte => "low_byte",
    S16 => "s16_like",
    S24 => "s24_like",
    Full => "full_i32",
});

axis!(ChannelStructure {
    Mono => "mono",
    IdenticalStereo => "identical_stereo",
    CorrelatedStereo => "correlated_stereo",
    AnticorrelatedStereo => "anticorrelated_stereo",
    IndependentStereo => "independent_stereo",
    Multichannel => "multichannel",
});

axis!(Temporal {
    Stationary => "stationary",
    Transient => "transient",
    Loop => "loop",
    OneShot => "one_shot",
    SlowlyVarying => "slowly_varying",
    StronglyModulated => "strongly_modulated",
});

axis!(Entropy {
    HighlyPredictable => "highly_predictable",
    LocallyPredictable => "locally_predictable",
    GloballyPeriodic => "globally_periodic",
    SparseResidual => "sparse_residual",
    SpectrallyStructured => "spectrally_structured",
    FullWidthRandom => "full_width_random",
    Scrambled => "scrambled",
});

axis!(WaveShape {
    Sine => "sine",
    Saw => "saw",
    Square => "square",
    Triangle => "triangle",
});

/// Loop semantics (identity-bearing: a loop is not a one-shot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Semantics {
    Loop { period_frames: u32 },
    OneShot,
}

impl Semantics {
    pub const fn kind(self) -> &'static str {
        match self {
            Semantics::Loop { .. } => "loop",
            Semantics::OneShot => "one_shot",
        }
    }

    pub const fn period_frames(self) -> Option<u32> {
        match self {
            Semantics::Loop { period_frames } => Some(period_frames),
            Semantics::OneShot => None,
        }
    }
}

/// The concrete signal recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    Silence,
    /// Constant level as a fraction of full scale (`amp_milli` in 0..=1000).
    Dc {
        amp_milli: i32,
    },
    Oscillator {
        hz: u32,
    },
    Harmonic {
        hz: u32,
        partials: u8,
    },
    /// One cycle of `shape` every `period` frames.
    Wavetable {
        period: u32,
        shape: WaveShape,
    },
    /// A `period`-frame block of `shape` repeated verbatim.
    ExactRepeat {
        period: u32,
        shape: WaveShape,
    },
    Am {
        carrier_hz: u32,
        mod_hz: u32,
        depth_milli: u16,
    },
    Fm {
        carrier_hz: u32,
        dev_hz: u32,
        index_milli: u16,
    },
    /// Decaying impulses every `spacing` frames (`decay_shift` sets the decay).
    Impulses {
        spacing: u32,
        decay_shift: u8,
    },
    /// `hits` percussive events with a shaped decay.
    Percussive {
        hits: u32,
        decay_shift: u8,
    },
    /// Formant-ish voice: `f0_hz`, three resonant peaks, slow vibrato.
    Voice {
        f0_hz: u32,
        formant_shift: u8,
        vibrato_hz: u32,
    },
    /// `voices` detuned oscillators summed (polyphonic mixture).
    Polyphony {
        voices: u8,
        base_hz: u32,
    },
    /// One-pole lowpass filtered noise (`cutoff_milli` in 1..=999).
    OnePole {
        cutoff_milli: u16,
        seed: u64,
    },
    /// Sine plus `corrections` sparse integer corrections (residual-governed).
    SparseResidual {
        hz: u32,
        corrections: u32,
        seed: u64,
    },
    /// Full-width uniform noise.
    Uniform {
        seed: u64,
    },
    /// Bit-reversed noise: uniform values, byte structure destroyed.
    Scrambled {
        seed: u64,
    },
}

/// One frozen corpus object specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spec {
    pub id: String,
    pub representation: Representation,
    pub amplitude: Amplitude,
    pub channel_structure: ChannelStructure,
    pub temporal: Temporal,
    pub entropy: Entropy,
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub frames: usize,
    pub semantics: Semantics,
    pub signal: Signal,
    /// Generator identity written into the manifest.
    pub source: String,
}

impl Spec {
    /// Duration in milliseconds (exact integer division).
    pub const fn duration_ms(&self) -> u64 {
        self.frames as u64 * 1000 / self.sample_rate_hz as u64
    }

    /// Whether B1 (FLAC) can encode this object by format domain.
    pub const fn b1_comparable(&self) -> bool {
        self.channels >= 1 && self.channels <= B1_MAX_CHANNELS
    }

    /// The generator identity string for the manifest.
    pub fn generator_kind(&self) -> &'static str {
        match self.signal {
            Signal::Silence => "silence",
            Signal::Dc { .. } => "dc",
            Signal::Oscillator { .. } => "oscillator",
            Signal::Harmonic { .. } => "harmonic",
            Signal::Wavetable { .. } => "wavetable",
            Signal::ExactRepeat { .. } => "exact_repeat",
            Signal::Am { .. } => "am",
            Signal::Fm { .. } => "fm",
            Signal::Impulses { .. } => "impulses",
            Signal::Percussive { .. } => "percussive",
            Signal::Voice { .. } => "voice",
            Signal::Polyphony { .. } => "polyphony",
            Signal::OnePole { .. } => "one_pole",
            Signal::SparseResidual { .. } => "sparse_residual",
            Signal::Uniform { .. } => "uniform",
            Signal::Scrambled { .. } => "scrambled",
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic primitives
// ---------------------------------------------------------------------------

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Q30 sine from the frozen 4096-entry table.
fn sine_q30(phase: u64) -> i64 {
    let table = crate::universe::phase::sine_table_i32();
    i64::from(table[(phase & 4095) as usize])
}

/// Sine at `hz` for frame `f`, with a 1/4096-cycle phase offset.
fn sine_hz(f: usize, hz: u32, rate: u32, phase_offset: u32) -> i64 {
    let phase = (f as u64)
        .wrapping_mul(4096)
        .wrapping_mul(u64::from(hz))
        .wrapping_div(u64::from(rate.max(1)))
        .wrapping_add(u64::from(phase_offset));
    sine_q30(phase)
}

/// One cycle of `shape` at fractional index `num/den` of the period.
fn wave_q30(shape: WaveShape, num: u64, den: u64) -> i64 {
    let den = den.max(1);
    let x = num % den;
    match shape {
        WaveShape::Sine => sine_q30(x * 4096 / den),
        WaveShape::Saw => (x as i64) * (2 << 30) / den as i64 - (1 << 30),
        WaveShape::Square => {
            if x * 2 < den {
                1 << 30
            } else {
                -(1 << 30)
            }
        }
        WaveShape::Triangle => {
            // u = 2x - den in [-den, den); peak at the period midpoint.
            let u = (2 * x) as i64 - den as i64;
            let a = u.abs();
            (1 << 30) * (den as i64 - 2 * a) / den as i64
        }
    }
}

/// Scale a Q30-domain value into the requested amplitude occupancy.
fn scale(v_q30: i64, amp: Amplitude) -> i32 {
    let full = v_q30.saturating_mul(2);
    let s = match amp {
        Amplitude::Full => full,
        Amplitude::S24 => full >> 8,
        Amplitude::S16 => full >> 16,
        Amplitude::LowByte => full >> 23,
    };
    s.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// Per-channel generator state.
struct ChanState {
    seed: u64,
    lp: i64,
    rate: u32,
    frames: usize,
}

impl ChanState {
    /// Stateless-per-frame noise stream: a hash of (seed, frame), so the value
    /// never depends on call order.
    fn noise(&self, f: usize) -> u64 {
        let mut s = self.seed ^ (f as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        splitmix(&mut s)
    }
}

/// One channel's value at frame `f`.
fn channel_value(sig: Signal, amp: Amplitude, f: usize, ch: usize, st: &mut ChanState) -> i32 {
    let rate = st.rate.max(1);
    let co = (ch as u32).wrapping_mul(512);
    match sig {
        Signal::Silence => 0,
        Signal::Dc { amp_milli } => scale(
            (1 << 30) * i64::from(amp_milli.clamp(-1000, 1000)) / 1000,
            amp,
        ),
        Signal::Oscillator { hz } => scale(sine_hz(f, hz, rate, co), amp),
        Signal::Harmonic { hz, partials } => {
            let mut acc = 0i64;
            for n in 1..=u32::from(partials.max(1)) {
                let g = (1 << 22) / (i64::from(n) * i64::from(n));
                acc += (sine_hz(f, hz.saturating_mul(n), rate, co.wrapping_add(n * 64)) * g) >> 22;
            }
            scale(acc, amp)
        }
        Signal::Wavetable { period, shape } | Signal::ExactRepeat { period, shape } => {
            let p = u64::from(period.max(1));
            wave_scale(shape, (f as u64) % p, p, amp)
        }
        Signal::Am {
            carrier_hz,
            mod_hz,
            depth_milli,
        } => {
            let carrier = sine_hz(f, carrier_hz, rate, co);
            let m = (sine_hz(f, mod_hz, rate, 0) + (1 << 30)) >> 1; // [0, 2^30]
            let depth = i64::from(depth_milli.clamp(0, 1000));
            let gain = (1 << 30) - depth * ((1 << 30) - m) / 1000;
            scale((carrier * gain) >> 30, amp)
        }
        Signal::Fm {
            carrier_hz,
            dev_hz,
            index_milli,
        } => {
            let modulator = sine_hz(f, dev_hz, rate, 0);
            // Q30 -> 1/4096-cycle units, scaled by the modulation index.
            let offset = (modulator >> 20) * i64::from(index_milli.clamp(0, 4000)) / 1000;
            let base = (f as u64)
                .wrapping_mul(4096)
                .wrapping_mul(u64::from(carrier_hz))
                .wrapping_div(u64::from(rate));
            let phase = (base as i64 + offset).rem_euclid(4096) as u64;
            scale(sine_q30(phase.wrapping_add(u64::from(co))), amp)
        }
        Signal::Impulses {
            spacing,
            decay_shift,
        } => {
            let sp = (spacing.max(1)) as usize;
            let d = f % sp;
            let v = if d == 0 {
                1 << 30
            } else {
                (1 << 30) >> (d >> decay_shift).min(30)
            };
            scale(v, amp)
        }
        Signal::Percussive { hits, decay_shift } => {
            let per = (st.frames / hits.max(1) as usize).max(1);
            let d = f % per;
            let v = if d == 0 {
                1 << 30
            } else {
                (1 << 30) >> (d >> decay_shift).min(30)
            };
            scale(v, amp)
        }
        Signal::Voice {
            f0_hz,
            formant_shift,
            vibrato_hz,
        } => {
            let vib = sine_hz(f, vibrato_hz, rate, 0) >> 28; // a few Hz of deviation
            let f0 = (i64::from(f0_hz) + vib * i64::from(f0_hz) / 64).max(20) as u32;
            let mut acc = 0i64;
            for k in 0..4u32 {
                let mult = 1 + k * 2 + u32::from(formant_shift % 2);
                acc +=
                    sine_hz(f, f0.saturating_mul(mult), rate, co.wrapping_add(k * 96)) >> (k + 1);
            }
            scale(acc, amp)
        }
        Signal::Polyphony { voices, base_hz } => {
            let mut acc = 0i64;
            for v in 0..u32::from(voices.max(1)) {
                let hz = base_hz * (100 + v) / 100 + (ch as u32) * 7;
                acc += sine_hz(f, hz.max(1), rate, v.wrapping_mul(137)) >> 2;
            }
            scale(acc, amp)
        }
        Signal::OnePole { cutoff_milli, .. } => {
            let raw = (st.noise(f) as i32) as i64;
            let a = i64::from(cutoff_milli.clamp(1, 999));
            st.lp = st.lp * (1000 - a) / 1000 + raw * a / 1000;
            let v = st.lp.clamp(-(1i64 << 30), 1i64 << 30);
            match amp {
                Amplitude::Full => (v * 2) as i32,
                Amplitude::S24 => (v / 128) as i32,
                Amplitude::S16 => (v / 32_768) as i32,
                Amplitude::LowByte => (v / 4_194_304) as i32,
            }
        }
        Signal::SparseResidual {
            hz, corrections, ..
        } => {
            let carrier = sine_hz(f, hz, rate, co);
            let stride = (st.frames / corrections.max(1) as usize).max(1);
            let corr = if corrections > 0 && f.is_multiple_of(stride) {
                ((st.noise(f) as i32) as i64) >> 4
            } else {
                0
            };
            scale(carrier + corr, amp)
        }
        Signal::Uniform { .. } => {
            let r = (st.noise(f) as u32) as i32;
            match amp {
                Amplitude::Full => r,
                Amplitude::S24 => r >> 8,
                Amplitude::S16 => r >> 16,
                Amplitude::LowByte => r >> 23,
            }
        }
        Signal::Scrambled { .. } => {
            let r = ((st.noise(f).reverse_bits()) as u32) as i32;
            match amp {
                Amplitude::Full => r,
                Amplitude::S24 => r >> 8,
                Amplitude::S16 => r >> 16,
                Amplitude::LowByte => r >> 23,
            }
        }
    }
}

fn wave_scale(shape: WaveShape, num: u64, den: u64, amp: Amplitude) -> i32 {
    scale(wave_q30(shape, num, den), amp)
}

fn temporal_gain(t: Temporal, f: usize, frames: usize) -> i64 {
    let n = frames.max(1) as i64;
    let x = f as i64;
    match t {
        Temporal::Stationary | Temporal::Loop | Temporal::OneShot => 1 << 16,
        Temporal::Transient => (1 << 16) * (1000 - (x * 1000 / n).min(999)) / 1000 + (1 << 10),
        Temporal::SlowlyVarying => (1 << 16) * (500 + 500 * x / n) / 1000,
        Temporal::StronglyModulated => {
            let m = sine_q30(((x as u64) * 4096 / (n.max(1) as u64)).min(4095)); // one cycle
            ((1 << 16) * (m + (1 << 30))) >> 30
        }
    }
}

fn apply_gain(v: i32, gain_q16: i64) -> i32 {
    ((i64::from(v) * gain_q16) >> 16).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// Deterministic per-object seed.
pub fn spec_seed(id: &str) -> u64 {
    let mut s: u64 = 0x6d65_7461_2d75_776f; // "meta-uwo"
    for b in id.bytes() {
        s = s.wrapping_mul(31).wrapping_add(u64::from(b));
    }
    s
}

/// Generate the exact canonical interleaved samples for one spec.
pub fn generate(spec: &Spec) -> Result<Vec<i32>> {
    if spec.channels == 0 || spec.channels > crate::limits::MAX_CHANNELS as u8 {
        return Err(Error::malformed(
            "corpus object channel count out of domain",
        ));
    }
    if spec.frames == 0 {
        return Err(Error::malformed("corpus object has no frames"));
    }
    if !RATES.contains(&spec.sample_rate_hz) {
        return Err(Error::malformed(
            "corpus object rate is not one of the frozen rates",
        ));
    }
    let ch = usize::from(spec.channels);
    let mut out = vec![0i32; spec.frames * ch];
    let base_seed = spec_seed(&spec.id);
    let mut states: Vec<ChanState> = (0..ch)
        .map(|c| ChanState {
            // Independent channels: independent seeds, derived deterministically.
            seed: base_seed ^ (c as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            lp: 0,
            rate: spec.sample_rate_hz,
            frames: spec.frames,
        })
        .collect();
    // Channel 0 is generated once per frame and reused by the correlated
    // structures, so "identical" really is identical.
    let mut ch0 = ChanState {
        seed: states[0].seed,
        lp: 0,
        rate: spec.sample_rate_hz,
        frames: spec.frames,
    };
    states[0].lp = 0;

    for f in 0..spec.frames {
        let gain = temporal_gain(spec.temporal, f, spec.frames);
        let base = apply_gain(
            channel_value(spec.signal, spec.amplitude, f, 0, &mut ch0),
            gain,
        );
        for c in 0..ch {
            let v = match spec.channel_structure {
                ChannelStructure::Mono | ChannelStructure::IdenticalStereo => base,
                ChannelStructure::AnticorrelatedStereo => {
                    if c == 0 {
                        base
                    } else {
                        base.saturating_neg()
                    }
                }
                ChannelStructure::CorrelatedStereo => {
                    if c == 0 {
                        base
                    } else {
                        // Correlated but not identical: a small independent
                        // dither around a slightly attenuated copy.
                        let d = ((states[c].noise(f) as i32) as i64 >> 6) as i32;
                        (base - base / 8).saturating_add(d / 64)
                    }
                }
                ChannelStructure::IndependentStereo | ChannelStructure::Multichannel => {
                    if c == 0 {
                        base
                    } else {
                        let v = channel_value(spec.signal, spec.amplitude, f, c, &mut states[c]);
                        apply_gain(v, gain)
                    }
                }
            };
            out[f * ch + c] = v;
        }
    }
    Ok(out)
}

/// Canonical content hash of an object's samples.
pub fn canonical_sha256(samples: &[i32]) -> [u8; 32] {
    let mut h = Sha256::new();
    for s in samples {
        h.update(&s.to_le_bytes());
    }
    h.finalize()
}

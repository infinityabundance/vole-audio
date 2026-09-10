//! The frozen flagship-corpus membership (Phase M).
//!
//! This list **is** the corpus: it is code, it is in the seal subject, and it is
//! frozen before any flagship result exists. `corpus freeze` writes the manifest
//! (with content hashes) from it; `corpus verify` regenerates every object from
//! the manifest and cross-checks this list, so membership cannot drift silently
//! in either direction.
//!
//! Rules this list obeys:
//!
//! * **orthogonal stratification** — representation, amplitude occupancy,
//!   channel structure, temporal structure and entropy character are named
//!   independently, so a result can be interpreted (the H.2 `random-control`
//!   lesson: random in time, redundant across channels);
//! * **hostile negative controls** — full-width uniform noise and scrambled
//!   noise, with independent per-channel seeds, independent stereo rather than
//!   duplicated channels, no shared low-byte structure, no short periods, no DC
//!   bias and no reduced dynamic range;
//! * **format-domain honesty** — objects above FLAC's 8-channel ceiling are a
//!   separate population (`high_channel_stress`) and never enter a B1-vs-VOLE
//!   aggregate;
//! * **frozen rates** — 44.1 / 48 / 96 / 192 kHz only.

use super::generate::{
    Amplitude, B1_MAX_CHANNELS, ChannelStructure, Entropy, RATES, Representation, Semantics,
    Signal, Spec, Temporal, WaveShape,
};

struct B {
    out: Vec<Spec>,
}

impl B {
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        stem: &str,
        representation: Representation,
        amplitude: Amplitude,
        channel_structure: ChannelStructure,
        temporal: Temporal,
        entropy: Entropy,
        sample_rate_hz: u32,
        seconds_milli: u32,
        channels: u8,
        semantics: Semantics,
        signal: Signal,
    ) {
        assert!(RATES.contains(&sample_rate_hz), "{stem}: frozen rates only");
        let frames = (sample_rate_hz as u64 * u64::from(seconds_milli) / 1000) as usize;
        assert!(frames > 0, "{stem}: zero frames");
        // The id must identify the object uniquely and readably: stem, rate,
        // channel count, amplitude occupancy and channel structure are all part
        // of it (two objects differing only in channel structure are different
        // objects, not duplicates).
        let id = format!(
            "{stem}-{}hz-{}ch-{}-{}",
            sample_rate_hz,
            channels,
            amplitude.as_str(),
            channel_structure.as_str()
        );
        self.out.push(Spec {
            id,
            representation,
            amplitude,
            channel_structure,
            temporal,
            entropy,
            sample_rate_hz,
            channels,
            frames,
            semantics,
            signal,
            source: format!(
                "generated: corpus::generate ({}), splitmix64/Q30 sine table, integer-only",
                signal_kind(signal)
            ),
        });
    }
}

fn signal_kind(s: Signal) -> &'static str {
    match s {
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

/// The frozen membership.
pub fn specs() -> Vec<Spec> {
    use Amplitude::*;
    use ChannelStructure::*;
    use Entropy::*;
    use Representation::*;
    use Semantics::{Loop, OneShot};
    use Temporal::*;

    let mut b = B { out: Vec::new() };

    // --- structural: perfectly predictable, multi-rate, amplitude sweep ------
    b.add(
        "silence",
        Literal,
        Full,
        Mono,
        Stationary,
        HighlyPredictable,
        48_000,
        2000,
        1,
        OneShot,
        Signal::Silence,
    );
    for amp in [LowByte, S16, S24, Full] {
        b.add(
            "dc",
            Literal,
            amp,
            Mono,
            Stationary,
            HighlyPredictable,
            48_000,
            2000,
            1,
            OneShot,
            Signal::Dc { amp_milli: 900 },
        );
    }
    for amp in [LowByte, S16, S24, Full] {
        b.add(
            "osc-1k",
            Oscillator,
            amp,
            Mono,
            Stationary,
            HighlyPredictable,
            48_000,
            2000,
            1,
            OneShot,
            Signal::Oscillator { hz: 1000 },
        );
    }
    // One object per remaining rate (the amplitude sweep above already covers
    // 48 kHz mono, so repeating it here would duplicate an id).
    for (rate, ms) in [(44_100, 2000), (96_000, 1000), (192_000, 500)] {
        b.add(
            "osc-1k",
            Oscillator,
            S16,
            Mono,
            Stationary,
            HighlyPredictable,
            rate,
            ms,
            1,
            OneShot,
            Signal::Oscillator { hz: 1000 },
        );
    }
    for amp in [LowByte, S16, S24, Full] {
        b.add(
            "harmonic-220",
            Oscillator,
            amp,
            Mono,
            Stationary,
            LocallyPredictable,
            48_000,
            2000,
            1,
            OneShot,
            Signal::Harmonic {
                hz: 220,
                partials: 8,
            },
        );
    }
    for amp in [LowByte, S16, S24, Full] {
        b.add(
            "wavetable-saw-1k",
            Wavetable,
            amp,
            Mono,
            Stationary,
            GloballyPeriodic,
            48_000,
            2000,
            1,
            Loop { period_frames: 48 },
            Signal::Wavetable {
                period: 48,
                shape: WaveShape::Saw,
            },
        );
    }
    for shape in [WaveShape::Square, WaveShape::Triangle, WaveShape::Sine] {
        b.add(
            &format!("wavetable-{}", shape.as_str()),
            Wavetable,
            S16,
            Mono,
            Stationary,
            GloballyPeriodic,
            48_000,
            2000,
            1,
            Loop { period_frames: 48 },
            Signal::Wavetable { period: 48, shape },
        );
    }
    for shape in [
        WaveShape::Saw,
        WaveShape::Square,
        WaveShape::Sine,
        WaveShape::Triangle,
    ] {
        b.add(
            &format!("repeat-1000-{}", shape.as_str()),
            ExactRepetition,
            S24,
            Mono,
            Loop,
            GloballyPeriodic,
            48_000,
            2000,
            1,
            Loop {
                period_frames: 1000,
            },
            Signal::ExactRepeat {
                period: 1000,
                shape,
            },
        );
    }
    for (rate, ms) in [(96_000, 1000), (192_000, 500)] {
        b.add(
            "wavetable-saw-1k",
            Wavetable,
            S24,
            Mono,
            Stationary,
            GloballyPeriodic,
            rate,
            ms,
            1,
            Loop {
                period_frames: rate / 1000,
            },
            Signal::Wavetable {
                period: rate / 1000,
                shape: WaveShape::Saw,
            },
        );
    }

    // --- channel structure: identical / correlated / anti / independent ------
    b.add(
        "osc-1k",
        Oscillator,
        S16,
        IdenticalStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        2000,
        2,
        OneShot,
        Signal::Oscillator { hz: 1000 },
    );
    b.add(
        "osc-1k",
        Oscillator,
        S16,
        CorrelatedStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        2000,
        2,
        OneShot,
        Signal::Oscillator { hz: 1000 },
    );
    b.add(
        "osc-1k",
        Oscillator,
        S16,
        AnticorrelatedStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        2000,
        2,
        OneShot,
        Signal::Oscillator { hz: 1000 },
    );
    b.add(
        "osc-detuned",
        Oscillator,
        S16,
        IndependentStereo,
        Stationary,
        LocallyPredictable,
        48_000,
        2000,
        2,
        OneShot,
        Signal::Oscillator { hz: 997 },
    );
    b.add(
        "poly-6",
        Compound,
        S16,
        Multichannel,
        Stationary,
        LocallyPredictable,
        48_000,
        1000,
        6,
        OneShot,
        Signal::Polyphony {
            voices: 6,
            base_hz: 110,
        },
    );
    b.add(
        "poly-8",
        Compound,
        S16,
        Multichannel,
        Stationary,
        LocallyPredictable,
        48_000,
        1000,
        8,
        OneShot,
        Signal::Polyphony {
            voices: 8,
            base_hz: 110,
        },
    );
    b.add(
        "voice-8",
        Residual,
        S16,
        Multichannel,
        SlowlyVarying,
        SparseResidual,
        48_000,
        1000,
        8,
        OneShot,
        Signal::Voice {
            f0_hz: 120,
            formant_shift: 1,
            vibrato_hz: 5,
        },
    );
    b.add(
        "noise-s16-8",
        Noise,
        S16,
        Multichannel,
        Stationary,
        SpectrallyStructured,
        48_000,
        500,
        8,
        OneShot,
        Signal::Uniform { seed: 0x8008 },
    );
    // FLAC format-domain population: > 8 channels, B1 NOT_APPLICABLE.
    for ch in [12u8, 32] {
        b.add(
            "poly-stress",
            Compound,
            S16,
            Multichannel,
            Stationary,
            LocallyPredictable,
            48_000,
            250,
            ch,
            OneShot,
            Signal::Polyphony {
                voices: 12,
                base_hz: 110,
            },
        );
    }
    b.add(
        "noise-stress",
        Noise,
        Full,
        Multichannel,
        Stationary,
        FullWidthRandom,
        48_000,
        250,
        12,
        OneShot,
        Signal::Uniform { seed: 0x8123 },
    );

    // --- temporal structure -------------------------------------------------
    b.add(
        "perc-8",
        Compound,
        S16,
        Mono,
        Temporal::OneShot,
        SparseResidual,
        48_000,
        2000,
        1,
        OneShot,
        Signal::Percussive {
            hits: 8,
            decay_shift: 6,
        },
    );
    b.add(
        "perc-64",
        Compound,
        S16,
        Mono,
        Transient,
        SparseResidual,
        48_000,
        2000,
        1,
        OneShot,
        Signal::Percussive {
            hits: 64,
            decay_shift: 4,
        },
    );
    b.add(
        "impulses-256",
        Residual,
        S24,
        Mono,
        Transient,
        SparseResidual,
        48_000,
        2000,
        1,
        Loop { period_frames: 256 },
        Signal::Impulses {
            spacing: 256,
            decay_shift: 5,
        },
    );
    b.add(
        "am-440",
        Compound,
        S16,
        Mono,
        StronglyModulated,
        LocallyPredictable,
        48_000,
        2000,
        1,
        OneShot,
        Signal::Am {
            carrier_hz: 440,
            mod_hz: 3,
            depth_milli: 700,
        },
    );
    b.add(
        "fm-440",
        Compound,
        S16,
        Mono,
        StronglyModulated,
        LocallyPredictable,
        48_000,
        2000,
        1,
        OneShot,
        Signal::Fm {
            carrier_hz: 440,
            dev_hz: 5,
            index_milli: 1500,
        },
    );
    b.add(
        "voice-slow",
        Residual,
        S24,
        Mono,
        SlowlyVarying,
        SparseResidual,
        48_000,
        2000,
        1,
        OneShot,
        Signal::Voice {
            f0_hz: 110,
            formant_shift: 0,
            vibrato_hz: 4,
        },
    );
    b.add(
        "am-slow",
        Compound,
        S24,
        Mono,
        SlowlyVarying,
        LocallyPredictable,
        48_000,
        2000,
        1,
        OneShot,
        Signal::Am {
            carrier_hz: 330,
            mod_hz: 1,
            depth_milli: 400,
        },
    );

    // --- spectral / residual-grey material ---------------------------------
    for cutoff in [50u16, 200, 800] {
        b.add(
            &format!("onepole-c{cutoff}"),
            Noise,
            S24,
            Mono,
            Stationary,
            SpectrallyStructured,
            48_000,
            2000,
            1,
            OneShot,
            Signal::OnePole {
                cutoff_milli: cutoff,
                seed: 0x0100 + u64::from(cutoff),
            },
        );
    }
    b.add(
        "onepole-full",
        Noise,
        Full,
        IndependentStereo,
        Stationary,
        SpectrallyStructured,
        48_000,
        2000,
        2,
        OneShot,
        Signal::OnePole {
            cutoff_milli: 300,
            seed: 0x0300,
        },
    );
    for corrections in [64u32, 512] {
        b.add(
            &format!("sparse-c{corrections}"),
            Residual,
            S24,
            Mono,
            Stationary,
            SparseResidual,
            48_000,
            2000,
            1,
            OneShot,
            Signal::SparseResidual {
                hz: 220,
                corrections,
                seed: 0x0a00 + u64::from(corrections),
            },
        );
    }
    b.add(
        "sparse-s16",
        Residual,
        S16,
        CorrelatedStereo,
        Stationary,
        SparseResidual,
        48_000,
        2000,
        2,
        OneShot,
        Signal::SparseResidual {
            hz: 220,
            corrections: 256,
            seed: 0x0b00,
        },
    );

    // --- hostile negative controls (full width, independent, unbiased) ------
    for (rate, ms) in [
        (44_100, 2000),
        (48_000, 2000),
        (96_000, 1000),
        (192_000, 500),
    ] {
        b.add(
            "uniform-full",
            Noise,
            Full,
            Mono,
            Stationary,
            FullWidthRandom,
            rate,
            ms,
            1,
            OneShot,
            Signal::Uniform {
                seed: 0xf00d_0000 + u64::from(rate),
            },
        );
    }
    b.add(
        "uniform-full",
        Noise,
        Full,
        IndependentStereo,
        Stationary,
        FullWidthRandom,
        48_000,
        2000,
        2,
        OneShot,
        Signal::Uniform { seed: 0x1111_2222 },
    );
    b.add(
        "uniform-full",
        Noise,
        Full,
        IndependentStereo,
        Stationary,
        FullWidthRandom,
        96_000,
        1000,
        2,
        OneShot,
        Signal::Uniform { seed: 0x3333_4444 },
    );
    b.add(
        "uniform-full",
        Noise,
        Full,
        Multichannel,
        Stationary,
        FullWidthRandom,
        48_000,
        500,
        8,
        OneShot,
        Signal::Uniform { seed: 0x5555_6666 },
    );
    b.add(
        "uniform-full",
        Noise,
        Full,
        AnticorrelatedStereo,
        Stationary,
        FullWidthRandom,
        48_000,
        1000,
        2,
        OneShot,
        Signal::Uniform { seed: 0x7777_8888 },
    );
    for amp in [S16, S24, LowByte] {
        b.add(
            "uniform",
            Noise,
            amp,
            Mono,
            Stationary,
            FullWidthRandom,
            48_000,
            2000,
            1,
            OneShot,
            Signal::Uniform {
                seed: 0x9999_0000 + amp as u64,
            },
        );
    }
    b.add(
        "scrambled-full",
        Noise,
        Full,
        Mono,
        Stationary,
        Scrambled,
        48_000,
        2000,
        1,
        OneShot,
        Signal::Scrambled { seed: 0xabcd_0123 },
    );
    b.add(
        "scrambled-full",
        Noise,
        Full,
        IndependentStereo,
        Stationary,
        Scrambled,
        48_000,
        2000,
        2,
        OneShot,
        Signal::Scrambled { seed: 0xabcd_4567 },
    );
    b.add(
        "scrambled-full",
        Noise,
        Full,
        Multichannel,
        Stationary,
        Scrambled,
        48_000,
        500,
        8,
        OneShot,
        Signal::Scrambled { seed: 0xabcd_89ab },
    );

    // Second sweep: the remaining rates, amplitude occupancy and channel counts.
    // Kept in its own function so the first (structural) sweep stays readable.
    additional(&mut b);

    b.out
}

/// The second sweep of the frozen membership: more rates, more amplitude
/// occupancy, more channel counts, more transient/spectral material and more
/// hostile controls at the remaining rates.
fn additional(b: &mut B) {
    use Amplitude::*;
    use ChannelStructure::*;
    use Entropy::*;
    use Representation::*;
    use Semantics::{Loop, OneShot};
    use Temporal::*;

    // Structural objects at 44.1 kHz and further tonal frequencies.
    b.add(
        "osc-220",
        Oscillator,
        S16,
        Mono,
        Stationary,
        HighlyPredictable,
        44_100,
        3000,
        1,
        OneShot,
        Signal::Oscillator { hz: 220 },
    );
    b.add(
        "osc-440",
        Oscillator,
        S16,
        Mono,
        Stationary,
        HighlyPredictable,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Oscillator { hz: 440 },
    );
    b.add(
        "osc-880",
        Oscillator,
        S16,
        Mono,
        Stationary,
        HighlyPredictable,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Oscillator { hz: 880 },
    );
    b.add(
        "osc-3520",
        Oscillator,
        S16,
        Mono,
        Stationary,
        HighlyPredictable,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Oscillator { hz: 3520 },
    );
    b.add(
        "osc-220",
        Oscillator,
        S16,
        Mono,
        Stationary,
        HighlyPredictable,
        96_000,
        1500,
        1,
        OneShot,
        Signal::Oscillator { hz: 220 },
    );
    b.add(
        "osc-220",
        Oscillator,
        S16,
        Mono,
        Stationary,
        HighlyPredictable,
        192_000,
        750,
        1,
        OneShot,
        Signal::Oscillator { hz: 220 },
    );
    b.add(
        "harmonic-p4",
        Oscillator,
        S16,
        Mono,
        Stationary,
        LocallyPredictable,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Harmonic {
            hz: 220,
            partials: 4,
        },
    );
    b.add(
        "harmonic-110",
        Oscillator,
        S16,
        Mono,
        Stationary,
        LocallyPredictable,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Harmonic {
            hz: 110,
            partials: 16,
        },
    );
    b.add(
        "harmonic-220",
        Oscillator,
        S16,
        IdenticalStereo,
        Stationary,
        LocallyPredictable,
        96_000,
        1500,
        2,
        OneShot,
        Signal::Harmonic {
            hz: 220,
            partials: 8,
        },
    );
    for (rate, ms, period) in [
        (44_100u32, 3000u32, 441u32),
        (96_000, 1500, 960),
        (192_000, 750, 1920),
    ] {
        b.add(
            "repeat-shape",
            ExactRepetition,
            S16,
            Mono,
            Loop,
            GloballyPeriodic,
            rate,
            ms,
            1,
            Loop {
                period_frames: period,
            },
            Signal::ExactRepeat {
                period,
                shape: WaveShape::Sine,
            },
        );
    }
    b.add(
        "wavetable-saw-1k",
        Wavetable,
        S16,
        Mono,
        Stationary,
        GloballyPeriodic,
        44_100,
        3000,
        1,
        Loop { period_frames: 44 },
        Signal::Wavetable {
            period: 44,
            shape: WaveShape::Saw,
        },
    );
    b.add(
        "wavetable-tri-500",
        Wavetable,
        S24,
        IdenticalStereo,
        Stationary,
        GloballyPeriodic,
        48_000,
        3000,
        2,
        Loop { period_frames: 96 },
        Signal::Wavetable {
            period: 96,
            shape: WaveShape::Triangle,
        },
    );
    b.add(
        "wavetable-sq-2k",
        Wavetable,
        S16,
        CorrelatedStereo,
        Stationary,
        GloballyPeriodic,
        48_000,
        3000,
        2,
        Loop { period_frames: 24 },
        Signal::Wavetable {
            period: 24,
            shape: WaveShape::Square,
        },
    );

    // Stereo structure across amplitude classes.
    b.add(
        "osc-1k",
        Oscillator,
        S24,
        IdenticalStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        3000,
        2,
        OneShot,
        Signal::Oscillator { hz: 1000 },
    );
    b.add(
        "osc-1k",
        Oscillator,
        Full,
        IdenticalStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        3000,
        2,
        OneShot,
        Signal::Oscillator { hz: 1000 },
    );
    b.add(
        "osc-1k",
        Oscillator,
        S24,
        CorrelatedStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        3000,
        2,
        OneShot,
        Signal::Oscillator { hz: 1000 },
    );
    b.add(
        "osc-1k",
        Oscillator,
        S24,
        AnticorrelatedStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        3000,
        2,
        OneShot,
        Signal::Oscillator { hz: 1000 },
    );
    b.add(
        "osc-1k",
        Oscillator,
        Full,
        AnticorrelatedStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        3000,
        2,
        OneShot,
        Signal::Oscillator { hz: 1000 },
    );
    b.add(
        "dc-stereo",
        Literal,
        S24,
        IndependentStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        2000,
        2,
        OneShot,
        Signal::Dc { amp_milli: 700 },
    );
    b.add(
        "silence",
        Literal,
        Full,
        IdenticalStereo,
        Stationary,
        HighlyPredictable,
        48_000,
        2000,
        2,
        OneShot,
        Signal::Silence,
    );

    // Compound / modulated / transient variants.
    b.add(
        "am-440",
        Compound,
        S24,
        Mono,
        StronglyModulated,
        LocallyPredictable,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Am {
            carrier_hz: 440,
            mod_hz: 3,
            depth_milli: 700,
        },
    );
    b.add(
        "am-440",
        Compound,
        Full,
        Mono,
        StronglyModulated,
        LocallyPredictable,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Am {
            carrier_hz: 440,
            mod_hz: 3,
            depth_milli: 700,
        },
    );
    b.add(
        "fm-880",
        Compound,
        S24,
        Mono,
        StronglyModulated,
        LocallyPredictable,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Fm {
            carrier_hz: 880,
            dev_hz: 7,
            index_milli: 2200,
        },
    );
    b.add(
        "voice-96k",
        Residual,
        S16,
        Mono,
        SlowlyVarying,
        SparseResidual,
        96_000,
        1500,
        1,
        OneShot,
        Signal::Voice {
            f0_hz: 130,
            formant_shift: 1,
            vibrato_hz: 5,
        },
    );
    b.add(
        "voice-192k",
        Residual,
        S24,
        Mono,
        SlowlyVarying,
        SparseResidual,
        192_000,
        750,
        1,
        OneShot,
        Signal::Voice {
            f0_hz: 90,
            formant_shift: 0,
            vibrato_hz: 6,
        },
    );
    b.add(
        "perc-16",
        Compound,
        S16,
        Mono,
        Temporal::OneShot,
        SparseResidual,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Percussive {
            hits: 16,
            decay_shift: 5,
        },
    );
    b.add(
        "perc-128",
        Compound,
        S24,
        Mono,
        Transient,
        SparseResidual,
        48_000,
        3000,
        1,
        OneShot,
        Signal::Percussive {
            hits: 128,
            decay_shift: 3,
        },
    );
    b.add(
        "perc-32-96k",
        Compound,
        S16,
        Mono,
        Transient,
        SparseResidual,
        96_000,
        1500,
        1,
        OneShot,
        Signal::Percussive {
            hits: 32,
            decay_shift: 5,
        },
    );
    b.add(
        "impulses-64",
        Residual,
        S16,
        Mono,
        Transient,
        SparseResidual,
        48_000,
        3000,
        1,
        Loop { period_frames: 64 },
        Signal::Impulses {
            spacing: 64,
            decay_shift: 4,
        },
    );
    b.add(
        "impulses-1024",
        Residual,
        S16,
        Mono,
        Transient,
        SparseResidual,
        48_000,
        3000,
        1,
        Loop {
            period_frames: 1024,
        },
        Signal::Impulses {
            spacing: 1024,
            decay_shift: 7,
        },
    );

    // Spectral and residual material at more rates/classes.
    b.add(
        "onepole-1k",
        Noise,
        S16,
        Mono,
        Stationary,
        SpectrallyStructured,
        48_000,
        3000,
        1,
        OneShot,
        Signal::OnePole {
            cutoff_milli: 1000,
            seed: 0x0400,
        },
    );
    b.add(
        "onepole-96k",
        Noise,
        S24,
        Mono,
        Stationary,
        SpectrallyStructured,
        96_000,
        1500,
        1,
        OneShot,
        Signal::OnePole {
            cutoff_milli: 120,
            seed: 0x0500,
        },
    );
    b.add(
        "onepole-192k",
        Noise,
        S16,
        Mono,
        Stationary,
        SpectrallyStructured,
        192_000,
        750,
        1,
        OneShot,
        Signal::OnePole {
            cutoff_milli: 300,
            seed: 0x0600,
        },
    );
    b.add(
        "sparse-full",
        Residual,
        Full,
        Mono,
        Stationary,
        SparseResidual,
        48_000,
        3000,
        1,
        OneShot,
        Signal::SparseResidual {
            hz: 330,
            corrections: 1024,
            seed: 0x0c00,
        },
    );
    b.add(
        "sparse-44k",
        Residual,
        S16,
        Mono,
        Stationary,
        SparseResidual,
        44_100,
        3000,
        1,
        OneShot,
        Signal::SparseResidual {
            hz: 330,
            corrections: 128,
            seed: 0x0d00,
        },
    );

    // Polyphony at more rates/channel counts, including the 16-channel stress.
    b.add(
        "poly-4",
        Compound,
        S16,
        Multichannel,
        Stationary,
        LocallyPredictable,
        48_000,
        1500,
        4,
        OneShot,
        Signal::Polyphony {
            voices: 4,
            base_hz: 110,
        },
    );
    b.add(
        "poly-6",
        Compound,
        S16,
        Multichannel,
        Stationary,
        LocallyPredictable,
        44_100,
        1500,
        6,
        OneShot,
        Signal::Polyphony {
            voices: 6,
            base_hz: 110,
        },
    );
    b.add(
        "poly-12",
        Compound,
        S24,
        Multichannel,
        Stationary,
        LocallyPredictable,
        96_000,
        500,
        12,
        OneShot,
        Signal::Polyphony {
            voices: 12,
            base_hz: 110,
        },
    );
    b.add(
        "poly-16-stress",
        Compound,
        S16,
        Multichannel,
        Stationary,
        LocallyPredictable,
        48_000,
        250,
        16,
        OneShot,
        Signal::Polyphony {
            voices: 12,
            base_hz: 110,
        },
    );
    b.add(
        "noise-6ch",
        Noise,
        S16,
        Multichannel,
        Stationary,
        SpectrallyStructured,
        48_000,
        500,
        6,
        OneShot,
        Signal::Uniform { seed: 0x6006 },
    );

    // More hostile controls at the remaining rates.
    b.add(
        "scrambled-44k",
        Noise,
        Full,
        Mono,
        Stationary,
        Scrambled,
        44_100,
        3000,
        1,
        OneShot,
        Signal::Scrambled { seed: 0xabcd_cdef },
    );
    b.add(
        "scrambled-96k",
        Noise,
        Full,
        Mono,
        Stationary,
        Scrambled,
        96_000,
        1500,
        1,
        OneShot,
        Signal::Scrambled { seed: 0xf00d_1111 },
    );
    b.add(
        "scrambled-192k",
        Noise,
        Full,
        Mono,
        Stationary,
        Scrambled,
        192_000,
        750,
        1,
        OneShot,
        Signal::Scrambled { seed: 0xf00d_2222 },
    );
    b.add(
        "uniform-4ch",
        Noise,
        Full,
        Multichannel,
        Stationary,
        FullWidthRandom,
        96_000,
        500,
        4,
        OneShot,
        Signal::Uniform { seed: 0x2468_1357 },
    );
    b.add(
        "uniform-44k",
        Noise,
        Full,
        IndependentStereo,
        Stationary,
        FullWidthRandom,
        44_100,
        3000,
        2,
        OneShot,
        Signal::Uniform { seed: 0x1357_2468 },
    );
}

/// Population counts of the frozen membership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Populations {
    pub whole_corpus_objects: usize,
    pub b1_comparable_objects: usize,
    pub b1_excluded_objects: usize,
    pub high_channel_stress_objects: usize,
}

/// Count the populations of a spec list.
pub fn populations(specs: &[Spec]) -> Populations {
    let mut p = Populations {
        whole_corpus_objects: specs.len(),
        b1_comparable_objects: 0,
        b1_excluded_objects: 0,
        high_channel_stress_objects: 0,
    };
    for s in specs {
        if s.channels <= B1_MAX_CHANNELS {
            p.b1_comparable_objects += 1;
        } else {
            p.b1_excluded_objects += 1;
            p.high_channel_stress_objects += 1;
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membership_is_frozen_and_stratified() {
        let s = specs();
        // The target population (~100 objects).
        assert!(
            s.len() >= 95,
            "flagship corpus must be ~100 objects, got {}",
            s.len()
        );
        // Ids are unique.
        let mut ids: Vec<&str> = s.iter().map(|x| x.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate corpus ids");

        // Every frozen rate appears, and all four axes are exercised.
        for r in RATES {
            assert!(
                s.iter().any(|x| x.sample_rate_hz == r),
                "no object at {r} Hz"
            );
        }
        for rep in [
            Representation::Literal,
            Representation::Oscillator,
            Representation::Wavetable,
            Representation::ExactRepetition,
            Representation::Residual,
            Representation::Compound,
            Representation::Noise,
        ] {
            assert!(s.iter().any(|x| x.representation == rep), "no {rep:?}");
        }
        for amp in [
            Amplitude::LowByte,
            Amplitude::S16,
            Amplitude::S24,
            Amplitude::Full,
        ] {
            assert!(s.iter().any(|x| x.amplitude == amp), "no {amp:?}");
        }
        for ch in [
            ChannelStructure::Mono,
            ChannelStructure::IdenticalStereo,
            ChannelStructure::CorrelatedStereo,
            ChannelStructure::AnticorrelatedStereo,
            ChannelStructure::IndependentStereo,
            ChannelStructure::Multichannel,
        ] {
            assert!(s.iter().any(|x| x.channel_structure == ch), "no {ch:?}");
        }
        for t in [
            Temporal::Stationary,
            Temporal::Transient,
            Temporal::Loop,
            Temporal::OneShot,
            Temporal::SlowlyVarying,
            Temporal::StronglyModulated,
        ] {
            assert!(s.iter().any(|x| x.temporal == t), "no {t:?}");
        }
        for e in [
            Entropy::HighlyPredictable,
            Entropy::LocallyPredictable,
            Entropy::GloballyPeriodic,
            Entropy::SparseResidual,
            Entropy::SpectrallyStructured,
            Entropy::FullWidthRandom,
            Entropy::Scrambled,
        ] {
            assert!(s.iter().any(|x| x.entropy == e), "no {e:?}");
        }

        // The >8-channel format-domain population exists and is excluded from B1.
        let p = populations(&s);
        assert!(p.high_channel_stress_objects >= 2);
        assert_eq!(
            p.whole_corpus_objects,
            p.b1_comparable_objects + p.b1_excluded_objects
        );
    }

    #[test]
    fn negative_controls_are_hostile_by_construction() {
        let s = specs();
        // The hostile stratum is defined as full-width occupancy with a random
        // or scrambled character; lower-amplitude uniform variants exist as
        // *separate* (limited-occupancy) strata and are not hostile controls.
        let controls: Vec<&Spec> = s
            .iter()
            .filter(|x| {
                matches!(x.entropy, Entropy::FullWidthRandom | Entropy::Scrambled)
                    && x.amplitude == Amplitude::Full
            })
            .collect();
        assert!(
            controls.len() >= 8,
            "too few full-width negative controls: {}",
            controls.len()
        );
        for c in &controls {
            // Independent channels where stereo/multichannel, never duplicated.
            match c.channel_structure {
                ChannelStructure::IdenticalStereo | ChannelStructure::CorrelatedStereo => {
                    panic!("{}: negative controls must not reuse channels", c.id)
                }
                _ => {}
            }
        }
        // Both uniform and scrambled hostile controls are present.
        assert!(
            controls
                .iter()
                .any(|c| c.entropy == Entropy::FullWidthRandom)
        );
        assert!(controls.iter().any(|c| c.entropy == Entropy::Scrambled));
    }

    #[test]
    fn generation_is_deterministic_and_unbiased() {
        let s = specs();
        let noise = s
            .iter()
            .find(|x| x.id == "uniform-full-48000hz-1ch-full_i32-mono")
            .expect("full-width mono control");
        let a = super::super::generate::generate(noise).unwrap();
        let b = super::super::generate::generate(noise).unwrap();
        assert_eq!(a, b, "generation must be deterministic");
        assert_eq!(a.len(), noise.frames);
        // Full-width and unbiased: both signs present, mean near zero, extremes
        // reach the wide range (no limited dynamic range). The mean bound is
        // generous (about 8 standard errors for this length) so it catches a
        // real offset, not ordinary sampling noise.
        let sum: i128 = a.iter().map(|&v| v as i128).sum();
        let mean = sum / a.len() as i128;
        assert!(
            mean.abs() < (1i128 << 25),
            "DC bias in a noise control: {mean}"
        );
        assert!(a.iter().any(|&v| v < 0) && a.iter().any(|&v| v > 0));
        let peak = a.iter().map(|&v| i64::from(v).abs()).max().unwrap();
        assert!(
            peak > (1i64 << 30),
            "noise control does not reach full width"
        );
        // No accidental short period: no p <= 1024 repeats the signal.
        for p in 1..=1024usize {
            if p >= a.len() {
                break;
            }
            let periodic = (p..a.len()).all(|i| a[i] == a[i - p]);
            assert!(!periodic, "noise control is periodic with period {p}");
        }
    }
}

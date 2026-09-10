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
//! * **orthogonal stratification** — source structure, amplitude occupancy,
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
    Amplitude, ChannelStructure, Entropy, RATES, Semantics, Signal, SourceStructure, Spec,
    Temporal, WaveShape, b1_comparable,
};

struct B {
    out: Vec<Spec>,
}

impl B {
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        stem: &str,
        source_structure: SourceStructure,
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
            source_structure,
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
    use Semantics::{Loop, OneShot};
    use SourceStructure::*;
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
    use Semantics::{Loop, OneShot};
    use SourceStructure::*;
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
        // B1 eligibility is derived from the format domain (channel count),
        // never read from a stored flag.
        if b1_comparable(s.channels) {
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

    /// The hostile incompressible stratum: full-width occupancy, random or
    /// scrambled character, and **genuinely independent channels**.
    ///
    /// An object that is temporally random but cross-channel structured (for
    /// example anticorrelated stereo, where `R = -L`) is a valuable control but
    /// is not *incompressible*, so it is deliberately excluded here.
    fn hostile_controls(s: &[Spec]) -> Vec<&Spec> {
        s.iter()
            .filter(|x| {
                matches!(x.entropy, Entropy::FullWidthRandom | Entropy::Scrambled)
                    && x.amplitude == Amplitude::Full
                    && matches!(
                        x.channel_structure,
                        ChannelStructure::Mono
                            | ChannelStructure::IndependentStereo
                            | ChannelStructure::Multichannel
                    )
            })
            .collect()
    }

    /// One channel's samples, deinterleaved from the canonical layout.
    fn channel(samples: &[i32], channels: usize, c: usize) -> Vec<i32> {
        samples.iter().skip(c).step_by(channels).copied().collect()
    }

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
            SourceStructure::Literal,
            SourceStructure::Oscillator,
            SourceStructure::Wavetable,
            SourceStructure::ExactRepetition,
            SourceStructure::Residual,
            SourceStructure::Compound,
            SourceStructure::Noise,
        ] {
            assert!(s.iter().any(|x| x.source_structure == rep), "no {rep:?}");
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
        let controls = hostile_controls(&s);
        assert!(
            controls.len() >= 8,
            "too few full-width independent-channel negative controls: {}",
            controls.len()
        );
        // Every hostile control must have genuinely independent channels: no
        // duplicated, correlated or mirrored channel is an incompressible one.
        for c in &controls {
            assert!(
                matches!(
                    c.channel_structure,
                    ChannelStructure::Mono
                        | ChannelStructure::IndependentStereo
                        | ChannelStructure::Multichannel
                ),
                "{}: a hostile control must have independent channels",
                c.id
            );
        }
        // Both uniform and scrambled hostile controls are present.
        assert!(
            controls
                .iter()
                .any(|c| c.entropy == Entropy::FullWidthRandom)
        );
        assert!(controls.iter().any(|c| c.entropy == Entropy::Scrambled));

        // The temporally-random, cross-channel-*structured* object is kept in
        // the frozen population (it is a valuable control) but must not be
        // counted as incompressible: `R = -L` is maximally predictable across
        // channels, which is exactly the H.2 `random-control` lesson.
        let anticorrelated = s
            .iter()
            .find(|x| {
                x.entropy == Entropy::FullWidthRandom
                    && x.amplitude == Amplitude::Full
                    && x.channel_structure == ChannelStructure::AnticorrelatedStereo
            })
            .expect("the anticorrelated full-width random control is frozen");
        assert!(
            !controls.iter().any(|c| c.id == anticorrelated.id),
            "{} is cross-channel structured and must not be a hostile control",
            anticorrelated.id
        );
    }

    #[test]
    fn generation_is_deterministic_and_unbiased() {
        let s = specs();
        let controls = hostile_controls(&s);
        assert!(!controls.is_empty());
        for c in &controls {
            let a = super::super::generate::generate(c).unwrap();
            let b = super::super::generate::generate(c).unwrap();
            assert_eq!(a, b, "{}: generation must be deterministic", c.id);
            let nch = usize::from(c.channels);
            assert_eq!(a.len(), c.frames * nch);

            let mut hashes: Vec<[u8; 32]> = Vec::with_capacity(nch);
            for ch in 0..nch {
                let v = channel(&a, nch, ch);
                // Full-width and unbiased, independently per channel. The sum
                // bound is ~8 standard errors for a uniform full-width stream,
                // so it catches a real offset rather than sampling noise.
                let sum: i128 = v.iter().map(|&x| x as i128).sum();
                let bound = 8 * (1i128 << 31) * (v.len() as u128).isqrt() as i128;
                assert!(
                    sum.abs() < bound,
                    "{} ch{ch}: DC bias (sum {sum}, bound {bound})",
                    c.id
                );
                assert!(
                    v.iter().any(|&x| x < 0) && v.iter().any(|&x| x > 0),
                    "{} ch{ch}: single sign",
                    c.id
                );
                let peak = v.iter().map(|&x| i64::from(x).abs()).max().unwrap();
                assert!(
                    peak > (1i64 << 30),
                    "{} ch{ch}: does not reach full width",
                    c.id
                );
                // No accidental short period (the `all` short-circuits on the
                // first mismatch, so this is cheap for real noise).
                for p in 1..=1024usize {
                    if p >= v.len() {
                        break;
                    }
                    assert!(
                        !(p..v.len()).all(|i| v[i] == v[i - p]),
                        "{} ch{ch}: periodic with period {p}",
                        c.id
                    );
                }
                hashes.push(super::super::generate::canonical_sha256(&v));
            }

            // Channels must be genuinely independent: no duplicate channel, and
            // no L == R or L == -R relationship.
            let mut sorted = hashes.clone();
            sorted.sort_unstable();
            let before = sorted.len();
            sorted.dedup();
            assert_eq!(before, sorted.len(), "{}: duplicate channel hash", c.id);
            if nch >= 2 {
                let l = channel(&a, nch, 0);
                let r = channel(&a, nch, 1);
                assert_ne!(l, r, "{}: L == R", c.id);
                assert!(
                    !l.iter().zip(&r).all(|(&x, &y)| y == x.saturating_neg()),
                    "{}: L == -R",
                    c.id
                );
            }
        }
    }
}

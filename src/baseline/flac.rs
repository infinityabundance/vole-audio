//! B1 — conventional lossless codec baseline: FLAC over the exact canonical
//! interleaved `i32` domain.
//!
//! Phase M hardens the historical H.2 comparator into a production baseline.
//! The H.2 court compares a **24-bit converted** source (an arithmetic `>> 8`
//! over the canonical codes) against an external `flac` executable, and says so;
//! that row stays frozen as historical evidence in
//! `courts::entropy_common`. B1 is a different, stronger thing:
//!
//! ```text
//! canonical VOLE i32 samples
//!         ├── VOLE representation     (priced by H.2 complete-cost)
//!         └── FLAC 32-bit             (this module)
//! ```
//!
//! Rules B1 must satisfy, frozen by review:
//!
//! * **exact i32 round trip** — `decode(encode(x)) == x` sample for sample, or
//!   the row fails correctness (never "close enough");
//! * **no conversion** — no s24 shift, no normalisation, no dithering, no
//!   resampling; the object's own channel count and sample rate;
//! * **32 bits/sample** — the deciding capability, because FLAC permits up to 32
//!   bits and the canonical observation is `i32`;
//! * **level 5 primary** — a single frozen conventional target, with levels 0
//!   and 8 recorded as secondary Pareto controls rather than chosen per result;
//! * **in-process, pure Rust** — B1 must not become unavailable because an
//!   external executable is not installed;
//! * **zero VOLE semantic authority** — the codec never decides anything about a
//!   `SampleObject`; it is a comparator.
//!
//! The encoder is `libflac-rs`, pinned exactly (`=0.143.1`), a `forbid(unsafe_code)`
//! pure-Rust port of libFLAC 1.4.3 whose output is byte-identical to the C
//! reference at matching settings. The pin lives in `Cargo.toml` so the encoder
//! cannot drift underneath a sealed receipt.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;

/// FLAC's own channel ceiling (the format, not this crate's `u1` limit of 32).
pub const FLAC_MAX_CHANNELS: u8 = 8;

/// Bits per sample of the B1 encoding: the full canonical `i32` width.
pub const B1_BITS_PER_SAMPLE: u32 = 32;

/// The frozen primary B1 preset. The official `flac` command-line tool defaults
/// to `-5`, and libFLAC documents level 5 as its default, so this is the
/// conventional target a reader would expect.
pub const B1_LEVEL_PRIMARY: u32 = 5;

/// Secondary Pareto controls: a speed-biased bound and a size-biased bound.
/// Recorded for shape, never substituted for the primary.
pub const B1_LEVEL_CONTROLS: [u32; 2] = [0, 8];

/// Evidence label for a compression level (`flac-0`, `flac-5`, `flac-8`).
pub fn b1_level_label(level: u32) -> String {
    format!("flac-{level}")
}

/// One exactly-verified B1 encoding of a window.
#[derive(Debug, Clone)]
pub struct FlacEncoding {
    /// Compression preset used (0–8).
    pub level: u32,
    pub channels: u8,
    pub bits_per_sample: u32,
    pub sample_rate: u32,
    pub source_frames: u64,
    /// Uncompressed canonical bytes fed to the encoder (4 per sample).
    pub source_bytes: u64,
    /// Complete `.flac` stream bytes (marker + metadata + frames).
    pub encoded_bytes: u64,
    /// Measured encode wall time (ns).
    pub encode_ns: u64,
    /// Measured decode wall time (ns).
    pub decode_ns: u64,
    /// Canonical observation hash of the source samples.
    pub source_sha256: [u8; 32],
    /// Canonical observation hash of the decoded samples (must equal the above).
    pub decoded_sha256: [u8; 32],
    /// `true` only when the decoded samples equal the source sample for sample.
    pub exact_roundtrip: bool,
    /// STREAMINFO audio MD5 verified by the decoder (independent integrity
    /// signal on top of the sample equality check).
    pub md5_ok: bool,
}

impl FlacEncoding {
    /// Compressed / uncompressed ratio (`1.0` = incompressible).
    pub fn ratio_vs_source(&self) -> f64 {
        self.encoded_bytes as f64 / self.source_bytes.max(1) as f64
    }
}

/// One exactly-verified B1 encoding of a window, **with its encoded bytes**.
///
/// B1 and the B4 runtime source both consume this exact artifact, so they can
/// never construct subtly different FLAC files.
#[derive(Debug, Clone)]
pub struct FlacArtifact {
    /// The complete `.flac` stream (marker + metadata + frames).
    pub bytes: Vec<u8>,
    /// The verified encoding metadata (level, times, hashes, exactness).
    pub encoding: FlacEncoding,
    /// SHA-256 of `bytes` (the artifact identity recorded in receipts).
    pub sha256: [u8; 32],
}

/// Encode `samples` (exact canonical interleaved i32) at `level` and require an
/// exact round trip, returning the verified bytes.
///
/// Returns an error — never a silently-degraded row — when the source is
/// outside FLAC's domain (0 or >8 channels, non-frame-aligned, empty, absurd
/// rate) or when the decoder does not reproduce the source exactly.
pub fn b1_flac_artifact(
    samples: &[i32],
    channels: u8,
    sample_rate: u32,
    level: u32,
) -> Result<FlacArtifact> {
    if channels == 0 || channels > FLAC_MAX_CHANNELS {
        return Err(Error::malformed(format!(
            "B1/FLAC supports 1..={FLAC_MAX_CHANNELS} channels (got {channels})"
        )));
    }
    if sample_rate == 0 || sample_rate > crate::limits::MAX_SAMPLE_RATE_HZ {
        return Err(Error::malformed("B1/FLAC sample rate out of domain"));
    }
    if level > 8 {
        return Err(Error::malformed(
            "B1/FLAC compression level out of domain (0..=8)",
        ));
    }
    let ch = usize::from(channels);
    if samples.is_empty() || !samples.len().is_multiple_of(ch) {
        return Err(Error::malformed(
            "B1/FLAC source must be a non-empty frame-aligned window",
        ));
    }

    // `EncoderConfig::new` defaults to level 8; the level is always set
    // explicitly so the baseline can never inherit a library default.
    let config =
        libflac_rs::EncoderConfig::new(u32::from(channels), B1_BITS_PER_SAMPLE, sample_rate)
            .with_compression_level(level);
    let encoder = libflac_rs::Encoder::new(config);

    let sw = Stopwatch::start();
    let encoded = encoder.encode(samples);
    let encode_ns = sw.elapsed_ns().max(0) as u64;
    let encoded_bytes = encoded.len();

    let sw = Stopwatch::start();
    let decoded = libflac_rs::decode(&encoded).ok_or_else(|| {
        Error::internal("B1/FLAC: the encoder produced a stream the decoder rejects")
    })?;
    let decode_ns = sw.elapsed_ns().max(0) as u64;

    if decoded.channels != u32::from(channels)
        || decoded.bits_per_sample != B1_BITS_PER_SAMPLE
        || decoded.sample_rate != sample_rate
    {
        return Err(Error::internal(format!(
            "B1/FLAC round trip changed the stream format: {}ch/{}bit/{}Hz -> {}ch/{}bit/{}Hz",
            channels,
            B1_BITS_PER_SAMPLE,
            sample_rate,
            decoded.channels,
            decoded.bits_per_sample,
            decoded.sample_rate
        )));
    }

    let source_sha256 = crate::universe::observation::observation_sha256(samples);
    let decoded_sha256 = crate::universe::observation::observation_sha256(&decoded.interleaved);
    // The integrity check is part of what a B1 encoding *is*, so it is enforced
    // here rather than left to callers: no B1 result can exist without a verified
    // STREAMINFO audio MD5, at any compression level.
    if !decoded.md5_ok {
        return Err(Error::internal(
            "B1/FLAC STREAMINFO audio MD5 did not verify",
        ));
    }
    let exact_roundtrip = decoded.interleaved == samples;
    if !exact_roundtrip {
        let first = samples
            .iter()
            .zip(&decoded.interleaved)
            .position(|(a, b)| a != b)
            .unwrap_or(samples.len().min(decoded.interleaved.len()));
        return Err(Error::internal(format!(
            "B1/FLAC is not exact: first mismatch at sample {first} \
             ({} source samples, {} decoded)",
            samples.len(),
            decoded.interleaved.len()
        )));
    }

    Ok(FlacArtifact {
        sha256: crate::hash::sha256::Sha256::digest(&encoded),
        bytes: encoded,
        encoding: FlacEncoding {
            level,
            channels,
            bits_per_sample: B1_BITS_PER_SAMPLE,
            sample_rate,
            source_frames: (samples.len() / ch) as u64,
            source_bytes: samples.len() as u64 * 4,
            encoded_bytes: encoded_bytes as u64,
            encode_ns,
            decode_ns,
            source_sha256,
            decoded_sha256,
            exact_roundtrip,
            md5_ok: decoded.md5_ok,
        },
    })
}

/// Encode and verify, returning only the encoding metadata (compatibility
/// wrapper; the artifact form is [`b1_flac_artifact`]).
pub fn b1_flac(
    samples: &[i32],
    channels: u8,
    sample_rate: u32,
    level: u32,
) -> Result<FlacEncoding> {
    Ok(b1_flac_artifact(samples, channels, sample_rate, level)?.encoding)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic splitmix64 fixture stream (fixtures only; never the
    /// semantic universe PRNG).
    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[test]
    fn extremes_round_trip_exactly_at_every_level() {
        // The full i32 domain: the values a 24-bit conversion would destroy.
        let mono: Vec<i32> = vec![
            i32::MIN,
            i32::MAX,
            -1,
            0,
            1,
            i32::MIN + 1,
            i32::MAX - 1,
            0x8000_0000u32 as i32,
            0x7fff_ffff,
        ];
        for level in [0u32, 5, 8] {
            let e = b1_flac(&mono, 1, 48_000, level).expect("extremes encode");
            assert!(e.exact_roundtrip, "level {level}");
            assert!(e.md5_ok, "level {level}: STREAMINFO MD5");
            assert_eq!(e.source_sha256, e.decoded_sha256, "level {level}");
            assert_eq!(e.bits_per_sample, 32);
        }
    }

    #[test]
    fn low_byte_only_differences_survive() {
        // Guards against anyone resurrecting the H.2 `>> 8` s24 path: two
        // signals that differ ONLY in their low bytes must both round-trip and
        // must remain different.
        let mut state = 0x1234_5678_9abc_def0u64;
        let full: Vec<i32> = (0..2048).map(|_| splitmix(&mut state) as i32).collect();
        let stripped: Vec<i32> = full.iter().map(|v| v & !0xFF).collect();
        assert_ne!(full, stripped, "fixture must exercise low bytes");

        for (label, src) in [("full", &full), ("low-byte-zero", &stripped)] {
            let e = b1_flac(src, 1, 48_000, B1_LEVEL_PRIMARY).expect(label);
            assert!(e.exact_roundtrip, "{label}");
            assert_eq!(e.source_sha256, e.decoded_sha256, "{label}");
        }
        let a = b1_flac(&full, 1, 48_000, B1_LEVEL_PRIMARY).unwrap();
        let b = b1_flac(&stripped, 1, 48_000, B1_LEVEL_PRIMARY).unwrap();
        assert_ne!(
            a.decoded_sha256, b.decoded_sha256,
            "low-byte-only differences must change the decoded identity"
        );
    }

    #[test]
    fn stereo_and_several_rates_round_trip() {
        let mut state = 0xdead_beef_cafe_babeu64;
        for channels in [1u8, 2] {
            for rate in [44_100u32, 48_000, 96_000, 192_000] {
                for frames in [1usize, 7, 4096, 4097] {
                    let n = frames * usize::from(channels);
                    let pcm: Vec<i32> =
                        (0..n).map(|_| (splitmix(&mut state) as i32) >> 3).collect();
                    let e = b1_flac(&pcm, channels, rate, B1_LEVEL_PRIMARY)
                        .unwrap_or_else(|err| panic!("{channels}ch/{rate}Hz/{frames}f: {err}"));
                    assert!(e.exact_roundtrip);
                    assert_eq!(e.channels, channels);
                    assert_eq!(e.sample_rate, rate);
                    assert_eq!(e.source_frames, frames as u64);
                }
            }
        }
    }

    #[test]
    fn low_entropy_content_compresses_and_noise_does_not() {
        // Long enough that FLAC's per-block framing overhead is amortised.
        let silence = vec![0i32; 65_536];
        let e = b1_flac(&silence, 1, 48_000, B1_LEVEL_PRIMARY).unwrap();
        assert!(
            e.ratio_vs_source() < 0.05,
            "silence ratio {}",
            e.ratio_vs_source()
        );

        let mut state = 0x0123_4567_89ab_cdefu64;
        let noise: Vec<i32> = (0..65_536).map(|_| splitmix(&mut state) as i32).collect();
        let n = b1_flac(&noise, 1, 48_000, B1_LEVEL_PRIMARY).unwrap();
        assert!(
            n.ratio_vs_source() > 0.95,
            "32-bit noise should be near-incompressible, got {}",
            n.ratio_vs_source()
        );
    }

    #[test]
    fn malformed_requests_are_rejected() {
        assert!(b1_flac(&[0, 1], 0, 48_000, 5).is_err(), "zero channels");
        assert!(b1_flac(&[0, 1], 9, 48_000, 5).is_err(), "nine channels");
        assert!(b1_flac(&[], 1, 48_000, 5).is_err(), "empty");
        assert!(
            b1_flac(&[0, 1, 2], 2, 48_000, 5).is_err(),
            "not frame aligned"
        );
        assert!(b1_flac(&[0, 1], 1, 0, 5).is_err(), "zero rate");
        assert!(
            b1_flac(&[0, 1], 1, 48_000, 9).is_err(),
            "level out of range"
        );
    }

    #[test]
    fn level_controls_are_monotone_in_the_expected_direction() {
        // Not a hard guarantee (it is a heuristic codec), but on structured
        // content level 8 must not be larger than level 0. A failure here means
        // the preset wiring is wrong, not that the codec is bad.
        let mut state = 0xabcd_ef01_2345_6789u64;
        let pcm: Vec<i32> = (0..16384)
            .map(|_| ((splitmix(&mut state) as i32) >> 6) % 4096)
            .collect();
        let l0 = b1_flac(&pcm, 1, 48_000, B1_LEVEL_CONTROLS[0]).unwrap();
        let l8 = b1_flac(&pcm, 1, 48_000, B1_LEVEL_CONTROLS[1]).unwrap();
        assert!(l0.exact_roundtrip && l8.exact_roundtrip);
        assert!(
            l8.encoded_bytes <= l0.encoded_bytes,
            "level 8 ({}) must not exceed level 0 ({}) on structured content",
            l8.encoded_bytes,
            l0.encoded_bytes
        );
    }
}

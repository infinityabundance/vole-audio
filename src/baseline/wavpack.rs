//! WavPack external baseline (Exp2 Seal L).
//!
//! A **stronger external control** than FLAC for the lossless comparison. It is
//! an *external* row with zero VOLE semantic authority: the `wavpack` /
//! `wvunpack` command-line tools are invoked, and every row is verified to
//! round-trip the exact canonical `i32` samples before its size is recorded.
//!
//! The baseline is honest about availability: when the tools are absent the row
//! is `Unavailable`, never a silent substitute.

use crate::error::{Error, Kind, Result};
use std::path::PathBuf;
use std::process::Command;

/// WavPack preset candidates (label, extra flags). Presets are canonical and
/// fixed; the default row and the high/asymmetric rows are all reported.
pub const WAVPACK_PRESETS: [(&str, &str); 4] = [
    ("wavpack-default", ""),
    ("wavpack-high", "-h"),
    ("wavpack-veryhigh", "-hh"),
    ("wavpack-max", "-hh -x6"),
];

/// Maximum channels WavPack's WAV path accepts in this baseline.
pub const WAVPACK_MAX_CHANNELS: u8 = 8;

/// One measured WavPack row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavPackEncoding {
    pub preset: String,
    pub channels: u8,
    pub bits_per_sample: u32,
    pub sample_rate: u32,
    pub source_frames: u64,
    /// Uncompressed canonical bytes fed to the encoder (4 per sample).
    pub source_bytes: u64,
    /// Complete `.wv` stream bytes.
    pub encoded_bytes: u64,
    /// Whether `wvunpack` reproduced the exact canonical samples.
    pub exact: bool,
}

impl WavPackEncoding {
    /// Compressed / uncompressed ratio (`1.0` = incompressible).
    pub fn ratio_vs_source(&self) -> f64 {
        self.encoded_bytes as f64 / self.source_bytes.max(1) as f64
    }
}

/// Whether the external tools are present on this host.
pub fn available() -> bool {
    Command::new("wavpack")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
        && Command::new("wvunpack")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

fn canonical_wav(samples: &[i32], channels: u8, sample_rate: u32) -> Vec<u8> {
    let c = usize::from(channels);
    let frames = (samples.len() / c.max(1)) as u32;
    let data_len = (samples.len() * 4) as u32;
    let block_align = (c * 4) as u16;
    let byte_rate = sample_rate * u32::from(block_align);
    let mut out = Vec::with_capacity(44 + samples.len() * 4);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&(c as u16).to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    let _ = frames;
    out
}

fn parse_wav_samples(bytes: &[u8]) -> Result<Vec<i32>> {
    // Minimal RIFF/WAVE parser: locate `fmt ` for channels and `data`.
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::malformed(
            "wavpack round-trip is not a RIFF/WAVE file",
        ));
    }
    let mut pos = 12usize;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = bytes
            .get(pos + 8..pos + 8 + len)
            .ok_or_else(|| Error::malformed("wavpack round-trip chunk is truncated"))?;
        if id == b"data" {
            data = Some(body);
            break;
        }
        pos = pos + 8 + len + (len & 1);
    }
    let data = data.ok_or_else(|| Error::malformed("wavpack round-trip has no data chunk"))?;
    if !data.len().is_multiple_of(4) {
        return Err(Error::malformed(
            "wavpack round-trip data is not 32-bit aligned",
        ));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    let mut i = 0usize;
    while i + 4 <= data.len() {
        out.push(i32::from_le_bytes(data[i..i + 4].try_into().unwrap()));
        i += 4;
    }
    Ok(out)
}

struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new() -> Result<Scratch> {
        let base = std::env::temp_dir().join(format!(
            "vole-wavpack-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&base)
            .map_err(|e| Error::new(Kind::Io, format!("cannot create wavpack scratch dir: {e}")))?;
        Ok(Scratch { dir: base })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Encode and verify `samples` with one WavPack preset.
///
/// Returns `Ok(None)` when the external tools are unavailable (an honest
/// `Unavailable` row), and `Err` when the tool ran but the round trip was not
/// exact.
pub fn wavpack_bytes(
    samples: &[i32],
    channels: u8,
    sample_rate: u32,
    preset_index: usize,
) -> Result<Option<WavPackEncoding>> {
    if !available() {
        return Ok(None);
    }
    if channels == 0 || channels > WAVPACK_MAX_CHANNELS {
        return Err(Error::malformed("wavpack channel count out of range"));
    }
    if !samples.len().is_multiple_of(usize::from(channels)) {
        return Err(Error::malformed(
            "wavpack sample count is not channel-aligned",
        ));
    }
    let (label, flags) = WAVPACK_PRESETS
        .get(preset_index)
        .copied()
        .ok_or_else(|| Error::malformed("unknown wavpack preset index"))?;
    let scratch = Scratch::new()?;
    let wav = scratch.dir.join("in.wav");
    let wv = scratch.dir.join("out.wv");
    let back = scratch.dir.join("back.wav");
    std::fs::write(&wav, canonical_wav(samples, channels, sample_rate))
        .map_err(|e| Error::new(Kind::Io, format!("cannot write wavpack input: {e}")))?;
    let mut cmd = Command::new("wavpack");
    cmd.arg("-y").arg("-q");
    for f in flags.split_whitespace() {
        cmd.arg(f);
    }
    cmd.arg(&wav).arg("-o").arg(&wv);
    let enc = cmd
        .output()
        .map_err(|e| Error::new(Kind::Io, format!("cannot run wavpack: {e}")))?;
    if !enc.status.success() {
        return Err(Error::new(
            Kind::Unsupported,
            format!("wavpack failed: {}", String::from_utf8_lossy(&enc.stderr)),
        ));
    }
    let encoded_bytes = std::fs::metadata(&wv)
        .map_err(|e| Error::new(Kind::Io, format!("cannot stat wavpack output: {e}")))?
        .len();
    // Verify exactness with wvunpack.
    let dec = Command::new("wvunpack")
        .arg("-y")
        .arg("-q")
        .arg(&wv)
        .arg("-o")
        .arg(&back)
        .output()
        .map_err(|e| Error::new(Kind::Io, format!("cannot run wvunpack: {e}")))?;
    if !dec.status.success() {
        return Err(Error::new(
            Kind::Unsupported,
            format!("wvunpack failed: {}", String::from_utf8_lossy(&dec.stderr)),
        ));
    }
    let decoded = std::fs::read(&back)
        .map_err(|e| Error::new(Kind::Io, format!("cannot read wavpack output: {e}")))?;
    let round = parse_wav_samples(&decoded)?;
    let exact = round == samples;
    if !exact {
        return Err(Error::new(
            Kind::Integrity,
            "wavpack round trip did not reproduce the canonical samples",
        ));
    }
    Ok(Some(WavPackEncoding {
        preset: label.to_string(),
        channels,
        bits_per_sample: 32,
        sample_rate,
        source_frames: (samples.len() / usize::from(channels)) as u64,
        source_bytes: samples.len() as u64 * 4,
        encoded_bytes,
        exact,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trips_through_the_local_parser() {
        let samples: Vec<i32> = (0..1000).map(|i| (i * 37) - 500).collect();
        let wav = canonical_wav(&samples, 1, 48_000);
        assert_eq!(parse_wav_samples(&wav).unwrap(), samples);
    }

    #[test]
    fn wavpack_presets_round_trip_exactly_when_available() {
        if !available() {
            eprintln!("wavpack/wvunpack unavailable; baseline recorded as Unavailable");
            return;
        }
        let samples: Vec<i32> = (0..4096)
            .map(|i| ((i as i64 * 2654435761) % 100_003) as i32 - 50_000)
            .collect();
        for idx in 0..WAVPACK_PRESETS.len() {
            let e = wavpack_bytes(&samples, 1, 48_000, idx)
                .unwrap()
                .expect("available");
            assert!(e.exact);
            assert!(e.encoded_bytes > 0);
            assert!(e.encoded_bytes < samples.len() as u64 * 4);
        }
    }
}

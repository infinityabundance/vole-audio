//! Shared helpers for the entropy courts (host-only).
//!
//! Everything here is court instrumentation: conversions for conventional
//! baselines, canonical-byte accounting, and small deterministic helpers.
//! Nothing in this module has semantic authority.

use crate::error::Result;
use std::path::PathBuf;
use std::process::Command;

/// Canonical U1 literal bytes for a descriptor + sample count
/// (header 46 + count prefix 8 + 4 bytes/sample).
pub fn canonical_u1_literal_bytes(extent_frames: u64, channels: u8) -> u64 {
    46 + 8 + extent_frames * u64::from(channels) * 4
}

/// s24 conversion used for the FLAC baseline: arithmetic shift (documented;
/// the baseline compares sizes only, and the same converted bytes feed the
/// pinned encoder every run).
pub fn to_s24(code: i32) -> i32 {
    code >> 8
}

/// A measured conventional baseline.
pub struct Baseline {
    /// Tool identity/command (exact, pinned).
    pub command: String,
    /// Version string of the tool.
    pub version: String,
    /// Compressed bytes.
    pub bytes: u64,
    /// Sample bit depth of the encoding (24 for FLAC s24).
    pub sample_bps: u32,
    /// Source bytes fed to the encoder (WAV payload at the sample depth).
    pub source_payload_bytes: u64,
}

/// Run the pinned `flac` executable over the samples (s24 conversion,
/// documented) if present; `Ok(None)` when the tool is unavailable
/// (`NOT_AVAILABLE`, never invented numbers).
pub fn flac_baseline(samples: &[i32], channels: u8, sample_rate: u32) -> Result<Option<Baseline>> {
    let Some(flac) = find_tool("flac")? else {
        return Ok(None);
    };
    let version = flac_version(&flac)?;
    let dir = std::env::temp_dir().join(format!(
        "vole-flac-{}-{}",
        std::process::id(),
        crate::evidence::timing::monotonic_raw_ns()
    ));
    std::fs::create_dir_all(&dir)?;
    let wav_path = dir.join("baseline.wav");
    let flac_path = dir.join("baseline.flac");
    let out = write_s24_wav(&wav_path, samples, channels, sample_rate)?;
    let status = Command::new(&flac)
        .args([
            "-f",
            "-o",
            flac_path.to_str().unwrap(),
            wav_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| crate::error::Error::external(format!("flac run: {e}")))?;
    if !status.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Ok(None);
    }
    let bytes = std::fs::metadata(&flac_path)?.len();
    let _ = std::fs::remove_dir_all(&dir);
    Ok(Some(Baseline {
        command: format!("{} -f -o baseline.flac baseline.wav", flac.display()),
        version,
        bytes,
        sample_bps: 24,
        source_payload_bytes: out,
    }))
}

/// Find a tool on PATH.
fn find_tool(name: &str) -> Result<Option<PathBuf>> {
    let probe = Command::new(name).arg("--version").output();
    match probe {
        Ok(o) if o.status.success() => Ok(Some(PathBuf::from(name))),
        _ => Ok(None),
    }
}

fn flac_version(flac: &PathBuf) -> Result<String> {
    let out = Command::new(flac).arg("--version").output()?;
    let s = String::from_utf8_lossy(&out.stdout).into_owned();
    Ok(s.lines().next().unwrap_or("flac").trim().to_string())
}

/// Write a canonical WAV (s24 in s32 container, documented conversion) and
/// return the payload byte count.
fn write_s24_wav(path: &std::path::Path, samples: &[i32], channels: u8, rate: u32) -> Result<u64> {
    use std::io::Write;
    let data_bytes = (samples.len() as u64) * 3;
    let mut f = std::fs::File::create(path)?;
    // RIFF header.
    f.write_all(b"RIFF")?;
    f.write_all(&((36 + data_bytes) as u32).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    // fmt chunk (PCM, s24 stored in 3-byte containers).
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&u16::from(channels).to_le_bytes())?;
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(&(rate * u32::from(channels) * 3).to_le_bytes())?; // byte rate
    f.write_all(&(u16::from(channels) * 3).to_le_bytes())?; // block align
    f.write_all(&24u16.to_le_bytes())?; // bits per sample
    // data chunk: 3-byte little-endian per channel sample.
    f.write_all(b"data")?;
    f.write_all(&(data_bytes as u32).to_le_bytes())?;
    for &s in samples {
        let v = to_s24(s);
        let b = v.to_le_bytes();
        f.write_all(&[b[0], b[1], b[2]])?;
    }
    f.sync_all()?;
    Ok(data_bytes)
}

/// Layout of a fixture for receipt params.
pub fn layout_name(channels: u8) -> &'static str {
    match channels {
        1 => "mono",
        2 => "stereo",
        _ => "channels",
    }
}

/// Mean of a slice of u64 nanoseconds (f64).
pub fn mean_ns(v: &[u64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<u64>() as f64 / v.len() as f64
}

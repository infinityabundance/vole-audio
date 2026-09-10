//! B1 reference oracle — the system `flac` executable, when present.
//!
//! B1 itself is the in-process pure-Rust baseline (`baseline::flac`), which must
//! never become unavailable. This module is a **non-authoritative oracle**: it
//! encodes the same exact i32 domain with the reference C encoder at identical
//! settings (32 bits/sample, level 5, no padding) and checks its own round trip,
//! so a divergence between the ported encoder and a modern reference is visible
//! as evidence rather than hidden.
//!
//! It exists because that divergence is real and material. `libflac-rs` ports
//! libFLAC **1.4.3**, whose constant-signal detection keys off a
//! `fixed_residual_bits_per_sample[1] == 0` test that the guess predictor only
//! produces below 28 bits/sample; at `subframe_bps >= 28` libFLAC 1.4.3 does
//! *not* select the CONSTANT subframe, so an all-zero 32-bit block costs about
//! one bit per sample. Newer reference encoders (1.5.x) select CONSTANT there and
//! can encode the same block in a few dozen bytes. B1's numbers therefore carry
//! the 1.4.3 semantics, and this row records where a modern reference differs.
//!
//! The oracle never enters the frozen static result: its bytes depend on which
//! reference encoder happens to be installed, so it cannot be a deterministic
//! cross-host vector. Its absence is `NOT_AVAILABLE`, never a failure.

use crate::error::Result;
use crate::evidence::timing::Stopwatch;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One reference-encoder measurement of the exact i32 domain.
#[derive(Debug, Clone)]
pub struct ReferenceFlac {
    /// Tool identity and exact arguments used (pinned).
    pub command: String,
    pub version: String,
    /// Complete `.flac` bytes with `--no-padding` (comparable to the in-memory
    /// baseline stream, which never carries a padding block).
    pub bytes: u64,
    /// Measured reference decode wall time (ns).
    pub decode_ns: u64,
    /// `true` only when the reference decoder reproduced the exact source bytes.
    pub exact_roundtrip: bool,
}

/// Encode and decode `samples` with the system `flac`, if it is installed.
///
/// Returns `Ok(None)` when the tool is absent or refuses the invocation
/// (`NOT_AVAILABLE`); a reference that runs but does not round-trip exactly is
/// reported with `exact_roundtrip: false` rather than swallowed.
pub fn reference_flac(
    samples: &[i32],
    channels: u8,
    sample_rate: u32,
    level: u32,
) -> Result<Option<ReferenceFlac>> {
    let Some(tool) = find_tool("flac") else {
        return Ok(None);
    };
    let version = tool_version(&tool).unwrap_or_else(|| "flac".to_string());
    let dir = scratch_dir();
    std::fs::create_dir_all(&dir)?;
    let raw = dir.join("b1.raw");
    let enc = dir.join("b1.flac");
    let dec = dir.join("b1.decoded.raw");

    let mut raw_bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        raw_bytes.extend_from_slice(&s.to_le_bytes());
    }
    {
        let mut f = std::fs::File::create(&raw)?;
        f.write_all(&raw_bytes)?;
        f.sync_all()?;
    }

    let ch = u16::from(channels);
    let status = Command::new(&tool)
        .args([
            format!("-{level}"),
            "--no-padding".to_string(),
            "-f".to_string(),
            "--force-raw-format".to_string(),
            "--endian=little".to_string(),
            "--sign=signed".to_string(),
            format!("--channels={ch}"),
            "--bps=32".to_string(),
            format!("--sample-rate={sample_rate}"),
            "-o".to_string(),
            enc.display().to_string(),
            raw.display().to_string(),
        ])
        .output();
    let Ok(out) = status else {
        let _ = std::fs::remove_dir_all(&dir);
        return Ok(None);
    };
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Ok(None);
    }
    let bytes = std::fs::metadata(&enc)?.len();
    let command = format!(
        "flac -{level} --no-padding --force-raw-format --endian=little --sign=signed \
         --channels={ch} --bps=32 --sample-rate={sample_rate}"
    );

    let sw = Stopwatch::start();
    let decoded = Command::new(&tool)
        .args([
            "-d".to_string(),
            "-f".to_string(),
            "--force-raw-format".to_string(),
            "--endian=little".to_string(),
            "--sign=signed".to_string(),
            "-o".to_string(),
            dec.display().to_string(),
            enc.display().to_string(),
        ])
        .output();
    let decode_ns = sw.elapsed_ns().max(0) as u64;
    let exact_roundtrip = match decoded {
        Ok(o) if o.status.success() => std::fs::read(&dec).map(|b| b == raw_bytes).unwrap_or(false),
        _ => false,
    };
    let _ = std::fs::remove_dir_all(&dir);

    Ok(Some(ReferenceFlac {
        command,
        version,
        bytes,
        decode_ns,
        exact_roundtrip,
    }))
}

fn scratch_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "vole-b1-ref-{}-{}",
        std::process::id(),
        crate::evidence::timing::monotonic_raw_ns()
    ))
}

fn find_tool(name: &str) -> Option<PathBuf> {
    Command::new(name)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| PathBuf::from(name))
}

fn tool_version(tool: &Path) -> Option<String> {
    let out = Command::new(tool).arg("--version").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next().map(|l| l.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oracle is optional by design: an absent tool is `NOT_AVAILABLE`, not a
    /// failure. When it *is* present it must round-trip the exact i32 domain — an
    /// oracle that is not exact would be worse than no oracle.
    #[test]
    fn reference_round_trips_exactly_when_present() {
        let pcm: Vec<i32> = (0..2048i32).map(|i| i * 12_345 - 7).collect();
        let Some(r) = reference_flac(&pcm, 1, 48_000, 5).expect("oracle run") else {
            return; // flac not installed: nothing to assert
        };
        assert!(r.bytes > 0, "reference produced an empty stream");
        assert!(
            r.exact_roundtrip,
            "reference flac did not reproduce the exact source bytes"
        );
        assert!(
            r.command.contains("--bps=32"),
            "must request 32 bits/sample"
        );
    }

    #[test]
    fn reference_and_in_process_agree_on_incompressible_content() {
        let mut state = 0x51ce_0b3f_9a2d_7744u64;
        let pcm: Vec<i32> = (0..4096)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z ^ (z >> 31)
            } as i32)
            .collect();
        let Some(r) = reference_flac(&pcm, 1, 48_000, 5).expect("oracle run") else {
            return;
        };
        assert!(r.exact_roundtrip);
        let b1 = crate::baseline::b1_flac(&pcm, 1, 48_000, 5).expect("b1");
        // Full-range noise is incompressible for both: the sizes must be close.
        let ratio = r.bytes as f64 / b1.encoded_bytes as f64;
        assert!(
            (0.9..1.1).contains(&ratio),
            "expected near-equal sizes on incompressible content, got {ratio} \
             (reference {}, B1 {})",
            r.bytes,
            b1.encoded_bytes
        );
    }
}

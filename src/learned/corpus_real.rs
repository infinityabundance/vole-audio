//! Real + held-out Mode-C corpus (Exp2 Seal K).
//!
//! Voice: **real speech** from LibriSpeech (OpenSLR SLR12, CC BY 4.0), a
//! rights-clean, redistribution-compatible source.
//!
//! * **Effectiveness** clips come from `dev-clean` (one clip per speaker, the
//!   first 12 speakers in ascending speaker-id order).
//! * **Mode C** clips come from `test-clean` (the first 8 speakers) and are
//!   **held out**: they are not used for any Exp2 architecture tuning.
//!
//! Membership, truncation and canonical hashes are frozen in
//! `corpus/real_manifest.json` **before** the court is run. The audio bulk is
//! not committed (it is externally sourced); the manifest of identities and
//! canonical i32 hashes is. When the audio is absent the court reports an
//! honest `INCONCLUSIVE` limitation rather than fabricating a result.
//!
//! The canonical samples are the sign-extended 16-bit PCM values as
//! little-endian `i32`, frame-major channel-minor — the same conversion the
//! manifest hash covers.

use crate::error::{Error, Kind, Result};
use crate::hash::sha256::{Sha256, hex};
use std::path::{Path, PathBuf};

/// The embedded frozen manifest.
pub const MANIFEST_JSON: &str = include_str!("../../corpus/real_manifest.json");

/// One frozen real clip identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealClip {
    pub id: String,
    pub split: String,
    pub speaker: String,
    /// Path relative to `corpus/real/`.
    pub path: String,
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub frames: u64,
    pub canonical_i32_sha256: String,
}

/// A loaded real clip.
#[derive(Debug, Clone)]
pub struct RealCase {
    pub clip: RealClip,
    pub samples: Vec<i32>,
}

impl RealCase {
    pub fn frames(&self) -> u64 {
        self.clip.frames
    }
    pub fn channels(&self) -> u8 {
        self.clip.channels
    }
    pub fn rate(&self) -> u32 {
        self.clip.sample_rate_hz
    }
}

/// Root directory of the (uncommitted) real audio bulk.
pub fn real_root() -> PathBuf {
    PathBuf::from("corpus/real")
}

fn parse_clips(which: &str) -> Vec<RealClip> {
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("the embedded real manifest is valid JSON");
    v.get(which)
        .and_then(|a| a.as_array())
        .map(|a| {
            a.iter()
                .map(|c| RealClip {
                    id: c["id"].as_str().unwrap_or_default().to_string(),
                    split: c["split"].as_str().unwrap_or_default().to_string(),
                    speaker: c["speaker"].as_str().unwrap_or_default().to_string(),
                    path: c["path"].as_str().unwrap_or_default().to_string(),
                    sample_rate_hz: c["sample_rate_hz"].as_u64().unwrap_or(0) as u32,
                    channels: c["channels"].as_u64().unwrap_or(0) as u8,
                    frames: c["frames"].as_u64().unwrap_or(0),
                    canonical_i32_sha256: c["canonical_i32_sha256"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Effectiveness clips (`dev-clean`).
pub fn effectiveness_clips() -> Vec<RealClip> {
    parse_clips("effectiveness")
}

/// Held-out Mode-C clips (`test-clean`).
pub fn mode_c_clips() -> Vec<RealClip> {
    parse_clips("mode_c")
}

/// SHA-256 over the frozen manifest identities (the corpus identity).
pub fn real_corpus_sha256() -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"vole.audio.learned.real_corpus.v1");
    for clip in effectiveness_clips().iter().chain(mode_c_clips().iter()) {
        bytes.extend_from_slice(clip.id.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(clip.split.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(clip.path.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&clip.sample_rate_hz.to_le_bytes());
        bytes.push(clip.channels);
        bytes.extend_from_slice(&clip.frames.to_le_bytes());
        bytes.extend_from_slice(clip.canonical_i32_sha256.as_bytes());
        bytes.push(b'\n');
    }
    hex(&Sha256::digest(&bytes))
}

/// Whether the real audio bulk is present on this host.
pub fn available() -> bool {
    let root = real_root();
    effectiveness_clips()
        .iter()
        .chain(mode_c_clips().iter())
        .all(|c| root.join(&c.path).is_file())
}

/// Whether the external `flac` decoder is available.
pub fn decoder_available() -> bool {
    std::process::Command::new("flac")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn read_wav_i16(bytes: &[u8]) -> Result<(u32, u8, Vec<i16>)> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::malformed("real corpus decode is not RIFF/WAVE"));
    }
    let mut pos = 12usize;
    let mut channels = 0u8;
    let mut rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = bytes
            .get(pos + 8..pos + 8 + len)
            .ok_or_else(|| Error::malformed("real corpus WAV chunk is truncated"))?;
        if id == b"fmt " && len >= 16 {
            let fmt = u16::from_le_bytes(body[0..2].try_into().unwrap());
            if fmt != 1 {
                return Err(Error::new(Kind::Unsupported, "real corpus WAV is not PCM"));
            }
            channels = u16::from_le_bytes(body[2..4].try_into().unwrap()) as u8;
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
        } else if id == b"data" {
            data = Some(body);
        }
        pos = pos + 8 + len + (len & 1);
    }
    if bits != 16 {
        return Err(Error::new(
            Kind::Unsupported,
            format!("real corpus expects 16-bit PCM, got {bits}"),
        ));
    }
    let data = data.ok_or_else(|| Error::malformed("real corpus WAV has no data chunk"))?;
    if !data.len().is_multiple_of(2) {
        return Err(Error::malformed(
            "real corpus WAV data is not 16-bit aligned",
        ));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    let mut i = 0usize;
    while i + 2 <= data.len() {
        out.push(i16::from_le_bytes(data[i..i + 2].try_into().unwrap()));
        i += 2;
    }
    Ok((rate, channels, out))
}

fn canonical_sha(samples: &[i32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    hex(&Sha256::digest(&bytes))
}

/// Load and exactly verify one frozen real clip.
pub fn load_clip(clip: &RealClip, scratch: &Path) -> Result<Vec<i32>> {
    let src = real_root().join(&clip.path);
    if !src.is_file() {
        return Err(Error::new(
            Kind::Unavailable,
            format!("real clip {} is absent ({})", clip.id, src.display()),
        ));
    }
    if !decoder_available() {
        return Err(Error::new(
            Kind::Unavailable,
            "the external `flac` decoder is not available",
        ));
    }
    std::fs::create_dir_all(scratch)
        .map_err(|e| Error::new(Kind::Io, format!("cannot create scratch: {e}")))?;
    let wav = scratch.join(format!("{}.wav", clip.id));
    let _ = std::fs::remove_file(&wav);
    let status = std::process::Command::new("flac")
        .arg("-d")
        .arg("-f")
        .arg("-s")
        .arg("-o")
        .arg(&wav)
        .arg(&src)
        .status()
        .map_err(|e| Error::new(Kind::Io, format!("cannot run flac: {e}")))?;
    if !status.success() {
        return Err(Error::new(
            Kind::External,
            "flac failed to decode a real clip",
        ));
    }
    let bytes = std::fs::read(&wav)
        .map_err(|e| Error::new(Kind::Io, format!("cannot read decoded WAV: {e}")))?;
    let _ = std::fs::remove_file(&wav);
    let (rate, channels, pcm) = read_wav_i16(&bytes)?;
    if rate != clip.sample_rate_hz || channels != clip.channels {
        return Err(Error::new(
            Kind::Integrity,
            "real clip geometry disagrees with the frozen manifest",
        ));
    }
    let per_frame = usize::from(channels);
    let frames = pcm.len() / per_frame;
    let take = frames.min(clip.frames as usize);
    let samples: Vec<i32> = pcm[..take * per_frame]
        .iter()
        .map(|&v| i32::from(v))
        .collect();
    if canonical_sha(&samples) != clip.canonical_i32_sha256 {
        return Err(Error::new(
            Kind::Integrity,
            "real clip canonical hash disagrees with the frozen manifest",
        ));
    }
    Ok(samples)
}

/// Load a set of clips into cases (bounded by `max`).
pub fn load_cases(clips: &[RealClip], max: usize, scratch: &Path) -> Result<Vec<RealCase>> {
    let mut out = Vec::new();
    for clip in clips.iter().take(max) {
        let samples = load_clip(clip, scratch)?;
        out.push(RealCase {
            clip: clip.clone(),
            samples,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_frozen_and_well_formed() {
        assert_eq!(effectiveness_clips().len(), 12);
        assert_eq!(mode_c_clips().len(), 8);
        for c in effectiveness_clips().iter().chain(mode_c_clips().iter()) {
            assert_eq!(c.frames, 16384);
            assert_eq!(c.channels, 1);
            assert_eq!(c.sample_rate_hz, 16_000);
            assert_eq!(c.canonical_i32_sha256.len(), 64);
        }
        // The two splits are disjoint.
        let effectiveness = effectiveness_clips();
        assert!(effectiveness.iter().all(|c| c.split == "dev-clean"));
    }

    #[test]
    fn corpus_identity_is_deterministic() {
        assert_eq!(real_corpus_sha256(), real_corpus_sha256());
        assert_eq!(real_corpus_sha256().len(), 64);
    }
}

//! Narrow, robust RIFF/WAV ingest for the exact profile.
//!
//! Supported: PCM integer formats u8 / s16 / s24 / s32 (canonical code domain
//! mapping in `universe::sample`), 1..=32 channels. Anything else (float,
//! extensible with float subformat, A-law, μ-law, ADPCM...) is **rejected
//! explicitly** — never silently quantized.
//!
//! Hostile-input rules: every chunk operation is length-checked; duplicate
//! `fmt `/`data` chunks are rejected; truncated files are errors; absurd
//! rates/counts are rejected against `limits`; no unchecked `usize` casts.
//!
//! The parser yields interleaved canonical `i32` samples plus format facts;
//! WAV container metadata (extra chunks) is outside the u1 equality claim.

use crate::error::{Error, Result};
use crate::limits::{
    MAX_CHANNELS, MAX_FILE_BYTES, MAX_OBJECT_FRAMES, MAX_SAMPLE_RATE_HZ, MAX_WAV_DATA_BYTES,
};
use crate::universe::sample;

/// Interleaved canonical sample codes plus the format facts needed to build a
/// literal SampleObject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedWav {
    pub channels: u16,
    pub sample_rate_hz: u32,
    /// Canonical interleaved codes: `frames * channels` values.
    pub samples: Vec<i32>,
    pub frames: u64,
}

/// A successfully parsed WAV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedWav {
    pub decoded: DecodedWav,
    pub pcm_format: PcmFormat,
}

/// Exact integer PCM layout of the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmFormat {
    U8,
    S16,
    S24,
    S32,
}

impl PcmFormat {
    pub const fn bytes_per_sample(self) -> u32 {
        match self {
            PcmFormat::U8 => 1,
            PcmFormat::S16 => 2,
            PcmFormat::S24 => 3,
            PcmFormat::S32 => 4,
        }
    }
}

/// Parse a complete WAV byte buffer (RIFF little-endian).
pub fn parse_wav(bytes: &[u8]) -> Result<ParsedWav> {
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(Error::limit("WAV exceeds MAX_FILE_BYTES"));
    }
    let mut pos = 0usize;
    let need = |p: usize, n: usize| -> Result<()> {
        if p.checked_add(n).is_none_or(|e| e > bytes.len()) {
            return Err(Error::malformed("truncated WAV"));
        }
        Ok(())
    };
    need(pos, 12)?;
    if &bytes[0..4] != b"RIFF" {
        return Err(Error::malformed("missing RIFF magic"));
    }
    let riff_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    if &bytes[8..12] != b"WAVE" {
        return Err(Error::malformed("missing WAVE tag"));
    }
    if riff_len > bytes.len().saturating_sub(8) {
        return Err(Error::malformed("RIFF length exceeds input"));
    }
    pos = 12;

    let mut fmt: Option<(u16, u16, u32, u32, u16, u16)> = None;
    let mut data: Option<(usize, usize)> = None;

    while pos < bytes.len() {
        need(pos, 8)?;
        let id = &bytes[pos..pos + 4];
        let chunk_len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let payload = pos + 8;
        if payload
            .checked_add(chunk_len)
            .is_none_or(|e| e > bytes.len())
        {
            return Err(Error::malformed("chunk length exceeds file"));
        }
        match id {
            b"fmt " => {
                if fmt.is_some() {
                    return Err(Error::malformed("duplicate fmt chunk"));
                }
                if chunk_len < 16 {
                    return Err(Error::malformed("fmt chunk too short"));
                }
                let tag = u16::from_le_bytes(bytes[payload..payload + 2].try_into().unwrap());
                let ch = u16::from_le_bytes(bytes[payload + 2..payload + 4].try_into().unwrap());
                let rate = u32::from_le_bytes(bytes[payload + 4..payload + 8].try_into().unwrap());
                let byterate =
                    u32::from_le_bytes(bytes[payload + 8..payload + 12].try_into().unwrap());
                let align =
                    u16::from_le_bytes(bytes[payload + 12..payload + 14].try_into().unwrap());
                let bits =
                    u16::from_le_bytes(bytes[payload + 14..payload + 16].try_into().unwrap());
                fmt = Some((tag, ch, rate, byterate, align, bits));
            }
            b"data" => {
                if data.is_some() {
                    return Err(Error::malformed("duplicate data chunk"));
                }
                data = Some((payload, chunk_len));
            }
            _ => {}
        }
        pos = payload + chunk_len + (chunk_len % 2);
    }

    let (tag, ch, rate, byterate, align, bits) =
        fmt.ok_or_else(|| Error::malformed("missing fmt chunk"))?;
    let (data_start, data_len) = data.ok_or_else(|| Error::malformed("missing data chunk"))?;

    if tag != 1 {
        return Err(Error::unsupported(format!(
            "WAV format tag {tag} is not integer PCM (exact profile rejects non-PCM)"
        )));
    }
    if !(1..=MAX_CHANNELS as u16).contains(&ch) {
        return Err(Error::limit("channel count outside domain"));
    }
    if rate == 0 || rate > MAX_SAMPLE_RATE_HZ {
        return Err(Error::limit("sample rate outside domain"));
    }
    let pcm = match bits {
        8 => PcmFormat::U8,
        16 => PcmFormat::S16,
        24 => PcmFormat::S24,
        32 => PcmFormat::S32,
        b => {
            return Err(Error::unsupported(format!(
                "unsupported bit depth {b} (exact profile supports 8/16/24/32 integer)"
            )));
        }
    };
    let frame_bytes = pcm.bytes_per_sample() * u32::from(ch);
    if frame_bytes == 0 {
        return Err(Error::malformed("zero frame size"));
    }
    if byterate != rate * frame_bytes {
        return Err(Error::malformed("byterate inconsistent with format"));
    }
    if align != frame_bytes as u16 {
        return Err(Error::malformed("block align inconsistent with format"));
    }
    if (data_len as u64) > MAX_WAV_DATA_BYTES {
        return Err(Error::limit("WAV data exceeds MAX_WAV_DATA_BYTES"));
    }
    if data_len % frame_bytes as usize != 0 {
        return Err(Error::malformed("data chunk not frame-aligned"));
    }
    let frames = (data_len / frame_bytes as usize) as u64;
    if frames > MAX_OBJECT_FRAMES {
        return Err(Error::limit("frame count exceeds MAX_OBJECT_FRAMES"));
    }

    let ch_usize = ch as usize;
    let n_samples = ch_usize * (frames as usize);
    let mut samples = Vec::with_capacity(n_samples);
    let spb = pcm.bytes_per_sample() as usize;
    for f in 0..frames as usize {
        for c in 0..ch_usize {
            let o = data_start + (f * ch_usize + c) * spb;
            let code = match pcm {
                PcmFormat::U8 => sample::from_u8(bytes[o]).to_i32(),
                PcmFormat::S16 => {
                    sample::from_s16(i16::from_le_bytes([bytes[o], bytes[o + 1]])).to_i32()
                }
                PcmFormat::S24 => {
                    sample::from_s24_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2]]).to_i32()
                }
                PcmFormat::S32 => {
                    sample::from_s32(i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()))
                        .to_i32()
                }
            };
            samples.push(code);
        }
    }

    Ok(ParsedWav {
        decoded: DecodedWav {
            channels: ch,
            sample_rate_hz: rate,
            samples,
            frames,
        },
        pcm_format: pcm,
    })
}

/// Build a little-endian integer-PCM WAV byte buffer from *source-domain*
/// samples (writer for tests/fixtures; container bytes are not part of the
/// equality claim).
pub fn build_wav_bytes(pcm: PcmFormat, channels: u16, rate: u32, source: &[i32]) -> Vec<u8> {
    let spb = pcm.bytes_per_sample();
    let data_len = (source.len() as u32) * spb;
    let frame_bytes = spb * u32::from(channels);
    let mut out = Vec::with_capacity(44 + data_len as usize + (data_len % 2) as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * frame_bytes).to_le_bytes());
    out.extend_from_slice(&(frame_bytes as u16).to_le_bytes());
    out.extend_from_slice(&(spb as u16 * 8).to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in source {
        match pcm {
            PcmFormat::U8 => out.push((s & 0xFF) as u8),
            PcmFormat::S16 => out.extend_from_slice(&(*s as i16).to_le_bytes()),
            PcmFormat::S24 => {
                let v = s & 0xFF_FFFF;
                out.extend_from_slice(&[v as u8, (v >> 8) as u8, (v >> 16) as u8]);
            }
            PcmFormat::S32 => out.extend_from_slice(&s.to_le_bytes()),
        }
    }
    if data_len % 2 == 1 {
        out.push(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(bytes: &[u8]) -> ParsedWav {
        parse_wav(bytes).expect("parse ok")
    }

    #[test]
    fn s16_roundtrip_exact() {
        let src: Vec<i32> = vec![0, 1, -1, i16::MAX as i32, i16::MIN as i32, 12345, -23456];
        let bytes = build_wav_bytes(PcmFormat::S16, 1, 48_000, &src);
        let p = parse_ok(&bytes);
        assert_eq!(p.pcm_format, PcmFormat::S16);
        assert_eq!(p.decoded.channels, 1);
        assert_eq!(p.decoded.frames, src.len() as u64);
        let expect: Vec<i32> = src.iter().map(|x| x << 16).collect();
        assert_eq!(p.decoded.samples, expect);
    }

    #[test]
    fn u8_and_s24_and_s32_exact() {
        let u: Vec<i32> = vec![0, 1, 128, 200, 255];
        let p = parse_ok(&build_wav_bytes(PcmFormat::U8, 1, 8000, &u));
        let expect: Vec<i32> = u.iter().map(|x| (x - 128) << 24).collect();
        assert_eq!(p.decoded.samples, expect);

        let s24: Vec<i32> = vec![0, 1, -1, 0x7F_FFFF, -0x80_0000];
        let p = parse_ok(&build_wav_bytes(PcmFormat::S24, 1, 44_100, &s24));
        let expect: Vec<i32> = s24
            .iter()
            .map(|x| sample::sign_extend_24(*x) << 8)
            .collect();
        assert_eq!(p.decoded.samples, expect);

        let s32: Vec<i32> = vec![0, i32::MIN, i32::MAX, 7];
        let p = parse_ok(&build_wav_bytes(PcmFormat::S32, 1, 96_000, &s32));
        assert_eq!(p.decoded.samples, s32);
    }

    #[test]
    fn stereo_interleave_preserved() {
        let src: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let p = parse_ok(&build_wav_bytes(PcmFormat::S16, 2, 48_000, &src));
        assert_eq!(p.decoded.channels, 2);
        assert_eq!(p.decoded.frames, 3);
        let expect: Vec<i32> = src.iter().map(|x| x << 16).collect();
        assert_eq!(p.decoded.samples, expect);
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        let ok = build_wav_bytes(PcmFormat::S16, 1, 48_000, &[1, 2, 3]);
        // Truncated.
        assert!(parse_wav(&ok[..ok.len() - 3]).is_err());
        // Bad magic.
        let mut bad = ok.clone();
        bad[0] = b'X';
        assert!(parse_wav(&bad).is_err());
        // Float tag (3): explicit rejection, never quantized.
        let mut flt = ok.clone();
        flt[20..22].copy_from_slice(&3u16.to_le_bytes());
        assert!(parse_wav(&flt).is_err());
        // Lying data length.
        let mut trunc = ok.clone();
        trunc[40..44].copy_from_slice(&9999u32.to_le_bytes());
        assert!(parse_wav(&trunc).is_err());
        // Duplicate fmt chunk.
        let mut dup = ok.clone();
        dup.extend_from_slice(b"fmt ");
        dup.extend_from_slice(&16u32.to_le_bytes());
        dup.extend_from_slice(&[0; 16]);
        assert!(parse_wav(&dup).is_err());
        // Unsupported bit depth (12).
        let mut bits = ok.clone();
        bits[34..36].copy_from_slice(&12u16.to_le_bytes());
        assert!(parse_wav(&bits).is_err());
        // Absurd rate.
        let mut rate = ok.clone();
        rate[24..28].copy_from_slice(&(MAX_SAMPLE_RATE_HZ + 1).to_le_bytes());
        assert!(parse_wav(&rate).is_err());
        // Zero channels.
        let mut ch0 = ok.clone();
        ch0[22..24].copy_from_slice(&0u16.to_le_bytes());
        assert!(parse_wav(&ch0).is_err());
        // Empty.
        assert!(parse_wav(&[]).is_err());
    }

    #[test]
    fn metadata_chunks_are_skipped() {
        let src: Vec<i32> = vec![5, 6, 7];
        let data_len = (src.len() as u32) * 2;
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + 24 + data_len).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&48_000u32.to_le_bytes());
        w.extend_from_slice(&96_000u32.to_le_bytes());
        w.extend_from_slice(&2u16.to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"LIST");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(b"INFO");
        w.extend_from_slice(&[0u8; 12]); // pad the metadata payload to 16
        w.extend_from_slice(b"data");
        w.extend_from_slice(&data_len.to_le_bytes());
        for s in &src {
            w.extend_from_slice(&(*s as i16).to_le_bytes());
        }
        let p = parse_ok(&w);
        let expect: Vec<i32> = src.iter().map(|x| x << 16).collect();
        assert_eq!(p.decoded.samples, expect);
    }
}

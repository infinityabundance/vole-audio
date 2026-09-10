//! Deterministic transport framing (Phase N, contract §36).
//!
//! Transport carries **procedural/state information, not mandatory PCM**, and
//! the literal fallback is preserved as an object payload. The framing is
//! explicit binary, little-endian and canonical — serde/bincode layout is never
//! normative.
//!
//! ```text
//! Frame :=
//!     KIND         u8      FrameKind
//!     FLAGS        u8      bit0 = LAST (final frame of the stream); others must be 0
//!     EPOCH        u32     media epoch this frame belongs to
//!     SEQUENCE     u64     monotonic within an epoch
//!     MEDIA_FRAME  i64     timeline anchor (`-1` where the frame is not tied to a frame)
//!     PAYLOAD_LEN  u32     0..=MAX_FRAME_PAYLOAD_BYTES
//!     PAYLOAD      bytes
//!     DIGEST       32      SHA-256 over KIND..PAYLOAD
//! ```
//!
//! Every decode is length-checked: no length is cast to `usize`, the frame count
//! is bounded, and trailing bytes are rejected.

use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;
use crate::limits::{MAX_FRAME_PAYLOAD_BYTES, MAX_FRAMES_PER_STREAM};

/// Fixed header size preceding the payload.
pub const FRAME_HEADER_BYTES: usize = 1 + 1 + 4 + 8 + 8 + 4;
/// Trailing frame digest size.
pub const FRAME_DIGEST_BYTES: usize = 32;
/// `FLAGS` bit 0: the final frame of the stream.
pub const FLAG_LAST: u8 = 0x01;
/// `MEDIA_FRAME` sentinel meaning "not tied to a timeline frame".
pub const NO_MEDIA_FRAME: i64 = -1;

/// The transport frame kinds. Unknown kinds are rejected, never skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    /// A SampleObject (canonical U1 object bytes; literal fallback included).
    Object = 1,
    /// A timed event (see [`super::event`]).
    Event = 2,
    /// A state update (voice/parameter state).
    State = 3,
    /// A checkpoint (see [`super::checkpoint`]).
    Checkpoint = 4,
    /// A dependency declaration (content id this stream requires).
    Dependency = 5,
    /// A clock/epoch control frame.
    Clock = 6,
    /// An integrity frame (stream-level digest attestation).
    Integrity = 7,
}

impl FrameKind {
    pub fn from_u8(v: u8) -> Result<FrameKind> {
        Ok(match v {
            1 => FrameKind::Object,
            2 => FrameKind::Event,
            3 => FrameKind::State,
            4 => FrameKind::Checkpoint,
            5 => FrameKind::Dependency,
            6 => FrameKind::Clock,
            7 => FrameKind::Integrity,
            other => {
                return Err(Error::new(
                    crate::error::Kind::Unsupported,
                    format!("unknown transport frame kind {other}"),
                ));
            }
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            FrameKind::Object => "object",
            FrameKind::Event => "event",
            FrameKind::State => "state",
            FrameKind::Checkpoint => "checkpoint",
            FrameKind::Dependency => "dependency",
            FrameKind::Clock => "clock",
            FrameKind::Integrity => "integrity",
        }
    }
}

/// One transport frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub kind: FrameKind,
    pub flags: u8,
    pub epoch: u32,
    pub sequence: u64,
    pub media_frame: i64,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn new(kind: FrameKind, epoch: u32, sequence: u64, media_frame: i64) -> Self {
        Frame {
            kind,
            flags: 0,
            epoch,
            sequence,
            media_frame,
            payload: Vec::new(),
        }
    }

    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }

    pub fn last(mut self) -> Self {
        self.flags |= FLAG_LAST;
        self
    }

    pub fn is_last(&self) -> bool {
        self.flags & FLAG_LAST != 0
    }
}

/// Encode one frame (header + payload + trailing digest).
pub fn encode_frame(frame: &Frame) -> Result<Vec<u8>> {
    let payload_len = u32::try_from(frame.payload.len())
        .map_err(|_| Error::limit("transport frame payload exceeds the bound"))?;
    if payload_len > MAX_FRAME_PAYLOAD_BYTES {
        return Err(Error::limit("transport frame payload exceeds the bound"));
    }
    if frame.flags & !FLAG_LAST != 0 {
        return Err(Error::malformed(
            "transport frame flags have unknown bits set",
        ));
    }
    let mut out = Vec::with_capacity(FRAME_HEADER_BYTES + frame.payload.len() + FRAME_DIGEST_BYTES);
    out.push(frame.kind as u8);
    out.push(frame.flags);
    out.extend_from_slice(&frame.epoch.to_le_bytes());
    out.extend_from_slice(&frame.sequence.to_le_bytes());
    out.extend_from_slice(&frame.media_frame.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(&frame.payload);
    let digest = Sha256::digest(&out);
    out.extend_from_slice(&digest);
    Ok(out)
}

/// Decode one frame from the front of `bytes`, returning it and the number of
/// bytes consumed.
pub fn decode_frame(bytes: &[u8]) -> Result<(Frame, usize)> {
    if bytes.len() < FRAME_HEADER_BYTES + FRAME_DIGEST_BYTES {
        return Err(Error::malformed("transport frame is truncated"));
    }
    let kind = FrameKind::from_u8(bytes[0])?;
    let flags = bytes[1];
    if flags & !FLAG_LAST != 0 {
        return Err(Error::malformed(
            "transport frame flags have unknown bits set",
        ));
    }
    let epoch = u32::from_le_bytes(bytes[2..6].try_into().unwrap());
    let sequence = u64::from_le_bytes(bytes[6..14].try_into().unwrap());
    let media_frame = i64::from_le_bytes(bytes[14..22].try_into().unwrap());
    let payload_len = u32::from_le_bytes(bytes[22..26].try_into().unwrap());
    if payload_len > MAX_FRAME_PAYLOAD_BYTES {
        return Err(Error::limit("transport frame payload exceeds the bound"));
    }
    let payload_len = usize::try_from(payload_len)
        .map_err(|_| Error::limit("transport frame payload length exceeds host usize"))?;
    let end = 26usize
        .checked_add(payload_len)
        .ok_or_else(|| Error::limit("transport frame length overflows"))?;
    let total = end
        .checked_add(FRAME_DIGEST_BYTES)
        .ok_or_else(|| Error::limit("transport frame length overflows"))?;
    if total > bytes.len() {
        return Err(Error::malformed("transport frame body is truncated"));
    }
    let digest = Sha256::digest(&bytes[..end]);
    if digest != bytes[end..total] {
        return Err(Error::new(
            crate::error::Kind::Integrity,
            "transport frame digest mismatch",
        ));
    }
    Ok((
        Frame {
            kind,
            flags,
            epoch,
            sequence,
            media_frame,
            payload: bytes[26..end].to_vec(),
        },
        total,
    ))
}

/// Decode a whole frame stream, rejecting trailing bytes and bounding the count.
pub fn decode_stream(bytes: &[u8]) -> Result<Vec<Frame>> {
    let mut frames = Vec::new();
    let mut at = 0usize;
    while at < bytes.len() {
        if frames.len() as u32 >= MAX_FRAMES_PER_STREAM {
            return Err(Error::limit("transport stream exceeds the frame bound"));
        }
        let (frame, used) = decode_frame(&bytes[at..])?;
        frames.push(frame);
        at += used;
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(
            &encode_frame(
                &Frame::new(FrameKind::Object, 0, 0, NO_MEDIA_FRAME).with_payload(vec![1, 2, 3]),
            )
            .unwrap(),
        );
        out.extend_from_slice(
            &encode_frame(&Frame::new(FrameKind::Event, 0, 1, 42).with_payload(vec![9])).unwrap(),
        );
        let inner = Sha256::digest(&out);
        let last = Frame::new(FrameKind::Integrity, 0, 2, NO_MEDIA_FRAME)
            .with_payload(inner.to_vec())
            .last();
        out.extend_from_slice(&encode_frame(&last).unwrap());
        out
    }

    #[test]
    fn round_trip_is_exact() {
        let bytes = stream();
        let frames = decode_stream(&bytes).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].kind, FrameKind::Object);
        assert_eq!(frames[0].payload, vec![1, 2, 3]);
        assert!(frames[2].is_last());
        // Re-encoding a decoded frame is byte-identical.
        for f in &frames {
            let re = encode_frame(f).unwrap();
            let (back, used) = decode_frame(&re).unwrap();
            assert_eq!(&back, f);
            assert_eq!(used, re.len());
        }
    }

    #[test]
    fn trailing_and_truncated_input_is_rejected() {
        let mut bytes = stream();
        let extra = encode_frame(&Frame::new(FrameKind::State, 0, 3, NO_MEDIA_FRAME)).unwrap();
        bytes.extend_from_slice(&extra[..5]); // a partial frame
        assert!(decode_stream(&bytes).is_err());
        assert!(decode_frame(&[]).is_err());
        let full = stream();
        assert!(decode_stream(&full[..full.len() - 1]).is_err());
    }

    #[test]
    fn unknown_kind_flags_and_oversized_payloads_are_rejected() {
        let mut bytes = encode_frame(&Frame::new(FrameKind::State, 0, 0, NO_MEDIA_FRAME)).unwrap();
        bytes[0] = 0x7F;
        let n = bytes.len() - FRAME_DIGEST_BYTES;
        let digest = Sha256::digest(&bytes[..n]);
        bytes[n..].copy_from_slice(&digest);
        assert!(decode_frame(&bytes).is_err());

        let mut flags = encode_frame(&Frame::new(FrameKind::State, 0, 0, NO_MEDIA_FRAME)).unwrap();
        flags[1] = 0x80;
        assert!(decode_frame(&flags).is_err());

        // A payload length beyond the bound must be refused before allocating.
        let mut bomb = encode_frame(&Frame::new(FrameKind::State, 0, 0, NO_MEDIA_FRAME)).unwrap();
        bomb[22..26].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_frame(&bomb).is_err());
    }

    #[test]
    fn digest_corruption_is_an_integrity_error() {
        let mut bytes = stream();
        bytes[FRAME_HEADER_BYTES] ^= 0xFF; // flip a payload byte
        let err = decode_frame(&bytes).unwrap_err();
        assert_eq!(err.kind(), crate::error::Kind::Integrity);
    }
}

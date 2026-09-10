//! Stream integrity (Phase N, contract §36).
//!
//! Every frame carries its own digest (checked by [`super::frame::decode_frame`]).
//! This module adds the **stream-level** view: a digest over the whole encoded
//! stream, an `INTEGRITY` frame that attests the body preceding it, and a report
//! binding frame count, byte counts, epoch set and sequence span.

use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;

use super::frame::{Frame, FrameKind, decode_stream, encode_frame};

/// SHA-256 over the whole encoded stream.
pub fn stream_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes)
}

/// An `INTEGRITY` frame attesting the digest of everything before it.
pub fn integrity_frame(epoch: u32, sequence: u64, body: &[u8]) -> Frame {
    Frame::new(
        FrameKind::Integrity,
        epoch,
        sequence,
        super::frame::NO_MEDIA_FRAME,
    )
    .with_payload(Sha256::digest(body).to_vec())
    .last()
}

/// Verify a stream's `INTEGRITY` frame, if it carries one.
///
/// The frame must be the final frame and its payload must equal the digest of
/// the body that precedes it, otherwise the stream is rejected.
pub fn verify_integrity_frame(bytes: &[u8]) -> Result<()> {
    let frames = decode_stream(bytes)?;
    let Some(last) = frames.last() else {
        return Err(Error::malformed("empty transport stream"));
    };
    if last.kind != FrameKind::Integrity {
        return Ok(()); // no attestation claimed
    }
    if !last.is_last() {
        return Err(Error::malformed(
            "an integrity frame must be the final frame of the stream",
        ));
    }
    // Recover the offset of the integrity frame by re-encoding the body.
    let mut body = Vec::new();
    for f in &frames[..frames.len() - 1] {
        body.extend_from_slice(&encode_frame(f)?);
    }
    if last.payload != Sha256::digest(&body) {
        return Err(Error::new(
            crate::error::Kind::Integrity,
            "transport stream integrity frame does not match its body",
        ));
    }
    Ok(())
}

/// A summary of a decoded stream, for receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityReport {
    pub frames: usize,
    pub total_bytes: u64,
    pub body_bytes: u64,
    pub digest: [u8; 32],
    pub epochs: Vec<u32>,
    pub first_sequence: Option<u64>,
    pub last_sequence: Option<u64>,
    pub has_integrity_frame: bool,
}

/// Build a report over an encoded stream (decode-validated).
pub fn report(bytes: &[u8]) -> Result<IntegrityReport> {
    let frames = decode_stream(bytes)?;
    let has_integrity_frame = frames.last().map(|f| f.kind) == Some(FrameKind::Integrity);
    let mut epochs: Vec<u32> = frames.iter().map(|f| f.epoch).collect();
    epochs.dedup();
    Ok(IntegrityReport {
        frames: frames.len(),
        total_bytes: bytes.len() as u64,
        body_bytes: bytes.len() as u64 - (frames.len() * super::frame::FRAME_DIGEST_BYTES) as u64,
        digest: stream_digest(bytes),
        epochs,
        first_sequence: frames.first().map(|f| f.sequence),
        last_sequence: frames.last().map(|f| f.sequence),
        has_integrity_frame,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(
            &encode_frame(&Frame::new(FrameKind::Object, 0, 0, -1).with_payload(vec![7])).unwrap(),
        );
        out.extend_from_slice(&encode_frame(&Frame::new(FrameKind::Event, 0, 1, 12)).unwrap());
        out
    }

    #[test]
    fn integrity_frame_attests_the_body() {
        let body = body();
        let mut stream = body.clone();
        let last = integrity_frame(0, 2, &body);
        stream.extend_from_slice(&encode_frame(&last).unwrap());
        verify_integrity_frame(&stream).unwrap();
        let rep = report(&stream).unwrap();
        assert_eq!(rep.frames, 3);
        assert!(rep.has_integrity_frame);
        assert_eq!(rep.first_sequence, Some(0));
        assert_eq!(rep.last_sequence, Some(2));
        assert_eq!(rep.epochs, vec![0]);
    }

    #[test]
    fn a_forged_integrity_attestation_is_rejected() {
        let body = body();
        let mut stream = body.clone();
        let mut last = integrity_frame(0, 2, &body);
        last.payload = vec![0u8; 32]; // wrong attestation
        stream.extend_from_slice(&encode_frame(&last).unwrap());
        let err = verify_integrity_frame(&stream).unwrap_err();
        assert_eq!(err.kind(), crate::error::Kind::Integrity);
    }
}

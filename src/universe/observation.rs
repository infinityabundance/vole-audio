//! Observation views (frozen canonical form).
//!
//! An observation is a bounded, deterministic materialization of audio state
//! over an interval. The canonical byte form for u1 hashing/archiving is:
//!
//! ```text
//! for each frame f in [0, frames):
//!   for each channel c in layout order:
//!     i32 sample code, little-endian
//! ```
//!
//! Identical semantics observed by any backend must produce identical bytes;
//! `observation_sha256` is the reference-equality primitive used by courts
//! (reference hash vs backend hash) and reference vectors.

use crate::hash::sha256::Sha256;
use crate::universe::layout::Layout;

/// Canonical interleaved byte length for `frames` of `layout`.
pub fn interleaved_byte_len(layout: Layout, frames: usize) -> usize {
    layout
        .interleaved_len(frames)
        .expect("frames * channels overflow")
        * 4
}

/// Write an interleaved i32 observation to its canonical little-endian bytes.
pub fn interleaved_to_bytes(samples: &[i32], out: &mut [u8]) {
    assert_eq!(out.len(), samples.len() * 4, "byte buffer length");
    for (i, s) in samples.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
    }
}

/// Canonical observation hash: SHA-256 over the canonical interleaved
/// little-endian bytes of the sample codes. Streams sample-wise so no
/// intermediate byte buffer is required (`no_std`-clean).
pub fn observation_sha256(samples: &[i32]) -> [u8; 32] {
    let mut h = Sha256::new();
    let mut le = [0u8; 4];
    for s in samples {
        le.copy_from_slice(&s.to_le_bytes());
        h.update(&le);
    }
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_bytes_are_little_endian() {
        let samples = [1i32, -1, 0, i32::MAX, i32::MIN];
        let mut out = vec![0u8; samples.len() * 4];
        interleaved_to_bytes(&samples, &mut out);
        assert_eq!(&out[0..4], &[1, 0, 0, 0]);
        assert_eq!(&out[4..8], &[0xFF; 4]);
        assert_eq!(&out[12..16], &[0xFF, 0xFF, 0xFF, 0x7F]);
        assert_eq!(&out[16..20], &[0, 0, 0, 0x80]);
    }

    #[test]
    fn hash_is_deterministic_and_sensitive() {
        let a = observation_sha256(&[1, 2, 3]);
        let b = observation_sha256(&[1, 2, 3]);
        let c = observation_sha256(&[1, 2, 4]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Reference vector (frozen): SHA-256 of the empty observation.
        let e = observation_sha256(&[]);
        assert_eq!(
            crate::hash::sha256::hex(&e),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn interleaved_len_matches_layout() {
        assert_eq!(interleaved_byte_len(Layout::Stereo, 10), 80);
        assert_eq!(interleaved_byte_len(Layout::Mono, 10), 40);
    }
}

//! Literal sampled SampleObject payload.
//!
//! Literal is the universal fallback and is always legal. Payload storage is
//! the **canonical interleaved** i32 form (frame-major, channel-minor,
//! matching the canonical observation form), so the same bytes that define
//! content identity can be read directly by any backend. Reads of a single
//! channel stride over the interleaved layout.

use crate::hash::sha256::Sha256;
use crate::limits::MAX_WAV_DATA_BYTES;
use crate::object::descriptor::{ObjectDescriptor, canonical_header_bytes};
use crate::object::id::ContentId;

/// Literal payload: interleaved sample codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Literal {
    /// Canonical interleaved sample codes (frames * channels).
    pub samples: Vec<i32>,
}

impl Literal {
    /// Build a literal payload; validates length ceilings against the
    /// descriptor extent/layout.
    pub fn new(descriptor: &ObjectDescriptor, samples: Vec<i32>) -> Option<Literal> {
        let ch = usize::from(descriptor.layout.count());
        let expect = usize::try_from(descriptor.extent_frames)
            .ok()?
            .checked_mul(ch)?;
        if samples.len() != expect {
            return None;
        }
        if (samples.len() as u64) * 4 > MAX_WAV_DATA_BYTES {
            return None;
        }
        Some(Literal { samples })
    }

    /// Sample at `(frame, channel)` (no bounds check — caller guarantees).
    #[inline]
    pub fn sample(&self, frame: usize, channel: usize, channels: usize) -> i32 {
        // Interleaved layout: frame-major.
        self.samples[frame * channels + channel]
    }

    /// Canonical payload bytes: `header || sample_count(u64 LE) || codes (i32 LE)`.
    pub fn canonical_bytes(&self, descriptor: &ObjectDescriptor) -> Vec<u8> {
        let mut out = canonical_header_bytes(descriptor);
        out.extend_from_slice(&(self.samples.len() as u64).to_le_bytes());
        for s in &self.samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    /// Content identity of a literal object.
    pub fn content_id(descriptor: &ObjectDescriptor, literal: &Literal) -> ContentId {
        ContentId(Sha256::digest(&literal.canonical_bytes(descriptor)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::descriptor::Representation;
    use crate::universe::layout::Layout;

    fn mono_descriptor(len: u64) -> ObjectDescriptor {
        ObjectDescriptor::new(Representation::Literal, len, Layout::Mono, None).unwrap()
    }

    #[test]
    fn payload_len_must_match_descriptor() {
        let d = mono_descriptor(4);
        assert!(Literal::new(&d, vec![1, 2, 3, 4]).is_some());
        assert!(Literal::new(&d, vec![1, 2, 3]).is_none());
        assert!(Literal::new(&d, vec![1, 2, 3, 4, 5]).is_none());
        // Stereo descriptor needs frames*2 samples.
        let sd = ObjectDescriptor::new(Representation::Literal, 3, Layout::Stereo, None).unwrap();
        assert!(Literal::new(&sd, vec![0; 6]).is_some());
        assert!(Literal::new(&sd, vec![0; 5]).is_none());
    }

    #[test]
    fn interleaved_reads_match_layout() {
        let d = ObjectDescriptor::new(Representation::Literal, 2, Layout::Stereo, None).unwrap();
        let lit = Literal::new(&d, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(lit.sample(0, 0, 2), 1);
        assert_eq!(lit.sample(0, 1, 2), 2);
        assert_eq!(lit.sample(1, 0, 2), 3);
        assert_eq!(lit.sample(1, 1, 2), 4);
    }

    #[test]
    fn identity_changes_with_content() {
        let d = mono_descriptor(2);
        let a = Literal::new(&d, vec![1, 2]).unwrap();
        let b = Literal::new(&d, vec![1, 3]).unwrap();
        let ia = Literal::content_id(&d, &a);
        let ib = Literal::content_id(&d, &b);
        assert_ne!(ia, ib);
        let d2 = mono_descriptor(3);
        let c = Literal::new(&d2, vec![1, 2, 0]).unwrap();
        assert_ne!(Literal::content_id(&d2, &c), ia);
    }

    #[test]
    fn empty_object_identity_is_stable() {
        let d = mono_descriptor(0);
        let l = Literal::new(&d, vec![]).unwrap();
        let id = Literal::content_id(&d, &l);
        let hex = crate::hash::sha256::hex(&id.to_bytes());
        assert_eq!(hex.len(), 64);
        // Determinism.
        assert_eq!(id, Literal::content_id(&d, &l));
    }
}

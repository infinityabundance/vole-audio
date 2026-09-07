//! Reference (alias) SampleObject.
//!
//! A referenced object is an alias to another object's *content*, with an
//! optional transpose (Q24 rate multiplier) and an optional loop override.
//! Content identity of a reference is computed from the target's content id,
//! so archives stay portable (no archive-local ids in identity).

use crate::hash::sha256::Sha256;
use crate::object::MAX_ACCUMULATED_TRANSPOSE_Q24;
use crate::object::descriptor::{LoopRegion, ObjectDescriptor, canonical_header_bytes};
use crate::object::id::ContentId;

/// Reference payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Referenced {
    /// Content id of the target object.
    pub target_content: ContentId,
    /// Transpose as a Q24 rate multiplier (unity = `1 << 24`), saturated at
    /// the accumulated ceiling.
    pub transpose_q24: i64,
    /// Optional loop override (validated against target extent at load).
    pub loop_override: Option<LoopRegion>,
}

impl Referenced {
    /// Validate a candidate reference payload against a descriptor.
    pub fn checked(
        target_content: ContentId,
        transpose_q24: i64,
        loop_override: Option<LoopRegion>,
    ) -> Option<Referenced> {
        if transpose_q24.unsigned_abs() > MAX_ACCUMULATED_TRANSPOSE_Q24 as u64 {
            return None;
        }
        Some(Referenced {
            target_content,
            transpose_q24,
            loop_override,
        })
    }

    /// Canonical payload bytes:
    /// `header || transpose(q24 LE) || loop_flag || loop_start || loop_end ||
    /// target_content(32)`.
    pub fn canonical_bytes(&self, descriptor: &ObjectDescriptor) -> Vec<u8> {
        let mut out = canonical_header_bytes(descriptor);
        out.extend_from_slice(&self.transpose_q24.to_le_bytes());
        match self.loop_override {
            Some(l) => {
                out.push(1);
                out.extend_from_slice(&l.start_frame.to_le_bytes());
                out.extend_from_slice(&l.end_frame.to_le_bytes());
            }
            None => {
                out.push(0);
                out.extend_from_slice(&0u64.to_le_bytes());
                out.extend_from_slice(&0u64.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.target_content.to_bytes());
        out
    }

    /// Content identity (kept for symmetry with `Literal::content_id`).
    pub fn content_id(descriptor: &ObjectDescriptor, r: &Referenced) -> ContentId {
        ContentId(Sha256::digest(&r.canonical_bytes(descriptor)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::descriptor::Representation;
    use crate::universe::layout::Layout;

    #[test]
    fn reference_identity_depends_on_target_and_transpose() {
        let d = ObjectDescriptor::new(Representation::Referenced, 100, Layout::Mono, None).unwrap();
        let target = ContentId([7; 32]);
        let a = Referenced::checked(target, 1 << 24, None).unwrap();
        let b = Referenced::checked(target, 1 << 23, None).unwrap();
        let c = Referenced::checked(ContentId([8; 32]), 1 << 24, None).unwrap();
        assert_eq!(
            Referenced::content_id(&d, &a),
            Referenced::content_id(&d, &a)
        );
        assert_ne!(
            Referenced::content_id(&d, &a),
            Referenced::content_id(&d, &b)
        );
        assert_ne!(
            Referenced::content_id(&d, &a),
            Referenced::content_id(&d, &c)
        );
    }

    #[test]
    fn transpose_bounds() {
        assert!(
            Referenced::checked(ContentId([0; 32]), MAX_ACCUMULATED_TRANSPOSE_Q24, None).is_some()
        );
        assert!(
            Referenced::checked(ContentId([0; 32]), MAX_ACCUMULATED_TRANSPOSE_Q24 + 1, None)
                .is_none()
        );
    }
}

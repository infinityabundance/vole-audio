//! Trivial procedural payloads: silence, constant, deterministic noise.
//!
//! These are endless sources (extent 0, no resident PCM): silence/constant
//! emit their level regardless of time; noise emits
//! `VOLE-SPLITMIX64-STREAM` samples keyed by the object's stream seed and the
//! *media frame*.

use crate::hash::sha256::Sha256;
use crate::object::descriptor::{ObjectDescriptor, Representation, canonical_header_bytes};
use crate::object::id::ContentId;

/// Constant level object (SampleCode domain).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Constant {
    pub level: i32,
}

/// Deterministic noise object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Noise {
    /// Stream seed; two voices on the same noise object at the same media
    /// frame produce identical samples (two copies of the same "recording").
    pub seed: u64,
}

impl Constant {
    pub const fn new(level: i32) -> Constant {
        Constant { level }
    }
}

impl Noise {
    pub const fn new(seed: u64) -> Noise {
        Noise { seed }
    }
}

/// Canonical bytes for the trivial payloads (header + fixed payload).
pub fn canonical_bytes(
    descriptor: &ObjectDescriptor,
    rep: Representation,
    payload: &[u8],
) -> Vec<u8> {
    let mut d = descriptor.clone();
    d.representation = rep;
    let mut out = canonical_header_bytes(&d);
    out.extend_from_slice(payload);
    out
}

pub fn content_id(bytes: &[u8]) -> ContentId {
    ContentId(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::universe::layout::Layout;

    fn od(rep: Representation) -> ObjectDescriptor {
        ObjectDescriptor::new(rep, 0, Layout::Mono, None).unwrap()
    }

    #[test]
    fn identity_distinguishes_level_seed_and_tag() {
        let d = od(Representation::Constant);
        let a = content_id(&canonical_bytes(
            &d,
            Representation::Constant,
            &5i32.to_le_bytes(),
        ));
        let b = content_id(&canonical_bytes(
            &d,
            Representation::Constant,
            &6i32.to_le_bytes(),
        ));
        assert_ne!(a, b);
        // Silence (empty payload) differs from a zero constant.
        let s = content_id(&canonical_bytes(&d, Representation::Silence, &[]));
        let z = content_id(&canonical_bytes(
            &d,
            Representation::Constant,
            &0i32.to_le_bytes(),
        ));
        assert_ne!(s, z);
        let dn = od(Representation::Noise);
        let n1 = content_id(&canonical_bytes(
            &dn,
            Representation::Noise,
            &7u64.to_le_bytes(),
        ));
        let n2 = content_id(&canonical_bytes(
            &dn,
            Representation::Noise,
            &8u64.to_le_bytes(),
        ));
        assert_ne!(n1, n2);
    }
}

//! The experimental learned-prediction namespace (`O.0`).
//!
//! Phase O does **not** modify the frozen `vole.audio.u1` / `u1/v1` semantics.
//! Learned hypotheses live under an explicit experimental profile so that no
//! learned object can be mistaken for a `u1/v1` `SampleObject`, and so that a
//! future profile admitting learned representations is a separate, versioned,
//! auditable decision.
//!
//! The learned profile still belongs to universe `vole.audio.u1`: a learned
//! object closes to the same canonical intrinsic sample domain and must satisfy
//! the same exactness rules. What it does not do is extend the frozen `u1/v1`
//! `Representation` taxonomy.

/// Universe the learned profile belongs to (unchanged).
pub const LEARNED_UNIVERSE: &str = "vole.audio.u1";

/// Experimental learned representation profile identity.
pub const LEARNED_PROFILE: &str = "vole.audio.learned.exp1";

/// Experimental profile version (integer form for binary formats).
pub const LEARNED_PROFILE_VERSION: u32 = 1;

/// Canonical profile tag bytes carried by every learned object.
pub const LEARNED_PROFILE_TAG: &[u8] = b"vole.audio.u1/vole.audio.learned.exp1";

/// Canonical learned container magic.
pub const LEARNED_MAGIC: &[u8; 12] = b"vole.learned";

/// Canonical learned container format version.
pub const LEARNED_FORMAT_VERSION: u8 = 1;

/// Evidence schema for Phase O receipts.
pub const LEARNED_EVIDENCE_SCHEMA: &str = "vole.audio.learned.evidence.v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_identity_is_explicit_and_never_u1_v1() {
        assert_eq!(LEARNED_UNIVERSE, "vole.audio.u1");
        assert_eq!(LEARNED_PROFILE, "vole.audio.learned.exp1");
        assert_ne!(LEARNED_PROFILE, "u1/v1");
        assert_eq!(
            LEARNED_PROFILE_TAG,
            b"vole.audio.u1/vole.audio.learned.exp1"
        );
        // The learned profile tag is not the frozen u1 tag.
        assert_ne!(LEARNED_PROFILE_TAG, crate::universe::u1::PROFILE_TAG_BYTES);
        assert_eq!(LEARNED_MAGIC, b"vole.learned");
    }
}

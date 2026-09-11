//! The experimental learned-prediction namespaces (`O.0`, Exp2 `0`).
//!
//! Phase O does **not** modify the frozen `vole.audio.u1` / `u1/v1` semantics.
//! Learned hypotheses live under explicit experimental profiles so that no
//! learned object can be mistaken for a `u1/v1` `SampleObject`, and so that a
//! future profile admitting learned representations is a separate, versioned,
//! auditable decision.
//!
//! Exp2 (this addendum) is a **separate** profile from Exp1. Exp1 is frozen
//! permanently: its profile tag, its four model kinds, its six residual codecs
//! and its canonical container bytes must never change. Exp2 imports every
//! Exp1 candidate and adds new mechanisms beside it, so that under the same
//! complete-physical-byte objective
//!
//! ```text
//! exp1_candidates ⊂ exp2_candidates  ⇒  min(exp2) ≤ min(exp1)
//! ```
//!
//! which is a structural no-regression guarantee rather than a benchmark claim.

/// Universe the learned profiles belong to (unchanged).
pub const LEARNED_UNIVERSE: &str = "vole.audio.u1";

/// Frozen experimental learned representation profile identity (Exp1).
pub const LEARNED_PROFILE: &str = "vole.audio.learned.exp1";

/// Frozen experimental profile version (integer form for binary formats).
pub const LEARNED_PROFILE_VERSION: u32 = 1;

/// Canonical profile tag bytes carried by every Exp1 learned object.
pub const LEARNED_PROFILE_TAG: &[u8] = b"vole.audio.u1/vole.audio.learned.exp1";

/// Exp2 experimental learned representation profile identity.
pub const LEARNED_EXP2_PROFILE: &str = "vole.audio.learned.exp2";

/// Exp2 profile version (integer form for binary formats).
pub const LEARNED_EXP2_PROFILE_VERSION: u32 = 1;

/// Canonical profile tag bytes carried by every Exp2 learned object.
pub const LEARNED_EXP2_PROFILE_TAG: &[u8] = b"vole.audio.u1/vole.audio.learned.exp2";

/// Exp3 experimental learned representation profile identity (Seal S4).
///
/// Exp3 imports every Exp2 candidate and adds the Seal S4 residual codecs
/// (general Golomb, centered Golomb). It exists so the Exp2 profile and the
/// frozen Seal-K real-corpus evidence are left byte-for-byte untouched.
pub const LEARNED_EXP3_PROFILE: &str = "vole.audio.learned.exp3";

/// Exp3 profile version (integer form for binary formats).
pub const LEARNED_EXP3_PROFILE_VERSION: u32 = 1;

/// Canonical profile tag bytes carried by every Exp3 learned object.
pub const LEARNED_EXP3_PROFILE_TAG: &[u8] = b"vole.audio.u1/vole.audio.learned.exp3";

/// Canonical learned container magic (shared by both profiles).
pub const LEARNED_MAGIC: &[u8; 12] = b"vole.learned";

/// Canonical learned container format version.
pub const LEARNED_FORMAT_VERSION: u8 = 1;

/// Evidence schema for Phase O receipts.
pub const LEARNED_EVIDENCE_SCHEMA: &str = "vole.audio.learned.evidence.v1";

/// Evidence schema for the Exp2 addendum receipts.
pub const LEARNED_EXP2_EVIDENCE_SCHEMA: &str = "vole.audio.learned.exp2.evidence.v1";

/// Evidence schema for the Exp3 addendum receipts.
pub const LEARNED_EXP3_EVIDENCE_SCHEMA: &str = "vole.audio.learned.exp3.evidence.v1";

/// The learned experimental profile an object belongs to.
///
/// The container layout is identical; only the profile tag and the admissible
/// model/residual vocabulary differ. Exp1 is permanently frozen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum LearnedProfile {
    /// `vole.audio.learned.exp1` — the sealed Phase O profile.
    Exp1 = 1,
    /// `vole.audio.learned.exp2` — the optimization addendum.
    Exp2 = 2,
    /// `vole.audio.learned.exp3` — the Seal S4 residual-codec addendum.
    Exp3 = 3,
}

impl LearnedProfile {
    /// Canonical profile identity string.
    pub const fn name(self) -> &'static str {
        match self {
            LearnedProfile::Exp1 => LEARNED_PROFILE,
            LearnedProfile::Exp2 => LEARNED_EXP2_PROFILE,
            LearnedProfile::Exp3 => LEARNED_EXP3_PROFILE,
        }
    }

    /// Integer profile version.
    pub const fn version(self) -> u32 {
        match self {
            LearnedProfile::Exp1 => LEARNED_PROFILE_VERSION,
            LearnedProfile::Exp2 => LEARNED_EXP2_PROFILE_VERSION,
            LearnedProfile::Exp3 => LEARNED_EXP3_PROFILE_VERSION,
        }
    }

    /// Canonical profile tag bytes written into the container.
    pub const fn tag(self) -> &'static [u8] {
        match self {
            LearnedProfile::Exp1 => LEARNED_PROFILE_TAG,
            LearnedProfile::Exp2 => LEARNED_EXP2_PROFILE_TAG,
            LearnedProfile::Exp3 => LEARNED_EXP3_PROFILE_TAG,
        }
    }

    /// Resolve a profile from its canonical tag bytes.
    pub fn from_tag(tag: &[u8]) -> Option<LearnedProfile> {
        if tag == LEARNED_PROFILE_TAG {
            Some(LearnedProfile::Exp1)
        } else if tag == LEARNED_EXP2_PROFILE_TAG {
            Some(LearnedProfile::Exp2)
        } else if tag == LEARNED_EXP3_PROFILE_TAG {
            Some(LearnedProfile::Exp3)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_identity_is_explicit_and_never_u1_v1() {
        assert_eq!(LEARNED_UNIVERSE, "vole.audio.u1");
        assert_eq!(LEARNED_PROFILE, "vole.audio.learned.exp1");
        assert_eq!(LEARNED_EXP2_PROFILE, "vole.audio.learned.exp2");
        assert_ne!(LEARNED_PROFILE, "u1/v1");
        assert_ne!(LEARNED_PROFILE, LEARNED_EXP2_PROFILE);
        assert_eq!(
            LEARNED_PROFILE_TAG,
            b"vole.audio.u1/vole.audio.learned.exp1"
        );
        assert_eq!(
            LEARNED_EXP2_PROFILE_TAG,
            b"vole.audio.u1/vole.audio.learned.exp2"
        );
        // The learned profile tags are not the frozen u1 tag.
        assert_ne!(LEARNED_PROFILE_TAG, crate::universe::u1::PROFILE_TAG_BYTES);
        assert_ne!(
            LEARNED_EXP2_PROFILE_TAG,
            crate::universe::u1::PROFILE_TAG_BYTES
        );
        assert_eq!(LEARNED_MAGIC, b"vole.learned");
        // Frozen Exp1 identity must be byte-identical to the sealed Phase O.
        assert_eq!(LEARNED_PROFILE_VERSION, 1);
        assert_eq!(LEARNED_FORMAT_VERSION, 1);
    }

    #[test]
    fn profiles_resolve_from_tags_and_never_cross() {
        assert_eq!(
            LearnedProfile::from_tag(LEARNED_PROFILE_TAG),
            Some(LearnedProfile::Exp1)
        );
        assert_eq!(
            LearnedProfile::from_tag(LEARNED_EXP2_PROFILE_TAG),
            Some(LearnedProfile::Exp2)
        );
        assert_eq!(LearnedProfile::from_tag(b"vole.audio.u1/v1"), None);
        assert_eq!(LearnedProfile::Exp1.tag(), LEARNED_PROFILE_TAG);
        assert_eq!(LearnedProfile::Exp2.tag(), LEARNED_EXP2_PROFILE_TAG);
        assert_eq!(LearnedProfile::Exp2.name(), LEARNED_EXP2_PROFILE);
        // Both tags have the same length, so the container framing length is
        // identical across profiles (accounting stays exact either way).
        assert_eq!(LEARNED_PROFILE_TAG.len(), LEARNED_EXP2_PROFILE_TAG.len());
    }
}

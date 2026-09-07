//! Universe identity for `vole.audio.u1`.
//!
//! The universe id and profile id are frozen strings that travel with every
//! object, archive, receipt, and device artifact. Changing semantics = new
//! profile id + fresh reference vectors; the code enforces nothing silently.

/// Canonical universe id (first semantic universe).
pub const UNIVERSE_ID: &str = "vole.audio.u1";

/// Profile id within the universe (v1 freeze).
pub const PROFILE_ID: &str = "u1/v1";

/// Profile version (integer form for binary formats).
pub const PROFILE_VERSION: u32 = 1;

/// Nominal sample rate of the u1 default profile.
pub const NOMINAL_RATE_HZ: u32 = crate::limits::DEFAULT_SAMPLE_RATE_HZ;

/// Default observation quantum (frames) for the scheduler.
pub const DEFAULT_QUANTUM: u32 = crate::limits::DEFAULT_QUANTUM_FRAMES;

/// Canonical bytes identifying this universe/profile on the wire.
pub const UNIVERSE_ID_BYTES: &[u8] = b"vole.audio.u1";

/// ASCII space = 0x20. Universe id plus profile in one canonical tag.
pub const PROFILE_TAG_BYTES: &[u8] = b"vole.audio.u1/u1/v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_strings_are_stable() {
        assert_eq!(UNIVERSE_ID, "vole.audio.u1");
        assert_eq!(PROFILE_ID, "u1/v1");
        assert_eq!(PROFILE_VERSION, 1);
        assert_eq!(PROFILE_TAG_BYTES, b"vole.audio.u1/u1/v1");
        assert_eq!(NOMINAL_RATE_HZ, 48_000);
        assert_eq!(DEFAULT_QUANTUM, 1024);
    }
}

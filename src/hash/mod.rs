//! Cryptographic identity and integrity primitives.
//!
//! SHA-256 (FIPS 180-4) is the u1 content-identity hash and the archive
//! integrity hash. The implementation is in-repo (see `sha256.rs`) so the same
//! bytes are available in `no_std` device builds and host builds with no
//! external dependency.
//!
//! This is a plain primitive module shared across the tree; it is not part of
//! the semantic-universe module layout.

pub mod sha256;

pub use sha256::Sha256;

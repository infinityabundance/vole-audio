//! Conventional baselines (Phase M, contract §47).
//!
//! The canonical comparison must include the B0–B9 ladder. This module owns the
//! **conventional** rows — the ones that are not VOLE surfaces:
//!
//! | id | baseline | implementation |
//! | -- | -------- | -------------- |
//! | B0 | literal PCM | exact canonical interleaved i32 bytes, uncompressed |
//! | B1 | conventional lossless codec | FLAC, in-process pure Rust, 32-bit |
//!
//! Both baselines have **zero VOLE semantic authority**. They are comparators:
//! the only correctness requirement they carry is that the exact information the
//! VOLE representation is priced against survives their round trip. B1
//! additionally has no escape hatch — it is not allowed to become `UNSUPPORTED`
//! — because the point of the comparison is a real conventional codec solving
//! the same information-preservation problem.

pub mod flac;
pub mod reference;

pub use flac::{
    B1_LEVEL_CONTROLS, B1_LEVEL_PRIMARY, FLAC_MAX_CHANNELS, FlacArtifact, FlacEncoding, b1_flac,
    b1_flac_artifact, b1_level_label,
};
pub use reference::{ReferenceFlac, reference_flac};

/// B0 — literal PCM: the canonical interleaved i32 bytes, uncompressed.
///
/// This is the information-preservation floor of the comparison: 4 bytes per
/// sample with no modelling whatsoever.
pub const fn b0_raw_pcm_bytes(samples: &[i32]) -> u64 {
    samples.len() as u64 * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b0_is_four_bytes_per_sample() {
        assert_eq!(b0_raw_pcm_bytes(&[]), 0);
        assert_eq!(b0_raw_pcm_bytes(&[0, 1, -1, i32::MIN]), 16);
    }
}

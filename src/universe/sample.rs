//! The canonical u1 sample-code domain.
//!
//! Frozen (U1_SPEC §"Sample domain"):
//!
//! * `SampleCode` is a signed 32-bit integer; the *entire* i32 range is valid
//!   (s32 PCM semantics, `i32::MIN` included).
//! * Integer PCM ingest maps *exactly* into this domain:
//!
//!   | source format      | mapping                                  |
//!   | ------------------ | ---------------------------------------- |
//!   | unsigned 8-bit     | `((x as i32) - 128) << 24`               |
//!   | signed 16-bit      | `x << 16`                                |
//!   | signed 24-bit      | `sign_extend_24(x) << 8`                 |
//!   | signed 32-bit      | `x`                                      |
//!
//! * `exact audio-sample reconstruction != exact WAV container byte
//!   reconstruction`. WAV metadata/chunk layout is not part of the u1 sample
//!   equality claim.
//! * Floating-point WAV input is rejected by the exact profile (no silent
//!   quantization). A lossy import profile, if requested later, records the
//!   transformed target hash separately.

use core::fmt;

/// Canonical signed 32-bit sample code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SampleCode(pub i32);

impl SampleCode {
    pub const fn new(v: i32) -> Self {
        Self(v)
    }

    pub const SILENCE: SampleCode = SampleCode(0);

    pub const fn to_i32(self) -> i32 {
        self.0
    }
}

impl fmt::Display for SampleCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Lossless conversion from unsigned 8-bit PCM sample.
#[inline]
pub const fn from_u8(x: u8) -> SampleCode {
    SampleCode(((x as i32) - 128) << 24)
}

/// Lossless conversion from signed 16-bit PCM sample.
#[inline]
pub const fn from_s16(x: i16) -> SampleCode {
    SampleCode((x as i32) << 16)
}

/// Lossless conversion from signed 24-bit PCM sample stored in the low 24
/// bits of an i32 (sign-extended first).
#[inline]
pub const fn from_s24(x: i32) -> SampleCode {
    SampleCode(sign_extend_24(x) << 8)
}

/// Lossless conversion from signed 32-bit PCM sample (identity).
#[inline]
pub const fn from_s32(x: i32) -> SampleCode {
    SampleCode(x)
}

/// Sign-extend the low 24 bits of `x` to a full i32.
#[inline]
pub const fn sign_extend_24(x: i32) -> i32 {
    // Arithmetic shift of the 24-bit value into the sign position, then back.
    (x << 8) >> 8
}

/// Convert a little-endian signed 24-bit triple (as 3 bytes) to i32.
#[inline]
pub fn from_s24_le_bytes(b: [u8; 3]) -> SampleCode {
    let v = (b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i32) << 16);
    from_s24(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_mapping_midpoint_is_silence() {
        assert_eq!(from_u8(128).0, 0);
        assert_eq!(from_u8(0).0, i32::MIN);
        // (255 - 128) << 24 == 127 << 24 (the u8 domain maps to
        // [-2^31, 2^31 - 2^24]; per the frozen mapping).
        assert_eq!(from_u8(255).0, 127 << 24);
        assert_eq!(from_u8(129).0, 1 << 24);
        assert_eq!(from_u8(127).0, -(1 << 24));
    }

    #[test]
    fn s16_mapping_is_left_shift_16() {
        assert_eq!(from_s16(0).0, 0);
        assert_eq!(from_s16(1).0, 1 << 16);
        assert_eq!(from_s16(-1).0, -(1 << 16));
        assert_eq!(from_s16(i16::MAX).0, (i16::MAX as i32) << 16);
        assert_eq!(from_s16(i16::MIN).0, (i16::MIN as i32) << 16);
    }

    #[test]
    fn s24_mapping_sign_extends() {
        assert_eq!(from_s24(0).0, 0);
        assert_eq!(from_s24(1).0, 1 << 8);
        assert_eq!(from_s24(-1).0, -(1 << 8));
        // 0x7FFFFF = +8388607
        assert_eq!(from_s24(0x7f_ff_ff).0, 0x7f_ff_ff << 8);
        // 0x800000 = -8388608
        assert_eq!(from_s24(0x80_00_00).0, -0x80_00_00 << 8);
        assert_eq!(from_s24(0xFF_FF_FF).0, -256);
    }

    #[test]
    fn sign_extend_24_matches_shift_pair() {
        for v in [
            0,
            1,
            -1,
            0x7f_ff_ff,
            0x80_00_00,
            0xff_ff_ff,
            0x12_34_56,
            -0x12_34_56,
        ] {
            assert_eq!(sign_extend_24(v), (v << 8) >> 8);
        }
    }

    #[test]
    fn le_triple_roundtrip() {
        // +0x123456
        assert_eq!(from_s24_le_bytes([0x56, 0x34, 0x12]).0, 0x123456 << 8);
        // -1
        assert_eq!(from_s24_le_bytes([0xff, 0xff, 0xff]).0, -(1 << 8));
    }
}

//! Voice gain semantics (frozen).
//!
//! Gain is a Q16 multiplier with unity `1<<16` and a hard ceiling of `2.0`
//! (`MAX_GAIN_Q16`). The amplitude chain is:
//!
//! ```text
//! e   = sat(rnd(env * gain, 16))          // combined Q16 multiplier
//! mch = sat(rnd(e * pan_ch, 16))          // per-channel Q16 multiplier
//! contribution = sat(rnd(obs * mch, 16))  // voice-bus boundary
//! ```
//!
//! Each stage is `mul_q16` — one rounding, one saturation, deterministic on
//! every backend. With unity env/gain/pan the chain is the identity.

use crate::universe::arithmetic::mul_q16;

/// Maximum voice gain multiplier (Q16) = 2.0 (+6 dB).
pub const MAX_GAIN_Q16: i32 = 1 << 17;

/// Combined env*gain Q16 multiplier (saturating).
#[inline]
pub fn env_gain_multiplier(env_q16: i32, gain_q16: i32) -> i32 {
    mul_q16(env_q16, gain_q16)
}

/// Per-channel multiplier: `sat(rnd(e * pan_ch, 16))`.
#[inline]
pub fn channel_multiplier(e_q16: i32, pan_ch_q16: i32) -> i32 {
    mul_q16(e_q16, pan_ch_q16)
}

/// Full per-channel contribution: `sat(rnd(obs * mch, 16))` — the voice-bus
/// boundary (bounded by `2^31-1`; see the mixing proof in U1_SPEC).
#[inline]
pub fn contribution(obs: i32, mch_q16: i32) -> i32 {
    mul_q16(obs, mch_q16)
}

/// Validate a gain value against the frozen domain.
pub const fn checked_gain(gain_q16: i32) -> bool {
    gain_q16 >= -MAX_GAIN_Q16 && gain_q16 <= MAX_GAIN_Q16
}

/// Reference helper for the chain with explicit stages (used by tests and by
/// documentation; call sites should use the composed helpers).
pub fn chain_stages(obs: i32, env_q16: i32, gain_q16: i32, pan_ch_q16: i32) -> i32 {
    let e = env_gain_multiplier(env_q16, gain_q16);
    let m = channel_multiplier(e, pan_ch_q16);
    contribution(obs, m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_chain_is_identity() {
        for obs in [0i32, 1, -1, 123_456, -987_654, i32::MAX - 1, i32::MIN + 2] {
            assert_eq!(
                chain_stages(obs, 1 << 16, 1 << 16, 1 << 16),
                obs,
                "unity chain must be identity for {obs}"
            );
        }
    }

    #[test]
    fn half_gain_scales_exactly() {
        assert_eq!(chain_stages(100_000, 1 << 16, 1 << 15, 1 << 16), 50_000);
        assert_eq!(chain_stages(99_999, 1 << 16, 1 << 15, 1 << 16), 50_000);
        assert_eq!(chain_stages(-99_999, 1 << 16, 1 << 15, 1 << 16), -50_000);
    }

    #[test]
    fn gain_ceiling_and_saturation() {
        assert!(checked_gain(MAX_GAIN_Q16));
        assert!(!checked_gain(MAX_GAIN_Q16 + 1));
        // 1.5 * obs where obs near full scale saturates at the voice bus.
        assert_eq!(
            chain_stages(1_500_000_000, 1 << 16, 3 * (1 << 15), 1 << 16),
            i32::MAX
        );
    }

    #[test]
    fn multiplier_domain() {
        let e = env_gain_multiplier(1 << 16, 1 << 16);
        assert_eq!(e, 1 << 16);
        let m = channel_multiplier(e, 1 << 15);
        assert_eq!(m, 1 << 15);
    }
}

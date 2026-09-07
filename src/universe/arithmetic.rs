//! Exact fixed-point arithmetic — the normative u1 arithmetic core.
//!
//! Freeze record (U1_SPEC §"Exact arithmetic"). Every normative multiply/
//! reduce/shift below is spelled out; the code never depends on implicit
//! release/debug overflow differences. All operations are
//! `no_std`-clean and compile unchanged for GPU device targets.
//!
//! Conventions
//! -----------
//! * Q16 multipliers: gain, pan, envelope, amplitude. Unity is exactly
//!   `1 << 16`.
//! * Q24 positions/rates/fractions (`FIXED_Q = 24`). Unity rate is exactly
//!   `1 << 24`.
//! * Q30 frozen tables (sine): values in `[-(2^30-1), 2^30-1]`.
//! * Rounding: **round half away from zero**, applied exactly once at each
//!   defined reduce point (`rnd_shift`).
//! * Saturation: applied only at defined semantic boundaries (`sat_i32`,
//!   voice-bus, final output). The mix accumulator itself is pure i64
//!   addition — order independent, never saturated until the output boundary.
//! * Mixing bound: per-voice contribution is saturated to `|c| <= 2^31-1`
//!   before entering the i64 mix, so with `MAX_ACTIVE_VOICES = 4096`
//!   `|mix| <= 4096 * (2^31-1) < 2^43`, far inside i64 (proof in U1_SPEC).

use crate::limits::{FIXED_Q, GAIN_Q};

/// Round half away from zero: `v / 2^q` with ties moving away from zero.
///
/// Precondition: `|v| < 2^62` so the negation path cannot reach `i64::MIN`.
#[inline]
pub fn rnd_shift(v: i64, q: u32) -> i64 {
    debug_assert!((1..=62).contains(&q));
    debug_assert!(v.unsigned_abs() < (1u64 << 62), "rnd_shift input bound");
    let half = 1i64 << (q - 1);
    if v >= 0 {
        (v + half) >> q
    } else {
        -(((-v) + half) >> q)
    }
}

/// Saturating conversion to i32 (the only i32 saturation primitive).
#[inline]
pub fn sat_i32(v: i64) -> i32 {
    v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// Q16 multiply-reduce: `round(a*b / 2^16)` then saturate to i32.
///
/// This is the standard gain/pan/envelope application point
/// (`mul_q16(obs, multiplier)`); `multiplier = 1<<16` is exact unity.
#[inline]
pub fn mul_q16(a: i32, b_q16: i32) -> i32 {
    sat_i32(rnd_shift(i64::from(a) * i64::from(b_q16), GAIN_Q))
}

/// Saturating i32 add for **defined** voice-bus boundaries only (never for the
/// i64 mix accumulator).
#[inline]
pub fn add_sat_i32(a: i32, b: i32) -> i32 {
    sat_i32(i64::from(a) + i64::from(b))
}

/// Saturating i32 multiply by a Q16 multiplier with the product reduced and
/// saturated exactly once (alias of `mul_q16` used at call sites where the
/// semantic boundary is an amplitude, for readability).
#[inline]
pub fn amp_q16(obs: i32, amp_q16: i32) -> i32 {
    mul_q16(obs, amp_q16)
}

/// Fixed-point position step: advance a Q24 position by a Q24 rate over
/// `frames` output frames (wrapping add — callers define the loop domain).
#[inline]
pub fn advance_q24(pos_q24: i64, rate_q24: i64, frames: i64) -> i64 {
    pos_q24.wrapping_add(rate_q24.wrapping_mul(frames))
}

/// Split a Q24 position into integer frame index and a 24-bit fraction.
#[inline]
pub fn split_q24(pos_q24: i64) -> (i64, u32) {
    let idx = pos_q24 >> FIXED_Q;
    let frac = (pos_q24 as u64 & ((1u64 << FIXED_Q) - 1)) as u32;
    (idx, frac)
}

/// Linear interpolation between two samples at a 24-bit fraction:
/// `a + round((b-a) * frac / 2^24)` (integer-exact, round half away from
/// zero). `a`/`b` are i32 sample codes widened to i64.
#[inline]
pub fn lerp_i32(a: i32, b: i32, frac_q24: u32) -> i32 {
    let d = i64::from(b) - i64::from(a);
    let step = rnd_shift(d * i64::from(frac_q24), FIXED_Q);
    sat_i32(i64::from(a) + step)
}

/// Oscillator-table linear interpolation (identical rule to `lerp_i32`).
#[inline]
pub fn lerp_table(a: i32, b: i32, frac_q24: u32) -> i32 {
    lerp_i32(a, b, frac_q24)
}

/// Convert a nominal frequency in Hz to a Q24 per-frame rate increment at the
/// given nominal sample rate: `round(freq * 2^24 / rate)`. Frequencies above
/// Nyquist (`rate/2`) are clamped to Nyquist and reported by the caller if it
/// needs to distinguish (sampler `rate` transforms also use this for pitch).
#[inline]
pub fn hz_to_rate_q24(freq_hz: u32, rate_hz: u32) -> i64 {
    debug_assert!(rate_hz > 0);
    let f = freq_hz.min(rate_hz / 2).max(1) as i64;
    let num = f << FIXED_Q;
    let den = i64::from(rate_hz);
    (num + den / 2) / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rnd_shift_half_away_from_zero() {
        // q=2: values/4.
        assert_eq!(rnd_shift(5, 2), 1); // 1.25
        assert_eq!(rnd_shift(6, 2), 2); // 1.5  -> 2
        assert_eq!(rnd_shift(7, 2), 2); // 1.75
        assert_eq!(rnd_shift(-5, 2), -1);
        assert_eq!(rnd_shift(-6, 2), -2); // -1.5 -> -2
        assert_eq!(rnd_shift(-7, 2), -2);
        assert_eq!(rnd_shift(4, 2), 1);
        assert_eq!(rnd_shift(-4, 2), -1);
        assert_eq!(rnd_shift(1, 1), 1); // 0.5 -> 1
        assert_eq!(rnd_shift(-1, 1), -1); // -0.5 -> -1
        assert_eq!(rnd_shift(0, 16), 0);
    }

    #[test]
    fn rnd_shift_q24_ties() {
        // 0.5 in q24 -> rounds to 1 (half away from zero).
        assert_eq!(rnd_shift(1 << 23, 24), 1);
        // -0.5 -> -1
        assert_eq!(rnd_shift(-(1 << 23), 24), -1);
        // 1.5 -> 2
        assert_eq!(rnd_shift((1 << 24) + (1 << 23), 24), 2);
        // -1.5 -> -2
        assert_eq!(rnd_shift(-((1 << 24) + (1 << 23)), 24), -2);
        // just below 0.5 -> 0
        assert_eq!(rnd_shift((1 << 23) - 1, 24), 0);
    }

    #[test]
    fn saturating_conversion() {
        assert_eq!(sat_i32(0), 0);
        assert_eq!(sat_i32(i64::from(i32::MAX)), i32::MAX);
        assert_eq!(sat_i32(i64::from(i32::MIN)), i32::MIN);
        assert_eq!(sat_i32(1 << 40), i32::MAX);
        assert_eq!(sat_i32(-(1 << 40)), i32::MIN);
    }

    #[test]
    fn q16_unity_is_identity() {
        for v in [0, 1, -1, 12345, -999999, i32::MAX, i32::MIN] {
            assert_eq!(mul_q16(v, 1 << 16), v, "unity multiplier must be identity");
        }
    }

    #[test]
    fn q16_half_scale() {
        // 100000 * 0.5
        assert_eq!(mul_q16(100_000, 1 << 15), 50_000);
        assert_eq!(mul_q16(-100_000, 1 << 15), -50_000);
        // odd value ties: 99999 * 0.5 = 49999.5 -> 50000 (half away)
        assert_eq!(mul_q16(99_999, 1 << 15), 50_000);
        assert_eq!(mul_q16(-99_999, 1 << 15), -50_000);
    }

    #[test]
    fn q16_max_gain_saturates_voice_bus() {
        // obs * 2.0 must saturate at i32::MAX (never wrap).
        assert_eq!(mul_q16(1_100_000_000, 1 << 17), i32::MAX);
        assert_eq!(mul_q16(-1_100_000_000, 1 << 17), i32::MIN);
        assert_eq!(mul_q16(i32::MAX, 1 << 17), i32::MAX);
        assert_eq!(mul_q16(i32::MIN, 1 << 17), i32::MIN);
        // 1e9 * 2.0 = 2e9 < i32::MAX -> no saturation.
        assert_eq!(mul_q16(1_000_000_000, 1 << 17), 2_000_000_000);
    }

    #[test]
    fn lerp_half_step() {
        // a=0,b=100,frac=0.5 -> 50
        assert_eq!(lerp_i32(0, 100, 1 << 23), 50);
        // a=0,b=1, frac just under 0.5 -> 0
        assert_eq!(lerp_i32(0, 1, (1 << 23) - 1), 0);
        // a=0,b=1, frac = 0.5 -> 1 (round half away)
        assert_eq!(lerp_i32(0, 1, 1 << 23), 1);
        // negative span: a=100,b=0 frac 0.5 -> 50
        assert_eq!(lerp_i32(100, 0, 1 << 23), 50);
        // a=0,b=-1, frac 0.5 -> -1 (round half away from zero)
        assert_eq!(lerp_i32(0, -1, 1 << 23), -1);
    }

    #[test]
    fn lerp_extremes_do_not_overflow() {
        // Oracle: exact rational lerp in i128, then clamp.
        let oracle = |a: i32, b: i32, f: u32| -> i32 {
            let num = i128::from(b) - i128::from(a);
            let prod = num * i128::from(f);
            let half = 1i128 << 23;
            let q = if prod >= 0 {
                (prod + half) >> 24
            } else {
                -(((-prod) + half) >> 24)
            };
            let v = i128::from(a) + q;
            v.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
        };
        for &(a, b) in &[(i32::MIN, i32::MAX), (i32::MAX, i32::MIN)] {
            for f in [
                0u32,
                1,
                (1 << 23) - 1,
                1 << 23,
                (1 << 23) + 1,
                (1 << 24) - 2,
                (1 << 24) - 1,
            ] {
                assert_eq!(lerp_i32(a, b, f), oracle(a, b, f));
            }
        }
        // Exact known edges: lerp(min, max, 1-2^-24) = max - 256
        // (d = 2^32-1; round((d*(2^24-1))/2^24) = d - 256).
        assert_eq!(lerp_i32(i32::MIN, i32::MAX, 0), i32::MIN);
        assert_eq!(lerp_i32(i32::MIN, i32::MAX, (1 << 24) - 1), i32::MAX - 256);
        assert_eq!(lerp_i32(i32::MAX, i32::MIN, (1 << 24) - 1), i32::MIN + 256);
    }

    #[test]
    fn hz_rate_conversion_rounds() {
        // 1 Hz at 48k: 2^24/48000 = 349.525... -> 350
        assert_eq!(hz_to_rate_q24(1, 48_000), 350);
        // 48k at 48k clamps to Nyquist 24k: 2^24/2 = 2^23
        assert_eq!(hz_to_rate_q24(48_000, 48_000), 1 << 23);
        // 24k at 48k: exactly half-rate -> 2^23
        assert_eq!(hz_to_rate_q24(24_000, 48_000), 1 << 23);
        // 12k at 48k = 0.25 -> 2^22
        assert_eq!(hz_to_rate_q24(12_000, 48_000), 1 << 22);
    }

    #[test]
    fn split_q24_matches_shift_math() {
        let pos = (1234 << 24) | 0x00AB_CDEF;
        let (idx, frac) = split_q24(pos);
        assert_eq!(idx, 1234);
        assert_eq!(frac, 0x00AB_CDEF);
        let (idx, frac) = split_q24(-((1 << 24) + 5));
        assert_eq!(idx, -2); // floor(-1.0000003) = -2
        assert_eq!(frac, (1 << 24) - 5);
    }
}

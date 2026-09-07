//! Rate/position semantics (frozen).
//!
//! Playback position is Q24 signed, advanced analytically:
//!
//! ```text
//! u(t) = p0 + rate_q24 * (t - t0)        // Q24 source coordinate
//! ```
//!
//! * `rate_q24` is signed: positive = forward, negative = reverse, zero =
//!   stationary. Magnitude domain: `0` or `[MIN_RATE_Q24, MAX_RATE_Q24]`.
//! * One-shot: audible while `0 <= u < extent<<24`; the natural end frame is
//!   computed exactly (see `end_frame`).
//! * Loop: the read coordinate wraps with the *single* mapping
//!   `w = A + ((u - A) mod L)` for both directions (Euclidean modulo), so
//!   reverse tape looping is just negative rate. `A`, `L` are Q24.
//! * Object extents and loop bounds are limited to `MAX_OBJECT_FRAMES`, and
//!   observation windows to `MAX_OBSERVATION_FRAMES`, so position arithmetic
//!   provably cannot overflow i64 (bounds in `limits.rs`).

use crate::limits::{FIXED_Q, MAX_OBJECT_FRAMES};

/// Minimum nonzero |rate|: 1/256 frames per frame.
pub const MIN_RATE_Q24: i64 = 1 << 8;
/// Maximum |rate|: 65536 frames per frame.
pub const MAX_RATE_Q24: i64 = 1 << 40;

/// Euclidean modulo for Q24 coordinates: result in `[0, m)`.
#[inline]
pub fn euclid_mod_q24(v: i64, m: i64) -> i64 {
    debug_assert!(m > 0);
    let r = v % m;
    if r < 0 { r + m } else { r }
}

/// Wrap a Q24 coordinate into loop region `[a_frame, b_frame)`.
#[inline]
pub fn wrap_loop_q24(u: i64, a_frame: u64, b_frame: u64) -> i64 {
    debug_assert!(b_frame > a_frame && b_frame <= MAX_OBJECT_FRAMES);
    let a = (a_frame << FIXED_Q) as i64;
    let len = ((b_frame - a_frame) << FIXED_Q) as i64;
    a + euclid_mod_q24(u - a, len)
}

/// True if a rate is inside the frozen domain.
pub const fn checked_rate(rate_q24: i64) -> bool {
    rate_q24 == 0
        || (rate_q24.unsigned_abs() >= MIN_RATE_Q24 as u64
            && rate_q24.unsigned_abs() <= MAX_RATE_Q24 as u64)
}

/// Natural end frame (media frames after `t0`) for a one-shot voice with no
/// loop: the first frame `t` at which `p0 + rate*(t-t0)` leaves
/// `[0, extent)`.
///
/// Forward: `t_end = t0 + ceil((extent<<24 - p0) / rate)`.
/// Reverse: `t_end = t0 + floor(p0 / -rate) + 1`.
/// Zero rate: no natural end (`None`).
pub fn one_shot_end_frame(p0_q24: i64, rate_q24: i64, extent_frames: u64, t0: i64) -> Option<i64> {
    debug_assert!(extent_frames <= MAX_OBJECT_FRAMES);
    if rate_q24 == 0 {
        return None;
    }
    let extent = (extent_frames << FIXED_Q) as i64;
    if rate_q24 > 0 {
        // p0 + rate*d >= extent  =>  d >= ceil((extent - p0)/rate)
        if p0_q24 >= extent {
            return Some(t0);
        }
        let num = extent - p0_q24;
        let d = (num + rate_q24 - 1) / rate_q24; // ceil for positive terms
        t0.checked_add(d)
    } else {
        // p0 + rate*d < 0 with rate<0 => d > p0/(-rate); first integer = floor(p0/(-rate)) + 1
        let step = -rate_q24;
        if p0_q24 < 0 {
            return Some(t0);
        }
        let d = p0_q24 / step + 1; // floor division is exact for p0 >= 0
        t0.checked_add(d)
    }
}

/// Advance a Q24 position by `rate` over `frames` output frames
/// (saturating; the frozen bounds make saturation unreachable).
#[inline]
pub fn advance(rate_q24: i64, frames: i64) -> i64 {
    rate_q24.wrapping_mul(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FR: i64 = 1 << FIXED_Q; // one frame in Q24

    #[test]
    fn wrap_loop_forward_and_reverse() {
        // Loop [2, 5) i.e. frames 2,3,4.
        let (a, b) = (2u64, 5u64);
        assert_eq!(wrap_loop_q24(2 * FR, a, b), 2 * FR);
        assert_eq!(wrap_loop_q24(4 * FR + FR / 2, a, b), 4 * FR + FR / 2);
        // Just at the end wraps to start.
        assert_eq!(wrap_loop_q24(5 * FR, a, b), 2 * FR);
        assert_eq!(wrap_loop_q24(6 * FR, a, b), 3 * FR);
        // Reverse overshoot below start wraps to the end.
        assert_eq!(wrap_loop_q24(2 * FR - 1, a, b), 5 * FR - 1);
        // Far below the region the Euclidean wrap continues the backward tape
        // sequence 4,3,2,4,3,...: u=-3 -> w=3.
        assert_eq!(wrap_loop_q24(-3 * FR, a, b), 3 * FR);
        assert_eq!(wrap_loop_q24(-4 * FR, a, b), 2 * FR);
        assert_eq!(wrap_loop_q24(-5 * FR, a, b), 4 * FR);
    }

    #[test]
    fn euclid_mod_is_nonnegative() {
        assert_eq!(euclid_mod_q24(-1, 3), 2);
        assert_eq!(euclid_mod_q24(-4, 3), 2);
        assert_eq!(euclid_mod_q24(4, 3), 1);
        assert_eq!(euclid_mod_q24(0, 3), 0);
    }

    #[test]
    fn one_shot_forward_end() {
        // extent 10, start 0, rate 1 frame/frame -> ends at t0+10.
        assert_eq!(one_shot_end_frame(0, FR, 10, 100), Some(110));
        // start at 5 -> 5 frames later.
        assert_eq!(one_shot_end_frame(5 * FR, FR, 10, 0), Some(5));
        // rate 0.5 -> 20 frames for extent 10.
        assert_eq!(one_shot_end_frame(0, FR / 2, 10, 0), Some(20));
        // fractional: start 0, rate 3, extent 10 -> ceil(10/3)=4.
        assert_eq!(one_shot_end_frame(0, 3 * FR, 10, 0), Some(4));
        // Already past the end: ends immediately.
        assert_eq!(one_shot_end_frame(10 * FR, FR, 10, 7), Some(7));
        // Zero rate: infinite.
        assert_eq!(one_shot_end_frame(0, 0, 10, 0), None);
    }

    #[test]
    fn one_shot_reverse_end() {
        // extent 10, start at frame 8, rate -1: frames 8..0 audible (9
        // frames), dead from t0+9.
        assert_eq!(one_shot_end_frame(8 * FR, -FR, 10, 0), Some(9));
        // rate -2 from frame 8: audible d in 0..=4, dead at 5.
        assert_eq!(one_shot_end_frame(8 * FR, -2 * FR, 10, 0), Some(5));
        // start at 0 with reverse: frame 0 audible only at t0, dead at t0+1.
        assert_eq!(one_shot_end_frame(0, -FR, 10, 3), Some(4));
    }

    #[test]
    fn rate_domain_validation() {
        assert!(checked_rate(0));
        assert!(checked_rate(1 << 8));
        assert!(checked_rate(-(1 << 8)));
        assert!(checked_rate(1 << 40));
        assert!(!checked_rate((1 << 8) - 1));
        assert!(!checked_rate((1 << 40) + 1));
        assert!(!checked_rate(-((1 << 40) + 1)));
    }
}

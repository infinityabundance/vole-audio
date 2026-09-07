//! Object-domain interpolation (frozen).
//!
//! Reads operate on a single channel plane (contiguous samples `[0, extent)`).
//! A Q24 read coordinate `w` selects a frame; continuation rules:
//!
//! * one-shot: `w` must satisfy `0 <= w < extent<<24` (the caller's voice-end
//!   logic guarantees it); interpolation at the last frame holds the final
//!   sample (`b = a`), no wrap.
//! * loop `[a, b)`: the coordinate is pre-wrapped by `rate::wrap_loop_q24`;
//!   when `idx0 == b-1` the interpolation neighbor wraps to `a` (periodic
//!   continuation of the loop region).
//!
//! Nearest: `plane[round(w)]` with the same continuation rule (ties round
//! half up in Q24, i.e. `frac >= 2^23` steps to the next index).

use crate::universe::arithmetic::lerp_i32;
use crate::universe::time::Frames;

/// Read mode for a channel plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    /// One-shot over the full extent (hold at the final sample).
    OneShot(Frames),
    /// Loop over `[start, end)` object frames.
    Loop { start: u64, end: u64 },
}

impl ReadMode {
    pub fn extent_frames(&self) -> u64 {
        match self {
            ReadMode::OneShot(f) => f.to_i64().max(0) as u64,
            ReadMode::Loop { end, .. } => *end,
        }
    }
}

/// Resolve `(idx0, frac24, idx1)` for a Q24 coordinate under a read mode.
/// `idx1` already applies the continuation rule (wrap to loop start, or hold
/// at the last frame).
#[inline]
pub fn resolve_indices(mode: &ReadMode, w_q24: i64) -> (usize, u32, usize) {
    let frac = (w_q24 as u64 & 0xFF_FFFF) as u32;
    match mode {
        ReadMode::OneShot(ext) => {
            let last = (ext.to_i64().max(0) as usize).saturating_sub(1);
            let i0 = (w_q24 >> 24) as usize;
            let i0 = i0.min(last);
            let i1 = if i0 < last { i0 + 1 } else { i0 };
            (i0, frac, i1)
        }
        ReadMode::Loop { start, end } => {
            let a = *start as usize;
            let b = *end as usize;
            let i0 = ((w_q24 >> 24) as usize).min(b - 1);
            let i1 = if i0 + 1 >= b { a } else { i0 + 1 };
            (i0, frac, i1)
        }
    }
}

/// Linear read of a channel plane at Q24 coordinate.
#[inline]
pub fn read_linear(plane: &[i32], mode: &ReadMode, w_q24: i64) -> i32 {
    let (i0, frac, i1) = resolve_indices(mode, w_q24);
    lerp_i32(plane[i0], plane[i1], frac)
}

/// Nearest read of a channel plane at a Q24 coordinate (ties step up).
#[inline]
pub fn read_nearest(plane: &[i32], mode: &ReadMode, w_q24: i64) -> i32 {
    let (i0, frac, i1) = resolve_indices(mode, w_q24);
    if frac >= (1 << 23) {
        plane[i1]
    } else {
        plane[i0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::universe::time::Frames;

    fn plane() -> Vec<i32> {
        vec![0, 10, 20, 30, 40] // extent 5
    }

    #[test]
    fn linear_holds_at_last_frame_one_shot() {
        let p = plane();
        let m = ReadMode::OneShot(Frames(5));
        let f = 1i64 << 24;
        assert_eq!(read_linear(&p, &m, 0), 0);
        assert_eq!(read_linear(&p, &m, f), 10);
        assert_eq!(read_linear(&p, &m, f + (1 << 23)), 15); // halfway 10->20
        // last frame holds: 4.5 -> 40 (idx1 == idx0).
        assert_eq!(read_linear(&p, &m, 4 * f + (1 << 23)), 40);
        assert_eq!(read_linear(&p, &m, 4 * f + (1 << 24) - 1), 40);
    }

    #[test]
    fn nearest_steps_half_up() {
        let p = plane();
        let m = ReadMode::OneShot(Frames(5));
        let f = 1i64 << 24;
        assert_eq!(read_nearest(&p, &m, f + (1 << 23) - 1), 10);
        assert_eq!(read_nearest(&p, &m, f + (1 << 23)), 20);
        assert_eq!(read_nearest(&p, &m, f + (1 << 23) + 1), 20);
    }

    #[test]
    fn loop_continuation_wraps_to_start() {
        let p = plane();
        // Loop [1, 4): frames 1..3 (10, 20, 30); extent of reads within.
        let m = ReadMode::Loop { start: 1, end: 4 };
        let f = 1i64 << 24;
        // coordinate 3.5 (idx0=3 == b-1): neighbor is a=1 -> interp 30->10.
        // Pre-wrap is the caller's job; resolve must wrap the *neighbor*.
        let w = crate::sampler::rate::wrap_loop_q24(3 * f + (1 << 23), 1, 4);
        assert_eq!(read_linear(&p, &m, w), 20);
        // Directly at the boundary index (idx0 = 3 == end-1) wraps b to a.
        let v = read_linear(&p, &m, 3 * f + (1 << 23));
        assert_eq!(v, 20); // 0.5*(30+10)
        // At loop start: idx0 = 1, frac 0 -> 10.
        assert_eq!(read_linear(&p, &m, f), 10);
    }

    #[test]
    fn resolve_indices_holds_last() {
        let m = ReadMode::OneShot(Frames(1));
        let (i0, frac, i1) = resolve_indices(&m, 0);
        assert_eq!((i0, frac, i1), (0, 0, 0));
    }
}

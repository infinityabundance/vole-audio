//! Pan law (frozen): linear equal-gain.
//!
//! `pan_q16` spans `[-1<<16, 1<<16]` (-1 hard left, +1 hard right). Per
//! channel gains (Q16):
//!
//! ```text
//! L = (1<<16 - pan) >> 1
//! R = (1<<16 + pan) >> 1
//! ```
//!
//! Center (pan=0) yields -6 dB per channel (equal-gain linear law).
//! `RoutedPan` maps a voice to a mono channel or to a stereo pair.

/// Compute the left/right Q16 gains for a pan value.
///
/// Exactness rule: `L + R == unity` for every pan (equal-gain). When the
/// halving is odd the extra unit goes to the louder side.
#[inline]
pub fn pan_gains(pan_q16: i32) -> (i32, i32) {
    let unity = 1i32 << 16;
    debug_assert!((-unity..=unity).contains(&pan_q16));
    let s = unity - pan_q16;
    let t = unity + pan_q16;
    let mut l = s >> 1;
    let mut r = t >> 1;
    // s + t == 2*unity (even), so s and t share parity; when odd, the split
    // loses one unit unless we give it to the louder side.
    if (s & 1) == 1 {
        if s > t {
            l += 1;
        } else {
            r += 1;
        }
    }
    (l, r)
}

/// Voice routing target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Route {
    /// Direct to one output channel (no pan).
    Mono(u8),
    /// Pan across output channels `(base, base + 1)`.
    StereoPair(u8),
}

impl Route {
    pub const fn checked_mono(ch: u8) -> Option<Route> {
        if (ch as u32) < crate::limits::MAX_CHANNELS {
            Some(Route::Mono(ch))
        } else {
            None
        }
    }

    pub const fn checked_stereo(base: u8) -> Option<Route> {
        if (base as u32 + 1) < crate::limits::MAX_CHANNELS {
            Some(Route::StereoPair(base))
        } else {
            None
        }
    }
}

/// Validate a pan value.
pub const fn checked_pan(pan_q16: i32) -> bool {
    pan_q16 >= -(1 << 16) && pan_q16 <= (1 << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pan_extremes_and_center() {
        let unity = 1 << 16;
        // Hard left: L full, R zero.
        let (l, r) = pan_gains(-unity);
        assert_eq!(l, unity);
        assert_eq!(r, 0);
        // Hard right.
        let (l, r) = pan_gains(unity);
        assert_eq!(l, 0);
        assert_eq!(r, unity);
        // Center: half each.
        let (l, r) = pan_gains(0);
        assert_eq!(l, unity >> 1);
        assert_eq!(r, unity >> 1);
        // Quarter right.
        let (l, r) = pan_gains(unity >> 1);
        assert_eq!(l, unity >> 2);
        assert_eq!(r, 3 * (unity >> 2));
    }

    #[test]
    fn pan_gains_are_exact_shifts() {
        for p in [-65536, -65535, -1, 0, 1, 65535, 65536] {
            let (l, r) = pan_gains(p);
            assert_eq!(l + r, 1 << 16, "equal-gain: L+R == unity for p={p}");
            assert!(l >= 0 && r >= 0);
        }
    }

    #[test]
    fn routing_validation() {
        assert!(Route::checked_mono(0).is_some());
        assert!(Route::checked_mono(31).is_some());
        assert!(Route::checked_mono(32).is_none());
        assert!(Route::checked_stereo(0).is_some());
        assert!(Route::checked_stereo(30).is_some());
        assert!(Route::checked_stereo(31).is_none());
    }

    #[test]
    fn pan_validation() {
        assert!(checked_pan(1 << 16));
        assert!(checked_pan(-(1 << 16)));
        assert!(!checked_pan((1 << 16) + 1));
    }
}

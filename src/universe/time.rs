//! Time model — frame/epoch coordinates and the time axes that must never be
//! conflated.
//!
//! u1 keeps these separate (U1_SPEC §"Time, rate, phase and event
//! semantics"):
//!
//! * logical media time (frames since the media epoch — the sampler timeline);
//! * object intrinsic frame coordinate (frames since the start of a
//!   SampleObject's extent);
//! * endpoint frame coordinate (frames consumed by the physical endpoint);
//! * nominal sample rate (integer Hz per profile);
//! * physical endpoint clock (host instrumentation, measured — see
//!   `evidence::timing` and the ALSA clock probe);
//! * wall-clock instrumentation time (monotonic, never authoritative media
//!   time).
//!
//! The authoritative representation is **integer frames**, not `f64`
//! seconds. No conversion function in this module rounds implicitly; host
//! frame<->clock conversions live in `clock.rs` with explicit documented
//! rounding.

use core::fmt;
use core::ops::{Add, Sub};

/// A frame count or frame offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Frames(pub i64);

/// Logical media frame coordinate: frames since the start of a media epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MediaFrame(pub i64);

/// Endpoint frame coordinate: frames consumed by the physical endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct EndpointFrame(pub i64);

/// Object intrinsic frame coordinate (signed; bounds are enforced elsewhere).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ObjectFrame(pub i64);

/// Media epoch id. Incremented when a discontinuity policy restarts time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EpochId(pub u32);

impl EpochId {
    pub const ZERO: EpochId = EpochId(0);
    pub const fn next(self) -> EpochId {
        EpochId(self.0.wrapping_add(1))
    }
}

impl Frames {
    pub const ZERO: Frames = Frames(0);
    pub const fn new(v: i64) -> Self {
        Self(v)
    }
    pub const fn to_i64(self) -> i64 {
        self.0
    }
}

impl MediaFrame {
    pub const ZERO: MediaFrame = MediaFrame(0);
    pub const fn new(v: i64) -> Self {
        Self(v)
    }
    pub const fn to_i64(self) -> i64 {
        self.0
    }
}

impl EndpointFrame {
    pub const ZERO: EndpointFrame = EndpointFrame(0);
    pub const fn new(v: i64) -> Self {
        Self(v)
    }
    pub const fn to_i64(self) -> i64 {
        self.0
    }
}

impl ObjectFrame {
    pub const ZERO: ObjectFrame = ObjectFrame(0);
    pub const fn new(v: i64) -> Self {
        Self(v)
    }
    pub const fn to_i64(self) -> i64 {
        self.0
    }
}

macro_rules! frame_arith {
    ($t:ty) => {
        impl Add for $t {
            type Output = $t;
            fn add(self, o: Self) -> Self {
                <$t>::new(self.0.wrapping_add(o.0))
            }
        }
        impl Sub for $t {
            type Output = $t;
            fn sub(self, o: Self) -> Self {
                <$t>::new(self.0.wrapping_sub(o.0))
            }
        }
        impl Add<i64> for $t {
            type Output = $t;
            fn add(self, o: i64) -> Self {
                <$t>::new(self.0.wrapping_add(o))
            }
        }
        impl Sub<i64> for $t {
            type Output = $t;
            fn sub(self, o: i64) -> Self {
                <$t>::new(self.0.wrapping_sub(o))
            }
        }
    };
}

frame_arith!(Frames);
frame_arith!(MediaFrame);
frame_arith!(EndpointFrame);
frame_arith!(ObjectFrame);

macro_rules! frame_display {
    ($t:ty) => {
        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

frame_display!(Frames);
frame_display!(MediaFrame);
frame_display!(EndpointFrame);
frame_display!(ObjectFrame);

/// Nominal sample rate: integer Hz. u1 freezes this as a plain integer on the
/// `u1` profile; profiles with other rates keep the same semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NominalRate(pub u32);

impl NominalRate {
    pub const fn new(hz: u32) -> Self {
        Self(hz)
    }
    pub const fn to_hz(self) -> u32 {
        self.0
    }
    /// Frames per `secs` computed without floats (used only by tooling; the
    /// authoritative media timeline is frames).
    pub fn frames_for_micros(self, micros: u64) -> u64 {
        (micros * u64::from(self.0) + 500_000) / 1_000_000
    }
}

impl Default for NominalRate {
    fn default() -> Self {
        Self(crate::universe::u1::NOMINAL_RATE_HZ)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_arithmetic_is_wrapping_and_typed() {
        let a = MediaFrame::new(100);
        let b = MediaFrame::new(25);
        assert_eq!((a - b).to_i64(), 75);
        assert_eq!((a + 10).to_i64(), 110);
        assert_eq!((a - 200).to_i64(), -100);
    }

    #[test]
    fn types_do_not_mix() {
        // Distinct newtypes: this would not compile if MediaFrame and
        // EndpointFrame were interchangeable (compile-time guarantee).
        let _m: MediaFrame = MediaFrame::ZERO;
        let _e: EndpointFrame = EndpointFrame::ZERO;
    }

    #[test]
    fn epoch_advance() {
        assert_eq!(EpochId::ZERO.next(), EpochId(1));
    }

    #[test]
    fn rate_frames_conversion_rounds_half_up() {
        let r = NominalRate::new(48_000);
        assert_eq!(r.frames_for_micros(1_000_000), 48_000);
        assert_eq!(r.frames_for_micros(500_000), 24_000);
        // 10.416666 ms -> 500 frames exactly at 48k
        assert_eq!(r.frames_for_micros(10_417), 500);
    }
}

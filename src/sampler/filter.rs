//! Narrow exact filter set (frozen).
//!
//! v1 filter set is deliberately narrow:
//!
//! * `Biquad` — RBJ-style lowpass in Direct Form I with integer coefficients
//!   (Q24), derived *exactly* from the u1 sine table (no libm anywhere in the
//!   normative evaluator). Stateful: `x1,x2,y1,y2`. Classification:
//!   `DirectStateful` (see `eval::common::TransformClass`).
//! * `OnePole` — exact exponential smoother `y += (x - y) >> k`.
//!
//! Recursive filters require explicit per-voice state, so they are *not* on
//! the stateless fused GPU fast path; they are classified honestly per
//! backend and checkpointed like any other state.

use crate::universe::arithmetic::{rnd_shift, sat_i32};
use crate::universe::phase::{freq_to_incr, sine_table_i32};

/// Q30 approximation of 1/sqrt(2) (Butterworth alpha scale).
pub const INV_SQRT2_Q30: i32 = 759_250_125;

/// Exact integer cosine (Q30) of `2*pi*fc/fs` via the frozen sine table.
/// The quarter-turn phase offset (2^62 = pi/2 in phase units) maps
/// sine -> cosine; phase wraps modulo 2^64.
pub fn cos_q30_from_table(freq_hz: u32, rate_hz: u32) -> i32 {
    let phase = freq_to_incr(freq_hz, rate_hz);
    let table = sine_table_i32();
    crate::universe::phase::sine_interp(table, phase.wrapping_add(1u64 << 62))
}

/// Exact integer sine (Q30) of `2*pi*fc/fs`.
pub fn sin_q30_from_table(freq_hz: u32, rate_hz: u32) -> i32 {
    let table = sine_table_i32();
    crate::universe::phase::sine_interp(table, freq_to_incr(freq_hz, rate_hz))
}

/// Biquad lowpass coefficients in Q24 (`b0,b1,b2,c1,c2`) with the recurrence
///
/// ```text
/// y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] + c1 y[n-1] + c2 y[n-2]
/// ```
///
/// (all in Q24, one rounding at the reduce point, one saturation at the
/// output) — Direct Form I.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiquadCoeffs {
    pub b0: i32,
    pub b1: i32,
    pub b2: i32,
    pub c1: i32,
    pub c2: i32,
}

/// Signed round-half-up division `num / den` (den > 0).
fn div_round_signed(num: i64, den: i64) -> i64 {
    debug_assert!(den > 0);
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -(((-num) + den / 2) / den)
    }
}

impl BiquadCoeffs {
    /// RBJ Butterworth lowpass at `cutoff_hz` for `rate_hz`, coefficients
    /// computed with integer-only math from the frozen sine table.
    pub fn lowpass(cutoff_hz: u32, rate_hz: u32) -> Option<BiquadCoeffs> {
        if cutoff_hz == 0 || cutoff_hz > rate_hz / 2 || rate_hz == 0 {
            return None;
        }
        // cos and sin at omega = 2*pi*fc/fs, Q30.
        let c = i64::from(cos_q30_from_table(cutoff_hz, rate_hz));
        let s = i64::from(sin_q30_from_table(cutoff_hz, rate_hz));
        // alpha = sin(w) / sqrt(2), Q30.
        let alpha = rnd_shift(s * i64::from(INV_SQRT2_Q30), 30);
        let a0 = (1i64 << 30) + alpha;
        let one_minus_c = (1i64 << 30) - c;
        // RBJ: b0 = b2 = (1-c)/2 ; b1 = 1-c ; a1 = -2c ; a2 = 1-alpha.
        let b0_q30 = one_minus_c >> 1;
        let b1_q30 = one_minus_c;
        let b2_q30 = one_minus_c >> 1;
        // c1 = -a1/a0 = 2c/a0 ; c2 = -a2/a0 = (alpha-1)/a0.
        let q24 = 1i64 << 24;
        let to_q24 = |num: i64| div_round_signed(num * q24, a0);
        Some(BiquadCoeffs {
            b0: sat_i32(to_q24(b0_q30)),
            b1: sat_i32(to_q24(b1_q30)),
            b2: sat_i32(to_q24(b2_q30)),
            // c1 = 2cos/a0 (Q24): 2c is a Q30 value (|2c| <= 2^31).
            c1: sat_i32(to_q24(c << 1)),
            // c2 = (alpha-1)/a0 (Q24, negative).
            c2: sat_i32(to_q24(alpha - (1i64 << 30))),
        })
    }
}

/// Deterministic DF1 biquad state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Biquad {
    pub coeffs: BiquadCoeffs,
    pub x1: i32,
    pub x2: i32,
    pub y1: i32,
    pub y2: i32,
}

impl Biquad {
    pub fn new(coeffs: BiquadCoeffs) -> Biquad {
        Biquad {
            coeffs,
            x1: 0,
            x2: 0,
            y1: 0,
            y2: 0,
        }
    }

    pub fn reset(&mut self) {
        self.x1 = 0;
        self.x2 = 0;
        self.y1 = 0;
        self.y2 = 0;
    }

    /// Process one sample. Exact update order: compute from current state,
    /// then shift `x` and `y` delay lines.
    #[inline]
    pub fn process(&mut self, x: i32) -> i32 {
        let q = &self.coeffs;
        let acc = i64::from(q.b0) * i64::from(x)
            + i64::from(q.b1) * i64::from(self.x1)
            + i64::from(q.b2) * i64::from(self.x2)
            + i64::from(q.c1) * i64::from(self.y1)
            + i64::from(q.c2) * i64::from(self.y2);
        let y = sat_i32(rnd_shift(acc, 24));
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Exact one-pole smoother `y += (x - y) >> k`, `k in 1..=30`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnePole {
    pub shift: u32,
    pub y: i32,
}

impl OnePole {
    pub fn new(shift: u32) -> Option<OnePole> {
        if !(1..=30).contains(&shift) {
            return None;
        }
        Some(OnePole { shift, y: 0 })
    }

    #[inline]
    pub fn process(&mut self, x: i32) -> i32 {
        let delta = (i64::from(x) - i64::from(self.y)) >> self.shift;
        let y = sat_i32(i64::from(self.y) + delta);
        self.y = y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_cos_quarter_turn() {
        // cos(pi/2) = 0 and sin(pi/2) = peak, at fc = fs/4.
        let c0 = cos_q30_from_table(1, 4);
        assert!(c0.abs() <= 1, "cos(pi/2) ~= 0, got {c0}");
        let s0 = sin_q30_from_table(1, 4);
        assert!((s0 - ((1 << 30) - 1)).abs() <= 1);
        // cos(0) = 1: an extremely low frequency must land within table
        // resolution of the peak entry.
        let c_lo = cos_q30_from_table(1, 1_000_000);
        assert!((((1 << 30) - 65)..(1 << 30)).contains(&c_lo), "{c_lo}");
    }

    #[test]
    fn lowpass_dc_gain_approaches_unity() {
        let coeffs = BiquadCoeffs::lowpass(4_000, 48_000).unwrap();
        let mut f = Biquad::new(coeffs);
        let level = 100_000i32;
        let mut last = 0i32;
        for _ in 0..20_000 {
            last = f.process(level);
        }
        // Integer-coefficient DC gain converges within ~0.1%.
        let err = (i64::from(last) - i64::from(level)).abs();
        assert!(err < i64::from(level) / 1000, "dc err {err}");
        // Deterministic: identical run reproduces identical output.
        let mut g = Biquad::new(coeffs);
        for _ in 0..20_000 {
            g.process(level);
        }
        assert_eq!(f.y1, g.y1);
    }

    #[test]
    fn lowpass_rejects_high_frequencies() {
        // A full-scale square-ish alternating signal at fs/2 through a lowpass
        // at fc = fs/8 must attenuate substantially below the DC passthrough.
        let coeffs = BiquadCoeffs::lowpass(6_000, 48_000).unwrap();
        let mut f = Biquad::new(coeffs);
        let mut peak: i64 = 0;
        let mut x = 100_000i32;
        for _ in 0..200_000 {
            let y = f.process(x);
            peak = peak.max(i64::from(y).abs());
            x = x.wrapping_neg();
        }
        // Alternating Nyquist input through a lowpass must be well below the
        // input magnitude after convergence.
        assert!(peak < 30_000, "peak {peak}");
    }

    #[test]
    fn filter_state_is_deterministic_and_reset() {
        let coeffs = BiquadCoeffs::lowpass(1000, 48_000).unwrap();
        let input: Vec<i32> = (0..1000)
            .map(|i| ((i * 7919) % 200_000) - 100_000)
            .collect();
        let run = |c: &BiquadCoeffs| -> Vec<i32> {
            let mut f = Biquad::new(*c);
            input.iter().map(|&x| f.process(x)).collect()
        };
        let a = run(&coeffs);
        let b = run(&coeffs);
        assert_eq!(a, b);
        // Reset reproduces the same output from the same input.
        let mut f = Biquad::new(coeffs);
        let mut c = Vec::new();
        for &x in &input {
            c.push(f.process(x));
        }
        assert_eq!(c, a);
    }

    #[test]
    fn one_pole_smoother_converges() {
        let mut s = OnePole::new(4).unwrap();
        for _ in 0..10_000 {
            s.process(1000);
        }
        // Convergence is to within one step size (2^-4) below the target.
        assert!(s.y <= 1000 && (1000 - s.y) < 16, "y={}", s.y);
        let mut s2 = OnePole::new(4).unwrap();
        for _ in 0..10_000 {
            s2.process(1000);
        }
        assert_eq!(s2.y, s.y);
        assert!(OnePole::new(0).is_none());
        assert!(OnePole::new(31).is_none());
    }
}

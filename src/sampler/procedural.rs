//! Procedural generator evaluation semantics (frozen, `no_std`).
//!
//! These functions are the *exact* observation semantics of the procedural
//! SampleObject classes. They take plain parameters (no heap, no object
//! store) so the same code compiles unchanged for GPU device kernels; host
//! code and kernels both call them with the same values.
//!
//! Freeze record (U1_SPEC §"Procedural generators"):
//!
//! * Oscillator/partial-bank phase advances **modulo 2^64** with a frozen
//!   per-output-frame increment. The increment is derived host-side from the
//!   object's base frequency and the voice's effective rate:
//!
//!   ```text
//!   eff_incr = clamp(round(2^64 * f0 * rate_eff / 2^24 / fs), 2^63)
//!   ```
//!
//!   (clamped at 2^63 = Nyquist of the phase domain). Partial `k` advances at
//!   `(k * eff_incr) mod 2^64` — exact integer DDS including alias fold.
//! * Oscillator output: `sat(rnd(T30 * amp, 15))` over the frozen sine table
//!   with 12-bit index + 24-bit fraction interpolation.
//! * Partial banks accumulate in i64 over partials and saturate **once** at
//!   the generator output boundary (order-independent for any backend).
//! * Noise: `VOLE-SPLITMIX64-STREAM`, `noise_sample(stream_key, media_frame)`
//!   (pure; `seek == sequential` by construction).
//! * Constant/silence are trivially their level/zero.

use crate::universe::arithmetic::{rnd_shift, sat_i32};
use crate::universe::phase::{sine_amp_to_code, sine_interp, sine_table_i32};

/// Nyquist ceiling of the phase domain (2^63 = half-turn = fs/2 at rate 1).
pub const INC_MAX: u64 = 1u64 << 63;

/// Effective per-output-frame phase increment for a base frequency and an
/// effective voice rate (both rational via Q24 rate): exact
/// `round(2^64 * f0 * rate / 2^24 / fs)`, clamped at `INC_MAX` (Nyquist).
///
/// Freeze: oscillators **clamp** rather than alias-fold when the scaled
/// frequency exceeds Nyquist (VCO-style cap). Harmonic partials of a partial
/// bank *do* alias-fold via their modulo-2^64 increment arithmetic.
///
/// Host-side only (device receives the precomputed increment in the flat
/// voice state). Uses i128 intermediates.
pub fn eff_incr(freq_hz: u32, rate_q24: i64, fs_hz: u32) -> u64 {
    let f = u128::from(freq_hz);
    let fs = u128::from(fs_hz.max(1));
    let rate_abs = rate_q24.unsigned_abs() as u128;
    // 2^64 * f0 * rate / 2^24 / fs == f0 * rate * 2^40 / fs
    let num = f * rate_abs * (1u128 << 40);
    let den = fs;
    let incr = (num + den / 2) / den;
    incr.min(u128::from(INC_MAX)) as u64
}

/// Phase of a voice-local oscillator at media frame `t`:
/// `(phase0 + incr * (t - t0)) mod 2^64` (all arithmetic wrapping; `t < t0`
/// is legal).
#[inline]
pub fn osc_phase(phase0: u64, incr: u64, t0: i64, t: i64) -> u64 {
    let d = (t - t0) as u128;
    phase0.wrapping_add(u128::from(incr).wrapping_mul(d) as u64)
}

/// Oscillator sample: sine table at `phase` scaled by `amp_q16` (Q16,
/// unity 1<<16) into the sample-code domain.
#[inline]
pub fn osc_sample(phase: u64, amp_q16: i32) -> i32 {
    let table = sine_table_i32();
    let v = sine_interp(table, phase);
    sine_amp_to_code(v, amp_q16)
}

/// One partial of a partial bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Partial {
    /// Harmonic number (1 = fundamental). 0 is rejected at construction.
    pub harmonic: u32,
    /// Amplitude in Q16 (unity 1<<16), domain `[-(1<<17), 1<<17]`.
    pub amp_q16: i32,
}

/// Partial-bank sample at `t`: i64 accumulation over partials with one final
/// saturation. Each partial's phase advances at `harmonic * incr` (alias fold
/// included); each partial's sine is scaled by its amplitude and summed.
///
/// `partials` must be sorted by ascending harmonic for canonical identity and
/// deterministic summation order (i64 addition is associative, so the order
/// is not semantically load-bearing — it is frozen for byte determinism of
/// the accumulator trace).
pub fn partial_bank_sample(partials: &[Partial], base_incr: u64, t0: i64, t: i64) -> i32 {
    let d = (t - t0) as u128;
    let table = sine_table_i32();
    let mut acc: i64 = 0;
    for p in partials {
        // Phase advance for partial k: (k * base_incr mod 2^64) * d mod 2^64.
        let k_incr =
            u128::from(p.harmonic).wrapping_mul(u128::from(base_incr)) & ((1u128 << 64) - 1);
        let phase = k_incr.wrapping_mul(d) as u64;
        let v = sine_interp(table, phase);
        acc += i64::from(v) * i64::from(p.amp_q16);
    }
    // One rounding + one saturation at the generator output boundary.
    sat_i32(rnd_shift(acc, 15))
}

/// Validate a partial list (bounds + ascending order + nonzero harmonics).
pub fn validate_partials(partials: &[Partial]) -> bool {
    let mut prev_harmonic: u32 = 0;
    if partials.is_empty() {
        return false;
    }
    if partials.len() as u32 > crate::limits::MAX_PARTIALS {
        return false;
    }
    for p in partials {
        if p.harmonic == 0 || p.harmonic <= prev_harmonic {
            return false;
        }
        if p.amp_q16.abs() > (1 << 17) {
            return false;
        }
        prev_harmonic = p.harmonic;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eff_incr_matches_direct_ratio() {
        // f0 = 440, rate = unity, fs = 48000 => incr == freq_to_incr(440).
        assert_eq!(
            eff_incr(440, 1 << 24, 48_000),
            crate::universe::phase::freq_to_incr(440, 48_000)
        );
        // Double rate doubles the increment.
        let base = eff_incr(440, 1 << 24, 48_000);
        let double = eff_incr(440, 2 << 24, 48_000);
        assert!((double as i128 - 2 * base as i128).abs() <= 2);
        // Rate 1/2 halves the frequency: eff f0 = 220.
        let half = eff_incr(440, 1 << 23, 48_000);
        assert_eq!(half, crate::universe::phase::freq_to_incr(220, 48_000));
        // Clamp at Nyquist (2^63).
        assert_eq!(eff_incr(1, 1 << 40, 48_000), INC_MAX);
    }

    #[test]
    fn osc_phase_accumulates_mod_2p64() {
        assert_eq!(osc_phase(0, 10, 0, 5), 50);
        // Wrap: incr near 2^64.
        let incr = u64::MAX - 3;
        assert_eq!(osc_phase(0, incr, 0, 2), (incr.wrapping_mul(2)));
        // Negative delta.
        assert_eq!(osc_phase(100, 10, 10, 5), 50);
    }

    #[test]
    fn oscillator_unity_amp_peaks_within_domain() {
        let incr = crate::universe::phase::freq_to_incr(440, 48_000);
        let mut peak: i64 = 0;
        for t in 0..200_000 {
            let phase = osc_phase(0, incr, 0, t);
            let s = osc_sample(phase, 1 << 16);
            peak = peak.max(i64::from(s));
        }
        // Peak near full scale positive but below i32::MAX (never saturates).
        assert!(peak > (1 << 30) && peak < i64::from(i32::MAX));
    }

    #[test]
    fn partial_bank_deterministic_and_bounded() {
        let partials = [
            Partial {
                harmonic: 1,
                amp_q16: 1 << 15,
            },
            Partial {
                harmonic: 2,
                amp_q16: 1 << 14,
            },
            Partial {
                harmonic: 3,
                amp_q16: 1 << 13,
            },
        ];
        assert!(validate_partials(&partials));
        let incr = crate::universe::phase::freq_to_incr(220, 48_000);
        let a = partial_bank_sample(&partials, incr, 0, 1234);
        let b = partial_bank_sample(&partials, incr, 0, 1234);
        assert_eq!(a, b);
        // Bounded by sum of |amps| * full scale in i64 (no saturation needed
        // at these levels, but the function must never panic).
        for t in 0..10_000 {
            let _ = partial_bank_sample(&partials, incr, 0, t);
        }
    }

    #[test]
    fn partial_validation() {
        assert!(!validate_partials(&[]));
        assert!(!validate_partials(&[Partial {
            harmonic: 0,
            amp_q16: 1
        }]));
        assert!(!validate_partials(&[
            Partial {
                harmonic: 2,
                amp_q16: 1
            },
            Partial {
                harmonic: 1,
                amp_q16: 1
            },
        ]));
        assert!(!validate_partials(&[Partial {
            harmonic: 1,
            amp_q16: (1 << 17) + 1,
        }]));
        let many = (1..=crate::limits::MAX_PARTIALS + 1)
            .map(|h| Partial {
                harmonic: h,
                amp_q16: 1,
            })
            .collect::<Vec<_>>();
        assert!(!validate_partials(&many));
    }
}

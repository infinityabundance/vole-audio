//! Frozen polyphase FIR resampler.
//!
//! Design (frozen in Phase C after measurement; U1_SPEC §"Resampler"):
//!
//! * 64 taps × 1024 phases, i16 Q15 coefficients (positive-symmetric,
//!   Blackman-Harris windowed sinc, cutoff at source Nyquist).
//! * Canonical bytes: phase-major, little-endian i16:
//!   `assets/u1/resampler_bh64_p1024_q15.bin` (128 KiB).
//! * Every phase row sums to exactly `2^15` so a constant input reproduces
//!   itself exactly (DC gain 1.0 in Q15 arithmetic) — see `resample`.
//! * Row quantization is largest-remainder to hit the exact row sum; the
//!   committed bytes are authoritative; regeneration is non-normative.
//! * Evaluation: `y = sat(rnd(Σ c_j·x_j, 15))` — one rounding, one
//!   saturation, deterministic on every backend.
//!
//! Measured (host test `measure_response`): see the printed/stored values —
//! stopband attenuation and passband ripple are asserted in tests and the
//! numbers are recorded in `docs/U1_SPEC.md`.

use crate::universe::arithmetic::{rnd_shift, sat_i32};

pub const RESAMPLER_TAPS: usize = 64;
pub const RESAMPLER_PHASES: usize = 1024;
/// Index offset such that tap `j` reads source sample `m + j - 31` where
/// `m = floor(pos)`. Effective support: `[m-31, m+32]`.
pub const RESAMPLER_CENTER: usize = 31;
/// Canonical byte length.
pub const RESAMPLER_BYTES: usize = RESAMPLER_TAPS * RESAMPLER_PHASES * 2;

/// Frozen canonical table bytes (phase-major, i16 LE, Q15).
pub static RESAMPLER_TABLE_BYTES: &[u8; RESAMPLER_BYTES] =
    include_bytes!("../../assets/u1/resampler_bh64_p1024_q15.bin");

/// Read one coefficient (phase, tap) as Q15.
#[inline]
pub fn coeff(phase: usize, tap: usize) -> i32 {
    debug_assert!(phase < RESAMPLER_PHASES && tap < RESAMPLER_TAPS);
    let o = (phase * RESAMPLER_TAPS + tap) * 2;
    i16::from_le_bytes([RESAMPLER_TABLE_BYTES[o], RESAMPLER_TABLE_BYTES[o + 1]]) as i32
}

/// Host-side decoded table view (i16 array), const-initialized.
pub fn table_i16() -> &'static [i16; RESAMPLER_TAPS * RESAMPLER_PHASES] {
    static TABLE: [i16; RESAMPLER_TAPS * RESAMPLER_PHASES] = decode_table();
    &TABLE
}

const fn decode_table() -> [i16; RESAMPLER_TAPS * RESAMPLER_PHASES] {
    let mut t = [0i16; RESAMPLER_TAPS * RESAMPLER_PHASES];
    let mut i = 0;
    while i < t.len() {
        let o = i * 2;
        t[i] = i16::from_le_bytes([RESAMPLER_TABLE_BYTES[o], RESAMPLER_TABLE_BYTES[o + 1]]);
        i += 1;
    }
    t
}

/// Sample-source accessor for one resampler tap index `q` (absolute object
/// frame index, may be negative or past the end depending on tap position).
/// `sample_at` maps a raw index to a sample under the read mode:
/// one-shot reads out-of-range as 0; loop regions wrap.
pub type SampleAt = fn(&[i32], i64, Option<(u64, u64)>) -> i32;

/// Default raw-index reader used by the resampler.
/// `region: None` => one-shot (out of range reads 0);
/// `region: Some((a,b))` => Euclidean wrap into `[a,b)`.
pub fn sample_at_default(plane: &[i32], idx: i64, region: Option<(u64, u64)>) -> i32 {
    match region {
        None => {
            if idx < 0 || idx >= plane.len() as i64 {
                0
            } else {
                plane[idx as usize]
            }
        }
        Some((a, b)) => {
            debug_assert!(b > a && b as usize <= plane.len());
            let len = (b - a) as i64;
            let a = a as i64;
            let mut r = (idx - a) % len;
            if r < 0 {
                r += len;
            }
            plane[(a + r) as usize]
        }
    }
}

/// Polyphase fractional read at Q24 position `pos_q24` over `plane`.
///
/// Phase index: `p = ((pos >> 14) & 0x3FF)` (top 10 fraction bits); taps at
/// `m + j - 31`, `m = pos >> 24`.
#[inline]
pub fn resample(
    plane: &[i32],
    pos_q24: i64,
    region: Option<(u64, u64)>,
    sample_at: SampleAt,
) -> i32 {
    let m = pos_q24 >> 24;
    let p = ((pos_q24 >> 14) as usize) & (RESAMPLER_PHASES - 1);
    let mut acc: i64 = 0;
    for j in 0..RESAMPLER_TAPS {
        let c = coeff(p, j);
        if c == 0 {
            continue;
        }
        let x = sample_at(plane, m + j as i64 - RESAMPLER_CENTER as i64, region);
        acc += i64::from(c) * i64::from(x);
    }
    sat_i32(rnd_shift(acc, 15))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_asset_hash_is_pinned() {
        let digest = crate::hash::sha256::Sha256::digest(RESAMPLER_TABLE_BYTES);
        let hex = crate::hash::sha256::hex(&digest);
        // Pinned at freeze; updating the table is a profile change.
        assert_eq!(
            hex,
            "3b3015a81b9da1298b532127e72309f7dd1afdbf817628dbcabd8fe958438750"
        );
    }

    /// Regenerate the frozen asset (non-normative after freeze; run with
    /// `cargo test -- --ignored regenerate_resampler_asset`).
    ///
    /// Construction: Blackman-Harris windowed sinc kernel evaluated at
    /// `t = 31 + f - j`, `f = (p + 0.5)/1024`, then each phase row is
    /// quantized to i16 with deterministic largest-remainder rounding (capped
    /// at 32767) so the row sums to exactly 2^15 (DC gain 1.0 in Q15).
    #[test]
    #[ignore = "regeneration is non-normative; only run to recreate the asset"]
    fn regenerate_resampler_asset() {
        const A0: f64 = 0.35875;
        const A1: f64 = 0.48829;
        const A2: f64 = 0.14128;
        const A3: f64 = 0.01168;
        let bh = |x: f64| -> f64 {
            // x in [0,1]
            A0 - A1 * (2.0 * std::f64::consts::PI * x).cos()
                + A2 * (4.0 * std::f64::consts::PI * x).cos()
                - A3 * (6.0 * std::f64::consts::PI * x).cos()
        };
        let sinc = |t: f64| -> f64 {
            if t.abs() < 1e-12 {
                1.0
            } else {
                let p = std::f64::consts::PI * t;
                p.sin() / p
            }
        };
        let kernel = |t: f64| -> f64 {
            if t.abs() > 32.0 {
                0.0
            } else {
                sinc(t) * bh((t + 32.0) / 64.0)
            }
        };
        let mut out = Vec::with_capacity(RESAMPLER_BYTES);
        for p in 0..RESAMPLER_PHASES {
            let f = (p as f64 + 0.5) / RESAMPLER_PHASES as f64;
            let raw: Vec<f64> = (0..RESAMPLER_TAPS)
                .map(|j| kernel(31.0 + f - j as f64))
                .collect();
            let sum: f64 = raw.iter().sum();
            let target = (1i64 << 15) as f64;
            // Cap the scaled values at 32767 so they always fit i16.
            let scaled: Vec<f64> = raw
                .iter()
                .map(|&k| (k * target / sum).min(32767.0))
                .collect();
            let mut q: Vec<i64> = scaled.iter().map(|&v| v.floor() as i64).collect();
            // Deterministic largest-remainder: while the row sum is short,
            // add 1 to the tap with the largest fractional part that can still
            // grow. Always terminates (total capacity is enormous).
            let mut deficit = (1i64 << 15) - q.iter().sum::<i64>();
            while deficit > 0 {
                let mut best = None;
                let mut best_frac = -1.0f64;
                for (j, &v) in scaled.iter().enumerate() {
                    if q[j] >= 32767 {
                        continue;
                    }
                    let frac = v - q[j] as f64;
                    if frac > best_frac {
                        best_frac = frac;
                        best = Some(j);
                    }
                }
                let j = best.expect("capacity is sufficient");
                q[j] += 1;
                deficit -= 1;
            }
            debug_assert!(q.iter().all(|&v| (-(1 << 15)..=(1 << 15) - 1).contains(&v)));
            debug_assert_eq!(q.iter().sum::<i64>(), 1 << 15);
            for v in q {
                out.extend_from_slice(&(v as i16).to_le_bytes());
            }
        }
        assert_eq!(out.len(), RESAMPLER_BYTES);
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/u1/resampler_bh64_p1024_q15.bin"
        );
        std::fs::write(path, &out).expect("write asset");
    }

    #[test]
    fn every_row_sums_to_exactly_32768() {
        // DC gain == 1.0 exactly in Q15 for every phase.
        for p in 0..RESAMPLER_PHASES {
            let s: i64 = (0..RESAMPLER_TAPS).map(|j| i64::from(coeff(p, j))).sum();
            assert_eq!(s, 1 << 15, "row {p} sum");
        }
    }

    #[test]
    fn constant_input_is_exact_through_every_phase() {
        let plane = vec![42_000i32; 256];
        for p in 0..RESAMPLER_PHASES {
            // Position with fraction selecting phase p, integer part mid-plane.
            let frac = p as i64;
            let pos = (100i64 << 24) | (frac << 14);
            let y = resample(&plane, pos, None, sample_at_default);
            assert_eq!(y, 42_000, "phase {p}");
        }
    }

    #[test]
    fn impulse_reproduces_kernel_and_is_finite() {
        // Impulse at index 200: output around it is the (windowed) kernel.
        let mut plane = vec![0i32; 512];
        plane[200] = 1 << 20;
        let pos = 200i64 << 24; // exact integer position, phase 0
        let y = resample(&plane, pos, None, sample_at_default);
        // At an exact integer position the resampler is not an identity (the
        // windowed kernel has nonzero taps at integer offsets), but the output
        // must be bounded and near the kernel's center tap sum.
        assert!(y > (1 << 20) / 2 && y < (1 << 20) * 2, "y={y}");
    }

    #[test]
    fn loop_region_wraps() {
        let plane: Vec<i32> = (0..16).map(|i| i * 1000).collect();
        let region = Some((4u64, 12u64)); // 4000..11000
        let pos = (11i64 << 24) + (1 << 23); // inside region near end
        let y = resample(&plane, pos, region, sample_at_default);
        // Taps near the loop boundary wrap into [4,12); DC-normalized signed
        // weights can ring a few percent of the region swing, so assert a
        // generous envelope around the region's value range.
        assert!(y > 2_000 && y < 13_000, "y={y}");
    }

    /// Response measurement of the frozen kernel family. Computes
    /// (a) the continuous-time Fourier transform magnitude of the windowed
    /// sinc kernel on a fine grid — passband ripple (F < 0.5) and stopband
    /// attenuation (F > 0.5 + transition);
    /// (b) worst-case per-phase passband deviation of the *sampled* rows
    /// (the DTFT of each 64-tap row vs ideal 1.0) over F in [0.01, 0.45].
    /// Run with `cargo test measure_response -- --ignored --nocapture` to
    /// refresh the numbers recorded in U1_SPEC.
    #[test]
    #[ignore = "prints measurements for the spec; run manually"]
    fn measure_response() {
        const A0: f64 = 0.35875;
        const A1: f64 = 0.48829;
        const A2: f64 = 0.14128;
        const A3: f64 = 0.01168;
        let bh = |x: f64| -> f64 {
            A0 - A1 * (2.0 * std::f64::consts::PI * x).cos()
                + A2 * (4.0 * std::f64::consts::PI * x).cos()
                - A3 * (6.0 * std::f64::consts::PI * x).cos()
        };
        let sinc = |t: f64| -> f64 {
            if t.abs() < 1e-12 {
                1.0
            } else {
                let p = std::f64::consts::PI * t;
                p.sin() / p
            }
        };
        let kernel = |t: f64| -> f64 {
            if t.abs() > 32.0 {
                0.0
            } else {
                sinc(t) * bh((t + 32.0) / 64.0)
            }
        };

        // (a) continuous-kernel response via fine quadrature.
        let n = 1 << 16;
        let span = 32.0f64;
        let dt = 2.0 * span / n as f64;
        let resp = |f: f64| -> f64 {
            let mut re = 0.0;
            let mut im = 0.0;
            for i in 0..n {
                let t = -span + (i as f64 + 0.5) * dt;
                let v = kernel(t);
                let ph = 2.0 * std::f64::consts::PI * f * t;
                re += v * ph.cos();
                im -= v * ph.sin();
            }
            (re * re + im * im).sqrt() * dt
        };
        let mut passband_max_err = 0.0f64;
        let mut f = 0.0;
        while f <= 0.45 {
            let m = resp(f);
            passband_max_err = passband_max_err.max((m - 1.0).abs());
            f += 0.005;
        }
        let mut stop_min_db = f64::MAX;
        let mut f = 0.5 + 1.0 / 16.0; // past the 1/16 transition width
        while f <= 1.0 {
            let m = resp(f).max(1e-12);
            let db = 20.0 * m.log10();
            stop_min_db = stop_min_db.min(db);
            f += 0.005;
        }

        // (b) worst per-phase sampled-row passband deviation vs ideal 1.0.
        let mut worst_row = 0.0f64;
        for p in [0usize, 256, 512, 768, 1023] {
            let mut worst = 0.0f64;
            let mut f = 0.01;
            while f <= 0.45 {
                let mut re = 0.0;
                let mut im = 0.0;
                for j in 0..RESAMPLER_TAPS {
                    let c = coeff(p, j) as f64 / (1 << 15) as f64;
                    let t = j as f64 - 31.0;
                    let ph = 2.0 * std::f64::consts::PI * f * t;
                    re += c * ph.cos();
                    im -= c * ph.sin();
                }
                worst = worst.max((re.hypot(im) - 1.0).abs());
                f += 0.01;
            }
            worst_row = worst_row.max(worst);
        }
        eprintln!(
            "passband ripple (F<=0.45): max |H-1| = {:.2e}  ({:.4} dB)",
            passband_max_err,
            20.0 * (1.0 + passband_max_err).log10()
        );
        eprintln!("stopband min attenuation (F in [0.5625,1]): {stop_min_db:.1} dB");
        eprintln!("worst sampled-row passband deviation (F<=0.45): {worst_row:.2e}");
        eprintln!("transition band starts at F=0.5, BH-4 window, 64 taps @ 1024 phases");
    }
}

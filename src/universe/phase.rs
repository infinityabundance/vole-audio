//! Oscillator phase semantics (frozen for u1).
//!
//! Freeze record (U1_SPEC §"Oscillator phase"):
//!
//! * Oscillator phase is a `u64` advancing **modulo 2^64** (free-running DDS).
//! * The phase increment is `round(2^64 * freq_hz / rate_hz)` computed with
//!   exact integer arithmetic (see `freq_to_incr`; no 128-bit division is
//!   required, so the same code compiles for GPU targets).
//! * The base table is a 4096-entry sine, i32 entries in
//!   `[-(2^30-1), 2^30-1]` (Q30), stored canonically as little-endian i32
//!   bytes in `assets/u1/sine_q30_4096.bin`. Those bytes are authoritative;
//!   regeneration is non-normative (U1_SPEC §"Tables").
//! * Addressing: `index = phase >> 52` (12 bits), interpolation fraction
//!   `frac24 = (phase >> 28) & 0xFF_FFFF` (24 bits). Interpolation is the
//!   integer-exact linear rule from `arithmetic.rs`.
//! * Amplitude: `amp_q16` with unity `1<<16`. The Q30 table value times a Q16
//!   amplitude reduces by 15 bits: `code = sat(rnd(table * amp, 15))`, giving
//!   a peak of `±(2^31-2)` at unity amplitude (exact; no saturation at
//!   unity).

use crate::universe::arithmetic::{rnd_shift, sat_i32};

pub const SINE_TABLE_BITS: u32 = 12;
pub const SINE_TABLE_LEN: usize = 1 << SINE_TABLE_BITS; // 4096
/// Canonical little-endian byte length of the table.
pub const SINE_TABLE_BYTES: usize = SINE_TABLE_LEN * 4;
/// Bits of interpolation fraction below the table index.
pub const SINE_FRAC_BITS: u32 = 24;
/// Shift from phase to table index (64 - 12).
pub const SINE_INDEX_SHIFT: u32 = 64 - SINE_TABLE_BITS;
/// Shift from phase to fraction top bit (64 - 12 - 24).
pub const SINE_FRAC_SHIFT: u32 = SINE_INDEX_SHIFT - SINE_FRAC_BITS;

/// Frozen canonical table bytes (little-endian i32, Q30).
pub static SINE_TABLE_BYTES_LE: &[u8; SINE_TABLE_BYTES] =
    include_bytes!("../../assets/u1/sine_q30_4096.bin");

/// Load a table entry from its canonical little-endian bytes.
#[inline]
pub fn sine_entry(index: usize) -> i32 {
    debug_assert!(index < SINE_TABLE_LEN);
    let o = index * 4;
    i32::from_le_bytes([
        SINE_TABLE_BYTES_LE[o],
        SINE_TABLE_BYTES_LE[o + 1],
        SINE_TABLE_BYTES_LE[o + 2],
        SINE_TABLE_BYTES_LE[o + 3],
    ])
}

/// Host-side lazy decoded table view (alignment-safe, const-initialized).
/// Device builds upload the canonical bytes instead; both surfaces observe
/// identical values.
pub fn sine_table_i32() -> &'static [i32; SINE_TABLE_LEN] {
    static TABLE: [i32; SINE_TABLE_LEN] = decode_sine_table();
    &TABLE
}

/// Const decode of the canonical little-endian bytes into an i32 table.
const fn decode_sine_table() -> [i32; SINE_TABLE_LEN] {
    let mut t = [0i32; SINE_TABLE_LEN];
    let mut i = 0;
    while i < SINE_TABLE_LEN {
        let o = i * 4;
        t[i] = i32::from_le_bytes([
            SINE_TABLE_BYTES_LE[o],
            SINE_TABLE_BYTES_LE[o + 1],
            SINE_TABLE_BYTES_LE[o + 2],
            SINE_TABLE_BYTES_LE[o + 3],
        ]);
        i += 1;
    }
    t
}

/// Phase increment for a frequency (Hz) at a nominal rate: exact
/// `round(2^64 * f / r)`, computed in 64-bit arithmetic only.
///
/// Frequencies above Nyquist (`r/2`) are clamped; `r < 2` yields `0`
/// (degenerate, documented — media rates are never that low in practice).
pub fn freq_to_incr(freq_hz: u32, rate_hz: u32) -> u64 {
    let r = u64::from(rate_hz);
    if r < 2 {
        return 0;
    }
    let nyquist = (r / 2) as u32;
    let f = u64::from(freq_hz.min(nyquist));
    if f == 0 {
        return 0;
    }
    // q = floor((2^64 * f + r/2) / r). Decompose t0 = f << 32:
    //   2^64 f = t0 << 32
    //   n = (t0 << 32) + r/2 = a*r*2^32 + m*2^32 + r/2,  a = t0/r, m = t0%r
    //   q = (a << 32) + (m << 32 + r/2)/r
    let t0 = f << 32;
    let a = t0 / r;
    let m = t0 % r;
    let hi = ((m << 32) + (r >> 1)) / r;
    (a << 32) + hi
}

/// Map a phase to `(table_index, frac24)`.
#[inline]
pub fn address(phase: u64) -> (u32, u32) {
    let index = (phase >> SINE_INDEX_SHIFT) as u32; // 0..4095 by construction
    let frac = ((phase >> SINE_FRAC_SHIFT) & 0xFF_FFFF) as u32;
    (index, frac)
}

/// Periodic linear interpolation over the sine table at a phase (table-domain
/// Q30 value). The table is periodic, so the wrap entry is `(index+1) & 4095`.
#[inline]
pub fn sine_interp(table: &[i32; SINE_TABLE_LEN], phase: u64) -> i32 {
    let (i, frac) = address(phase);
    let a = table[i as usize];
    let b = table[((i + 1) & (SINE_TABLE_LEN as u32 - 1)) as usize];
    crate::universe::arithmetic::lerp_table(a, b, frac)
}

/// Reduce a Q30 table value by a Q16 amplitude into the sample-code domain:
/// `code = sat(rnd(table * amp, 15))`.
#[inline]
pub fn sine_amp_to_code(table_value_q30: i32, amp_q16: i32) -> i32 {
    let v = i64::from(table_value_q30) * i64::from(amp_q16);
    sat_i32(rnd_shift(v, 15))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference increment via u128 (test oracle only).
    fn freq_to_incr_ref(freq_hz: u32, rate_hz: u32) -> u64 {
        let r = u128::from(rate_hz);
        if r < 2 {
            return 0;
        }
        let f = u128::from(freq_hz.min((r / 2) as u32));
        if f == 0 {
            return 0;
        }
        (((f << 64) + (r / 2)) / r) as u64
    }

    #[test]
    fn freq_to_incr_matches_u128_oracle() {
        for rate in [8_000u32, 44_100, 48_000, 96_000, 192_000, 1_000_000] {
            for freq in [1u32, 440, 1_000, rate / 4, rate / 2, rate, rate * 2] {
                assert_eq!(freq_to_incr(freq, rate), freq_to_incr_ref(freq, rate));
            }
        }
    }

    #[test]
    fn incr_semantics() {
        // 1 Hz at 48k advances ~ 2^64/48000 per sample.
        let i = freq_to_incr(1, 48_000);
        assert!(i > (1u64 << 48) && i < (1u64 << 49));
        // Nyquist at 48k = 2^63 exactly.
        assert_eq!(freq_to_incr(24_000, 48_000), 1u64 << 63);
        // Above Nyquist clamps to Nyquist.
        assert_eq!(freq_to_incr(48_000, 48_000), 1u64 << 63);
        // Doubling frequency approximately doubles the increment (within one
        // rounding ulp at each step).
        let d = (freq_to_incr(2, 48_000) * 2) as i128 - freq_to_incr(4, 48_000) as i128;
        assert!(d.abs() <= 2);
    }

    #[test]
    fn addressing_splits_phase() {
        // phase with index bits and frac bits set distinctly.
        let phase: u64 = (1234u64 << 52) | (0xAB_CDEFu64 << 28) | 0x0FFF_FFFF;
        let (i, f) = address(phase);
        assert_eq!(i, 1234);
        assert_eq!(f, 0xAB_CDEF);
        assert_eq!(address(0), (0, 0));
        // Full phase wraps to the last table slot (mod 2^64) with max frac.
        let (i, f) = address(u64::MAX);
        assert_eq!(i, 4095);
        assert_eq!(f, 0xFF_FFFF);
    }

    #[test]
    fn frozen_asset_hash_is_pinned() {
        // U1_SPEC records this hash; changing the table bytes is a profile
        // change, not an edit.
        let digest = crate::hash::sha256::Sha256::digest(SINE_TABLE_BYTES_LE);
        let hex = crate::hash::sha256::hex(&digest);
        assert_eq!(
            hex,
            "455d4647044595871d5f07789581abc28a6499b2bf622f4f3b62ed85297fdf21"
        );
    }

    #[test]
    fn table_entries_are_frozen_and_symmetric() {
        // Byte length is canonical.
        assert_eq!(SINE_TABLE_BYTES_LE.len(), 16_384);
        // Entry 0 = 0; quarter-cycle positive peak capped at 2^30-1.
        assert_eq!(sine_entry(0), 0);
        assert_eq!(sine_entry(1024), (1 << 30) - 1);
        assert_eq!(sine_entry(2048), 0); // sin(pi)
        assert_eq!(sine_entry(3072), -((1 << 30) - 1));
        // Anti-symmetry: s[i] == -s[(i + 2048) & 4095].
        for i in 0..2048 {
            assert_eq!(sine_entry(i), -sine_entry((i + 2048) & 4095));
        }
        // Odd symmetry about index 0: s[i] == -s[(4096 - i) & 4095].
        for i in 0..4096 {
            assert_eq!(sine_entry(i), -sine_entry((4096 - i) & 4095));
        }
        // Even symmetry about index 1024: s[1024+d] == s[1024-d].
        for d in 1..1024 {
            assert_eq!(sine_entry(1024 + d), sine_entry(1024 - d));
        }
        // Domain.
        for i in 0..4096 {
            let v = sine_entry(i);
            assert!((-((1 << 30) - 1)..=(1 << 30) - 1).contains(&v));
        }
    }

    #[test]
    fn interp_continuity_and_edges() {
        let table = sine_table_i32();
        // At exact table indices the interpolation reproduces the entry.
        for i in [0usize, 1, 100, 2048, 4095] {
            let phase = (i as u64) << 52;
            assert_eq!(sine_interp(table, phase), table[i]);
        }
        // Midway between entry 0 (0) and entry 1 (~2^30*sin(2pi/4096)): the
        // interpolated magnitude must be strictly between the two neighbors.
        let mid = sine_interp(table, 1u64 << 51);
        assert!(table[0] < mid && mid < table[1]);
        // Wrap: phase just below 2^64 interpolates between entry 4095 and 0.
        let wrap = sine_interp(table, u64::MAX - (1u64 << 51));
        assert!(wrap > table[4095] || wrap < table[0]);
    }

    #[test]
    fn unity_amplitude_never_saturates() {
        for i in 0..4096 {
            let v = sine_amp_to_code(sine_entry(i), 1 << 16);
            assert!(v > i32::MIN && v < i32::MAX, "entry {i} saturates: {v}");
        }
        assert_eq!(sine_amp_to_code(sine_entry(1024), 1 << 16), i32::MAX - 1);
        assert_eq!(sine_amp_to_code(sine_entry(3072), 1 << 16), i32::MIN + 2);
        // Doubled amplitude clips at the voice-bus boundary.
        assert_eq!(sine_amp_to_code(sine_entry(1024), 1 << 17), i32::MAX);
        assert_eq!(sine_amp_to_code(sine_entry(3072), 1 << 17), i32::MIN);
    }

    /// Regenerate the frozen asset (non-normative after freeze; run with
    /// `cargo test -- --ignored regenerate_sine_asset`).
    ///
    /// Construction: compute the first quadrant (indices 0..=1024) with f64
    /// sin, clamp the peak to `2^30-1`, then mirror with **exact integer
    /// symmetry** so the frozen table is bit-exact odd-symmetric about index 0
    /// and even-symmetric about index 1024 (properties the spec and tests
    /// rely on). Float error therefore cannot break symmetry.
    #[test]
    #[ignore = "regeneration is non-normative; only run to recreate the asset"]
    fn regenerate_sine_asset() {
        let cap = ((1 << 30) - 1) as f64;
        let mut t = [0i32; SINE_TABLE_LEN];
        // First quadrant inclusive: k in 0..=1024.
        for (k, slot) in t.iter_mut().enumerate().take(1025) {
            let x = core::f64::consts::TAU * (k as f64) / (SINE_TABLE_LEN as f64);
            let raw = x.sin() * (1u64 << 30) as f64;
            let rounded = raw.round().clamp(-cap, cap);
            *slot = rounded as i32;
        }
        // Even symmetry about index 1024: S[2048 - k] = S[k], k = 1..=1023.
        for k in 1..=1023usize {
            t[2048 - k] = t[k];
        }
        t[2048] = 0; // sin(pi) exactly.
                     // Odd half: S[2048 + k] = -S[k] for k = 1..=1024.
        for k in 1..=1024usize {
            t[2048 + k] = -t[k];
        }
        // Odd mirror about 0: S[4096 - k] = -S[k] for k = 1..=1023.
        for k in 1..=1023usize {
            t[4096 - k] = -t[k];
        }
        // Sanity: t[0] = 0 and t[4096-1024] = t[3072] = -(2^30 - 1).
        assert_eq!(t[0], 0);
        assert_eq!(t[1024], (1 << 30) - 1);
        assert_eq!(t[3072], -((1 << 30) - 1));
        let mut bytes = Vec::with_capacity(SINE_TABLE_BYTES);
        for e in t {
            bytes.extend_from_slice(&e.to_le_bytes());
        }
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/u1/sine_q30_4096.bin");
        std::fs::write(path, &bytes).expect("write asset");
        assert_eq!(bytes.len(), SINE_TABLE_BYTES);
    }
}

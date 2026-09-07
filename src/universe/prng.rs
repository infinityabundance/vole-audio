//! Frozen deterministic PRNGs.
//!
//! Freeze record (U1_SPEC §"Procedural generators / noise"):
//!
//! * Noise is a *pure function of (stream key, frame)* so that observation is
//!   randomly accessible: `seek == sequential` holds by construction and no
//!   hidden per-voice state exists. Algorithm id `VOLE-SPLITMIX64-STREAM`.
//! * Sequential PRNG needs (inverse-search jitter, tests, offline tooling)
//!   use xoshiro256\*\* with a documented 256-bit state and splitmix64
//!   seeding. Algorithm id `VOLE-XOSHIRO256STARSTAR-1`.
//! * `rand::thread_rng()` is **never** used in normative media semantics.
//!
//! Both implementations are pure integer, `no_std`-clean, and deterministic
//! across scalar/SIMD/GPU.

/// SplitMix64 (David Blackman / public domain description, as published with
/// the xoshiro family). One 64-bit mixing step.
#[inline]
pub fn splitmix64_next(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Finalizer for streaming a 64-bit state to a 64-bit hash value
/// (SplitMix64's mixer without the counter add).
#[inline]
pub fn splitmix64_finalize(x: u64) -> u64 {
    let mut x = x;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Derive a stream-scrambling constant from a stream key (avoids linear
/// frame-key correlations across streams).
#[inline]
pub fn stream_scramble(stream_key: u64) -> u64 {
    splitmix64_finalize(stream_key ^ 0xD1B5_4A32_D192_ED03)
}

/// Pure per-frame noise draw for a stream.
///
/// `noise(stream_key, frame)` = splitmix64 over a mixed (scrambled stream,
/// frame) pair. Frame is mixed with a bijective integer avalanche so adjacent
/// frames decorrelate.
#[inline]
pub fn noise_i64(stream_key: u64, frame: i64) -> i64 {
    let s = stream_scramble(stream_key);
    let f = splitmix64_finalize((frame as u64).wrapping_add(0x9E37_79B9_7F4A_7C15));
    splitmix64_finalize(s ^ f.rotate_left(17)) as i64
}

/// Noise sample in the full i32 sample-code domain (uniform over all 2^32
/// codes as a bit pattern; used as `SampleCode`).
#[inline]
pub fn noise_sample(stream_key: u64, frame: i64) -> i32 {
    noise_i64(stream_key, frame) as i32
}

/// xoshiro256\*\* state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XoShiro256 {
    s: [u64; 4],
}

impl XoShiro256 {
    pub const fn from_state(s: [u64; 4]) -> Self {
        Self { s }
    }

    /// Seed from a 64-bit key using splitmix64 (documented seeding rule).
    pub fn from_seed(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            *slot = splitmix64_next(&mut z);
        }
        Self { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Canonical `jump` (2^128 draws) — see the xoshiro reference code.
    pub fn jump(&mut self) {
        const JUMP: [u64; 4] = [
            0x180EC6D33CFD0ABA,
            0xD5A61266F0C9392C,
            0xA9582618E03FC9AA,
            0x39ABDC4529B1661C,
        ];
        let mut s = [0u64; 4];
        for &j in &JUMP {
            for b in 0..64 {
                if j & (1u64 << b) != 0 {
                    s[0] ^= self.s[0];
                    s[1] ^= self.s[1];
                    s[2] ^= self.s[2];
                    s[3] ^= self.s[3];
                }
                self.next_u64();
            }
        }
        self.s = s;
    }
}

impl Default for XoShiro256 {
    fn default() -> Self {
        Self::from_seed(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_reference_vector() {
        // Published SplitMix64 reference: first output from seed 0.
        let mut z = 0u64;
        assert_eq!(splitmix64_next(&mut z), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn noise_is_pure_function_of_key_and_frame() {
        assert_eq!(noise_i64(7, 100), noise_i64(7, 100));
        assert_eq!(noise_sample(7, 100), noise_sample(7, 100));
        assert_ne!(noise_i64(7, 100), noise_i64(8, 100));
        assert_ne!(noise_i64(7, 100), noise_i64(7, 101));
        assert_ne!(noise_i64(0, 0), noise_i64(0, 1));
    }

    #[test]
    fn noise_sample_covers_bit_domain() {
        // Both sign halves are reachable over a modest sweep (statistical,
        // not exhaustive — the mapping is a bijection over u64->i64->i32).
        let mut seen_neg = false;
        let mut seen_pos = false;
        for f in 0..4096 {
            let v = noise_sample(0xDEAD_BEEF, f);
            if v < 0 {
                seen_neg = true;
            } else {
                seen_pos = true;
            }
        }
        assert!(seen_neg && seen_pos);
    }

    #[test]
    fn xoshiro_state_advances_and_jumps() {
        let mut a = XoShiro256::from_state([1, 2, 3, 4]);
        let first = a.next_u64();
        let mut b = XoShiro256::from_state([1, 2, 3, 4]);
        b.jump();
        // After jump the sequence must differ from the first output stream.
        let after_jump = b.next_u64();
        assert_ne!(after_jump, first);
    }

    #[test]
    fn xoshiro_deterministic() {
        let mut a = XoShiro256::from_seed(42);
        let mut b = XoShiro256::from_seed(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn zero_seed_state_is_non_degenerate() {
        // Even a zero seed must produce a working sequence (documented).
        let mut r = XoShiro256::from_seed(0);
        let mut prev = r.next_u64();
        for _ in 0..100 {
            let v = r.next_u64();
            assert_ne!(v, prev);
            prev = v;
        }
    }
}

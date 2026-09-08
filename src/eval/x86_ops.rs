//! x86-64 vector ops with uniform names across the two ISA floors.
//!
//! `v512` is native AVX-512 (8× i64 lanes); `v256` is AVX2 (4× i64 lanes)
//! with 64-bit multiply / arithmetic-shift / signed min-max emulated (AVX2
//! has no native 64-bit vector variants of those). Every op is exact: the
//! emulations produce identical lane values to their AVX-512 natives (and
//! therefore to the scalar integer semantics) for every input — each op is
//! unit-tested against the scalar reference.
//!
//! Shifts take a *runtime* count and use the variable-shift instructions
//! (`vpsrlvq`/`vpsravq`/`vpsllvq`); counts must be in `1..=63`.
//!
//! The kernel lane bodies in `eval::x86` are macro-generated once and
//! instantiated per floor against these ops, so the lane math cannot drift
//! between ISAs.
//!
//! # SAFETY
//!
//! Every function is `unsafe` and carries `#[target_feature]`: callers must
//! run under the matching runtime-detected target features (see
//! `eval::backend`).

#![cfg(target_arch = "x86_64")]
// Thin intrinsic-wrapper layer: every function below is an `unsafe fn` whose
// body is entirely intrinsic calls (or tests trampolines), so the unsafe
// boundary is the function signature itself. Rust 2024's
// `unsafe_op_in_unsafe_fn` would demand a block around every single intrinsic;
// the explicit `unsafe fn` + `#[target_feature]` contract is the audited
// boundary here (documented in the module docs and the kernel callers).
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

/// AVX-512 floor: 8× i64 lanes, native ops (AVX-512F + DQ + VL).
pub(crate) mod v512 {
    use super::*;

    pub(crate) const W: usize = 8;

    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn splat(v: i64) -> __m512i {
        _mm512_set1_epi64(v)
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn sub(a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }
    /// Wrapping low-64 multiply (mod 2^64).
    #[target_feature(enable = "avx512dq")]
    pub(crate) unsafe fn mulq(a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi64(a, b)
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn srl(a: __m512i, k: i32) -> __m512i {
        _mm512_srlv_epi64(a, _mm512_set1_epi64(i64::from(k)))
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn sra(a: __m512i, k: i32) -> __m512i {
        _mm512_srav_epi64(a, _mm512_set1_epi64(i64::from(k)))
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn sll(a: __m512i, k: i32) -> __m512i {
        _mm512_sllv_epi64(a, _mm512_set1_epi64(i64::from(k)))
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn band(a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn bor(a: __m512i, b: __m512i) -> __m512i {
        _mm512_or_si512(a, b)
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn bxor(a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn maxq(a: __m512i, b: __m512i) -> __m512i {
        _mm512_max_epi64(a, b)
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn minq(a: __m512i, b: __m512i) -> __m512i {
        _mm512_min_epi64(a, b)
    }
    /// select lanes: `if a > b { gt } else { le }` (signed compare).
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn gt_sel(a: __m512i, b: __m512i, gt: __m512i, le: __m512i) -> __m512i {
        let k = _mm512_cmpgt_epi64_mask(a, b);
        _mm512_mask_mov_epi64(le, k, gt)
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn loadu(p: *const i64) -> __m512i {
        _mm512_loadu_si512(p as *const _)
    }
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn storeu(p: *mut i64, v: __m512i) {
        _mm512_storeu_si512(p as *mut _, v);
    }
}

/// AVX2 floor: 4× i64 lanes, emulated 64-bit ops.
pub(crate) mod v256 {
    use super::*;

    pub(crate) const W: usize = 4;

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn splat(v: i64) -> __m256i {
        _mm256_set1_epi64x(v)
    }
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi64(a, b)
    }
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sub(a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi64(a, b)
    }
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn srl(a: __m256i, k: i32) -> __m256i {
        _mm256_srlv_epi64(a, _mm256_set1_epi64x(i64::from(k)))
    }
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sll(a: __m256i, k: i32) -> __m256i {
        _mm256_sllv_epi64(a, _mm256_set1_epi64x(i64::from(k)))
    }
    /// Arithmetic shift right, emulated: logical shift + sign fill.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sra(a: __m256i, k: i32) -> __m256i {
        debug_assert!((1..=63).contains(&k));
        let r = _mm256_srlv_epi64(a, _mm256_set1_epi64x(i64::from(k)));
        let neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a); // all-ones where a < 0
        let fill = _mm256_sllv_epi64(neg, _mm256_set1_epi64x(i64::from(64 - k)));
        _mm256_or_si256(r, fill)
    }
    /// Signed max via compare + select.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn maxq(a: __m256i, b: __m256i) -> __m256i {
        gt_sel(a, b, a, b)
    }
    /// Signed min via compare + select.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn minq(a: __m256i, b: __m256i) -> __m256i {
        gt_sel(a, b, b, a)
    }
    /// select lanes: `if a > b { gt } else { le }`.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn gt_sel(a: __m256i, b: __m256i, gt: __m256i, le: __m256i) -> __m256i {
        let m = _mm256_cmpgt_epi64(a, b);
        _mm256_or_si256(
            _mm256_and_si256(gt, m),
            _mm256_and_si256(le, _mm256_xor_si256(m, _mm256_set1_epi64x(-1))),
        )
    }
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn band(a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn bor(a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn bxor(a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }
    /// Low-64 multiply, emulated with 32-bit multiplies (exact for every
    /// input pair, mod 2^64): `a*b = al*bl + ((al*bh + ah*bl) << 32)`.
    ///
    /// `vpmuludq` multiplies the low 32-bit dwords of *every* 64-bit lane
    /// (dwords 0 and 2 of each 128-bit lane), so all four lanes are covered
    /// with no shuffling.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mulq(a: __m256i, b: __m256i) -> __m256i {
        let mask32 = _mm256_set1_epi64x(0xFFFF_FFFF);
        let a_lo = _mm256_and_si256(a, mask32);
        let b_lo = _mm256_and_si256(b, mask32);
        let a_hi = _mm256_srli_epi64(a, 32);
        let b_hi = _mm256_srli_epi64(b, 32);
        let p00 = _mm256_mul_epu32(a_lo, b_lo);
        let p01 = _mm256_mul_epu32(a_lo, b_hi);
        let p10 = _mm256_mul_epu32(a_hi, b_lo);
        let cross = _mm256_add_epi64(p01, p10);
        let cross = _mm256_slli_epi64(cross, 32);
        _mm256_add_epi64(p00, cross)
    }
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn loadu(p: *const i64) -> __m256i {
        _mm256_loadu_si256(p as *const _)
    }
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn storeu(p: *mut i64, v: __m256i) {
        _mm256_storeu_si256(p as *mut _, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::universe::prng::XoShiro256;

    fn mulq_ref(a: i64, b: i64) -> i64 {
        (a as u64).wrapping_mul(b as u64) as i64
    }

    #[target_feature(enable = "avx2")]
    unsafe fn run_v256_ops() {
        let mut rng = XoShiro256::from_seed(0xC0FFEE);
        let mut a = [0i64; 4];
        let mut b = [0i64; 4];
        let mut out = [0i64; 4];
        for _ in 0..200_000 {
            for v in &mut a {
                *v = rng.next_u64() as i64;
            }
            for v in &mut b {
                *v = rng.next_u64() as i64;
            }
            let r = v256::mulq(v256::loadu(a.as_ptr()), v256::loadu(b.as_ptr()));
            v256::storeu(out.as_mut_ptr(), r);
            for j in 0..4 {
                assert_eq!(out[j], mulq_ref(a[j], b[j]), "mulq lane {j}");
            }
        }
        for &(x, y) in &[
            (0i64, 0i64),
            (0, i64::MAX),
            (1, -1),
            (i64::MAX, i64::MAX),
            (i64::MIN, i64::MIN),
            (i64::MIN, i64::MAX),
            (i64::MIN, 1),
            (-1, -1),
        ] {
            let r = v256::mulq(v256::splat(x), v256::splat(y));
            v256::storeu(out.as_mut_ptr(), r);
            for &got in out.iter() {
                assert_eq!(got, mulq_ref(x, y));
            }
        }
        for k in [1i32, 15, 16, 24, 31, 63] {
            for _ in 0..50_000 {
                for v in &mut a {
                    *v = rng.next_u64() as i64;
                }
                let r = v256::sra(v256::loadu(a.as_ptr()), k);
                v256::storeu(out.as_mut_ptr(), r);
                for j in 0..4 {
                    assert_eq!(out[j], a[j] >> k, "sra k={k} lane {j}");
                }
            }
        }
        for _ in 0..50_000 {
            for v in &mut a {
                *v = rng.next_u64() as i64;
            }
            for v in &mut b {
                *v = rng.next_u64() as i64;
            }
            let av = v256::loadu(a.as_ptr());
            let bv = v256::loadu(b.as_ptr());
            v256::storeu(out.as_mut_ptr(), v256::minq(av, bv));
            for j in 0..4 {
                assert_eq!(out[j], a[j].min(b[j]), "min lane {j}");
            }
            v256::storeu(out.as_mut_ptr(), v256::maxq(av, bv));
            for j in 0..4 {
                assert_eq!(out[j], a[j].max(b[j]), "max lane {j}");
            }
            v256::storeu(out.as_mut_ptr(), v256::gt_sel(av, bv, av, bv));
            for j in 0..4 {
                assert_eq!(
                    out[j],
                    if a[j] > b[j] { a[j] } else { b[j] },
                    "sel lane {j}"
                );
            }
        }
    }

    #[test]
    fn v256_emulated_ops_match_reference() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe { run_v256_ops() };
    }

    #[target_feature(enable = "avx2")]
    unsafe fn run_v256_rnd_trace() {
        // Reproduce the kernel rnd_shift path on the failing value
        // (v = -2^46 exactly).
        let v = -70368744177664i64;
        let vv = v256::splat(v);
        let s = v256::sra(vv, 63);
        let a = v256::sub(v256::bxor(vv, s), s);
        let h = v256::add(a, v256::splat(1 << 15));
        let mut out = [0i64; 4];
        v256::storeu(out.as_mut_ptr(), h);
        assert_eq!(out[0], 70368744177664 + 32768, "h wrong: {}", out[0]);
        let r = v256::sra(h, 16);
        v256::storeu(out.as_mut_ptr(), r);
        assert_eq!(out[0], 1 << 30, "sra broken: {}", out[0]);
        // And the mulq emulation on the same magnitude.
        let p = v256::mulq(v256::splat(-1073741824), v256::splat(65536));
        v256::storeu(out.as_mut_ptr(), p);
        assert_eq!(out[0], -70368744177664, "mulq broken: {}", out[0]);
    }

    #[test]
    fn v256_rnd_shift_trace() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe { run_v256_rnd_trace() };
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512vl")]
    unsafe fn run_v512_basic() {
        let mut out = [0i64; 8];
        v512::storeu(out.as_mut_ptr(), v512::splat(-7));
        assert!(out.iter().all(|&x| x == -7));
        v512::storeu(out.as_mut_ptr(), v512::sra(v512::splat(-8), 2));
        assert!(out.iter().all(|&x| x == -2));
        v512::storeu(
            out.as_mut_ptr(),
            v512::gt_sel(
                v512::splat(5),
                v512::splat(3),
                v512::splat(1),
                v512::splat(2),
            ),
        );
        assert!(out.iter().all(|&x| x == 1));
        // Native 512 vs emulated 256 agreement on identical data.
        let mut rng = XoShiro256::from_seed(0x51A7);
        let mut a = [0i64; 8];
        let mut b = [0i64; 8];
        let mut r512 = [0i64; 8];
        for _ in 0..20_000 {
            for v in &mut a {
                *v = rng.next_u64() as i64;
            }
            for v in &mut b {
                *v = rng.next_u64() as i64;
            }
            let av = v512::loadu(a.as_ptr());
            let bv = v512::loadu(b.as_ptr());
            v512::storeu(r512.as_mut_ptr(), v512::mulq(av, bv));
            let mut o4 = [0i64; 4];
            for group in 0..2 {
                let mut a4 = [0i64; 4];
                let mut b4 = [0i64; 4];
                for j in 0..4 {
                    a4[j] = a[group * 4 + j];
                    b4[j] = b[group * 4 + j];
                }
                let r4 = v256::mulq(v256::loadu(a4.as_ptr()), v256::loadu(b4.as_ptr()));
                v256::storeu(o4.as_mut_ptr(), r4);
                for j in 0..4 {
                    assert_eq!(r512[group * 4 + j], o4[j], "cross-floor lane {j}");
                }
            }
        }
    }

    #[test]
    fn v512_ops_and_cross_floor_agreement() {
        if !std::is_x86_feature_detected!("avx512f")
            || !std::is_x86_feature_detected!("avx512dq")
            || !std::is_x86_feature_detected!("avx512vl")
        {
            return;
        }
        unsafe { run_v512_basic() };
    }
}

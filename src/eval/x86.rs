//! Frame-blocked x86-64 kernels (Phase F).
//!
//! Lanes = consecutive output frames. Each lane computes the identical integer
//! math as the scalar world engine (the semantic authority); only arithmetic
//! is batched in vector registers. Unavoidable scalar-per-lane points:
//! envelope division (no vector integer division), content/sine-table loads,
//! and extraction into the i64 mix accumulator.
//!
//! Two floors are generated from one macro source (`kernels!`) so the lane
//! math cannot drift between ISAs: `kavx2` (4 lanes; 64-bit ops emulated,
//! see `eval::x86_ops`) and `kavx512` (8 lanes; native). The engine
//! (`eval::simd`) calls the block functions only for whole blocks inside a
//! single envelope segment and inside the content read domain; tail frames
//! are rendered through the scalar oracle's own `contribution_at`.
//!
//! # SAFETY
//!
//! Every kernel is `unsafe fn` carrying `#[target_feature]`; callers must
//! runtime-verify the feature (via `eval::backend::detect_isa`).

// Kernel bodies are `unsafe fn`s whose unsafe operations (intrinsics via the
// ops layer, unchecked content loads) are covered by the caller contracts in
// the module docs; see the identical note in `eval::x86_ops`.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::eval::x86_ops;
use crate::sampler::mix::Mixer;
use core::arch::x86_64::{__m256i, __m512i};

/// Per-voice context for content-class kernels (Literal and the cycle
/// classes Wavetable/SingleCycle/ExactRepeat).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ContentCtx<'a> {
    /// Content samples, interleaved by object channel.
    pub samples: &'a [i32],
    /// Object layout channels (stride between consecutive frames).
    pub obj_channels: usize,
    /// First object channel read.
    pub ch_a: usize,
    /// Second object channel read (stereo route).
    pub ch_b: usize,
    /// Read both channels (stereo route).
    pub read_stereo: bool,
    /// Play source is a periodic cycle (always wraps `[0, extent)`).
    pub is_cycle: bool,
    /// Loop region `[region_a, region_b)` in frames (cycles use
    /// `region_a = 0`, `region_b = extent`).
    pub has_region: bool,
    pub region_a: u64,
    pub region_b: u64,
    /// Object extent (frames) — the read-domain ceiling.
    pub extent_frames: u64,
    pub start_pos_q24: i64,
    pub rate_q24: i64,
    pub t_on: i64,
    pub gain_q16: i32,
    /// True when per-lane single-correction wrap is exact for the widest
    /// block (`|rate| * (Wmax - 1) <= len`); false forces per-lane Euclidean
    /// wrap (always exact).
    pub wrap_fast: bool,
    /// Linear interpolation (false = nearest).
    pub linear: bool,
    /// Mono route target channel (stereo route uses `stereo_base`).
    pub mono_ch: Option<u8>,
    /// Stereo pair base channel.
    pub stereo_base: Option<u8>,
    /// Pan gains for the stereo route (Q16).
    pub gl: i32,
    pub gr: i32,
}

/// Per-voice context for endless-class kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EndlessCtx {
    pub kind: EndlessKind,
    pub t_on: i64,
    pub gain_q16: i32,
    pub out_ch: u8,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum EndlessKind {
    Silence,
    Constant(i32),
    Noise { seed: u64 },
    Oscillator { incr: u64, amp_q16: i32 },
}

macro_rules! kernels {
    ($name:ident, $feat:literal, $ops:path, $vt:ty) => {
        pub(crate) mod $name {
            use super::*;
            use crate::eval::simd::SegKind;
            use $ops as o;

            const W: usize = o::W;
            /// Public block width (frames per kernel block) for this floor.
            pub(crate) const BLOCK_W: usize = o::W;

            /// Euclidean wrap of a Q24 position into `[a_w, a_w + len_w)`.
            #[inline]
            unsafe fn rate_wrap(u: i64, a_w: i64, len_w: i64) -> i64 {
                if len_w == 0 {
                    return a_w;
                }
                let r = (u - a_w) % len_w;
                a_w + if r < 0 { r + len_w } else { r }
            }

            // -- vector mirrors of the scalar exact helpers -----------------

            /// `rnd_shift(v, q)` (round half away from zero) per lane.
            #[target_feature(enable = $feat)]
            unsafe fn rnd_shift_v(v: $vt, q: u32) -> $vt {
                let s = o::sra(v, 63);
                let a = o::sub(o::bxor(v, s), s); // |v|
                let r = o::sra(o::add(a, o::splat(1i64 << (q - 1))), q as i32);
                o::sub(o::bxor(r, s), s)
            }

            /// `sat_i32` per lane (clamp to the i32 code domain).
            #[target_feature(enable = $feat)]
            unsafe fn sat_i32_v(v: $vt) -> $vt {
                o::minq(
                    o::maxq(v, o::splat(i64::from(i32::MIN))),
                    o::splat(i64::from(i32::MAX)),
                )
            }

            /// `mul_q16(a, b)`: one rounding (half away from zero), one
            /// saturation — the exact gain-chain application point.
            #[target_feature(enable = $feat)]
            unsafe fn mul_q16_v(a: $vt, b: $vt) -> $vt {
                sat_i32_v(rnd_shift_v(o::mulq(a, b), 16))
            }

            /// Vector lerp (identical rule to `arithmetic::lerp_i32`).
            #[target_feature(enable = $feat)]
            unsafe fn lerp_v(a: $vt, b: $vt, frac: $vt) -> $vt {
                let d = o::sub(b, a);
                let step = rnd_shift_v(o::mulq(d, frac), 24);
                sat_i32_v(o::add(a, step))
            }

            /// Per-lane envelope levels (Q16) at frames `t0..t0+W`.
            #[target_feature(enable = $feat)]
            unsafe fn env_block(kind: SegKind, anchor: i64, t0: i64, out: &mut [i64; W]) {
                for (j, slot) in out.iter_mut().enumerate() {
                    *slot = i64::from(crate::eval::simd::seg_level(kind, anchor, t0 + j as i64));
                }
            }

            /// Scatter one contribution vector into the mixer (per-lane
            /// extract; the mixer accumulator is plain i64).
            #[target_feature(enable = $feat)]
            unsafe fn scatter(c: $vt, out_ch: u8, t0: i64, start_frame: i64, mixer: &mut Mixer) {
                let mut arr = [0i64; W];
                o::storeu(arr.as_mut_ptr(), c);
                for j in 0..W {
                    mixer.add(
                        (t0 + j as i64 - start_frame) as usize,
                        out_ch,
                        arr[j] as i32,
                    );
                }
            }

            /// Per-lane oscillator observation (sample-code domain): frozen
            /// Q30 table via `sine_interp` addressing + `sine_amp_to_code`.
            #[target_feature(enable = $feat)]
            unsafe fn obs_osc(kind: EndlessKind, t_on: i64, t0: i64, out: &mut [i64; W]) {
                let (incr, amp) = match kind {
                    EndlessKind::Oscillator { incr, amp_q16 } => (incr, amp_q16),
                    _ => unreachable!("obs_osc only for oscillators"),
                };
                let mut t = [0i64; W];
                for (j, v) in t.iter_mut().enumerate() {
                    *v = t0 + j as i64 - t_on;
                }
                // phase = incr * (t - t_on) mod 2^64 (low-64 product of the bit
                // patterns is identical to the scalar u128 wrapping math).
                let d = o::loadu(t.as_ptr());
                let phase = o::mulq(o::splat(incr as i64), d);
                let idxv = o::srl(phase, crate::universe::phase::SINE_INDEX_SHIFT as i32);
                let fracv = o::band(
                    o::srl(phase, crate::universe::phase::SINE_FRAC_SHIFT as i32),
                    o::splat(0xFF_FFFF),
                );
                let table = crate::universe::phase::sine_table_i32();
                let mut a = [0i64; W];
                let mut b = [0i64; W];
                let mut lane = [0i64; W];
                for j in 0..W {
                    o::storeu(lane.as_mut_ptr(), idxv);
                    let i = (lane[j] as usize) & (crate::universe::phase::SINE_TABLE_LEN - 1);
                    a[j] = i64::from(table[i]);
                    b[j] = i64::from(table[(i + 1) & (crate::universe::phase::SINE_TABLE_LEN - 1)]);
                }
                let mut fr = [0i64; W];
                o::storeu(fr.as_mut_ptr(), fracv);
                let v = lerp_v(
                    o::loadu(a.as_ptr()),
                    o::loadu(b.as_ptr()),
                    o::loadu(fr.as_ptr()),
                );
                // amp reduce: sat(rnd(v * amp, 15))
                let r = sat_i32_v(rnd_shift_v(o::mulq(v, o::splat(i64::from(amp))), 15));
                o::storeu(out.as_mut_ptr(), r);
            }

            /// Per-lane deterministic noise — vectorized
            /// VOLE-SPLITMIX64-STREAM (`prng::noise_sample`); outputs are
            /// sign-extended from the low 32 bits (the i32 code domain).
            #[target_feature(enable = $feat)]
            unsafe fn obs_noise(seed: u64, t0: i64, out: &mut [i64; W]) {
                #[target_feature(enable = $feat)]
                unsafe fn mix(x: $vt) -> $vt {
                    let x = o::bxor(x, o::srl(x, 30));
                    let x = o::mulq(x, o::splat(0xBF58_476D_1CE4_E5B9u64 as i64));
                    let x = o::bxor(x, o::srl(x, 27));
                    let x = o::mulq(x, o::splat(0x94D0_49BB_1331_11EBu64 as i64));
                    o::bxor(x, o::srl(x, 31))
                }
                let s = crate::universe::prng::stream_scramble(seed);
                let mut t = [0i64; W];
                for (j, v) in t.iter_mut().enumerate() {
                    *v = (t0 + j as i64) as u64 as i64;
                }
                let tv = o::loadu(t.as_ptr());
                let f = mix(o::add(tv, o::splat(0x9E37_79B9_7F4A_7C15u64 as i64)));
                let rot = o::bor(o::sll(f, 17), o::srl(f, 47));
                let r = mix(o::bxor(o::splat(s as i64), rot));
                let mut arr = [0i64; W];
                o::storeu(arr.as_mut_ptr(), r);
                for j in 0..W {
                    out[j] = i64::from(arr[j] as i32);
                }
            }

            /// Raw Q24 positions at `t0..t0+W` (wrapping, like scalar
            /// `position_at`).
            #[target_feature(enable = $feat)]
            unsafe fn positions_v(ctx: &ContentCtx, t0: i64) -> $vt {
                let d0 = t0 - ctx.t_on;
                let pos0 = ctx
                    .start_pos_q24
                    .wrapping_add(ctx.rate_q24.wrapping_mul(d0));
                let mut jv = [0i64; W];
                for (j, v) in jv.iter_mut().enumerate() {
                    *v = j as i64;
                }
                o::add(
                    o::splat(pos0),
                    o::mulq(o::splat(ctx.rate_q24), o::loadu(jv.as_ptr())),
                )
            }

            /// Per-lane content observation for both read channels.
            #[target_feature(enable = $feat)]
            unsafe fn obs_content(
                ctx: &ContentCtx,
                t0: i64,
                obs_a: &mut [i64; W],
                obs_b: &mut [i64; W],
            ) {
                let posv = positions_v(ctx, t0);
                let (a, b) = if ctx.is_cycle || ctx.has_region {
                    if ctx.is_cycle {
                        (0u64, ctx.extent_frames)
                    } else {
                        (ctx.region_a, ctx.region_b)
                    }
                } else {
                    (0u64, ctx.extent_frames) // one-shot: clamped, no wrap
                };
                let a_w = (a << crate::limits::FIXED_Q) as i64;
                let len_w = ((b - a) << crate::limits::FIXED_Q) as i64;

                // Wrap each lane's raw position into the read domain.
                let mut w = [0i64; W];
                if ctx.is_cycle || ctx.has_region {
                    if ctx.wrap_fast {
                        // Normalize the block anchor once (exact), then one
                        // conditional correction per lane (exact because
                        // |rate| * (W-1) <= len).
                        let w0 = rate_wrap(
                            ctx.start_pos_q24
                                .wrapping_add(ctx.rate_q24.wrapping_mul(t0 - ctx.t_on)),
                            a_w,
                            len_w,
                        );
                        let mut jv = [0i64; W];
                        for (j, v) in jv.iter_mut().enumerate() {
                            *v = j as i64;
                        }
                        let wv = o::add(
                            o::splat(w0),
                            o::mulq(o::splat(ctx.rate_q24), o::loadu(jv.as_ptr())),
                        );
                        // x >= a+len -> x - len ; then x < a -> x + len
                        // (strict comparisons: lanes inside [a, a+len) are
                        // left untouched; a == 0 lanes stay nonnegative)
                        let upper = o::gt_sel(
                            wv,
                            o::splat(a_w + len_w - 1),
                            o::sub(wv, o::splat(len_w)),
                            wv,
                        );
                        let lower =
                            o::gt_sel(o::splat(a_w), upper, o::add(upper, o::splat(len_w)), upper);
                        o::storeu(w.as_mut_ptr(), lower);
                    } else {
                        // Exact per-lane Euclidean wrap (always safe).
                        o::storeu(w.as_mut_ptr(), posv);
                        for j in 0..W {
                            w[j] = rate_wrap(w[j], a_w, len_w);
                        }
                    }
                } else {
                    o::storeu(w.as_mut_ptr(), posv);
                }

                // Continuation neighbors (identical rules to the scalar
                // `read_content`).
                let region_last = if ctx.is_cycle || ctx.has_region {
                    b as i64 - 1
                } else {
                    ctx.extent_frames as i64 - 1
                };
                let wrap_to: usize = if ctx.is_cycle {
                    0
                } else if ctx.has_region {
                    a as usize
                } else {
                    usize::MAX // latch: next == idx
                };
                let mut fr = [0i64; W];
                for (j, slot) in fr.iter_mut().enumerate() {
                    *slot = w[j] & 0xFF_FFFF;
                }
                let fracv = o::loadu(fr.as_ptr());
                let chan = if ctx.read_stereo { 2 } else { 1 };
                for ch in 0..chan {
                    let ch_off = if ch == 0 { ctx.ch_a } else { ctx.ch_b };
                    let mut a_arr = [0i64; W];
                    let mut b_arr = [0i64; W];
                    for j in 0..W {
                        let idx =
                            ((w[j] >> crate::limits::FIXED_Q) as usize).min(region_last as usize);
                        let nxt = if idx == region_last as usize {
                            if wrap_to == usize::MAX { idx } else { wrap_to }
                        } else {
                            idx + 1
                        };
                        debug_assert!(idx < ctx.extent_frames as usize);
                        debug_assert!(nxt < ctx.extent_frames as usize);
                        let base = idx * ctx.obj_channels + ch_off;
                        a_arr[j] = i64::from(*ctx.samples.get_unchecked(base));
                        b_arr[j] =
                            i64::from(*ctx.samples.get_unchecked(nxt * ctx.obj_channels + ch_off));
                    }
                    let av = o::loadu(a_arr.as_ptr());
                    let bv = o::loadu(b_arr.as_ptr());
                    let obs = if ctx.linear {
                        lerp_v(av, bv, fracv)
                    } else {
                        // Nearest: pick b when frac >= 1/2 and the neighbor is
                        // a different frame (otherwise latch on a). Select
                        // masks are all-ones / zero per lane.
                        let take_b =
                            o::gt_sel(fracv, o::splat((1 << 23) - 1), o::splat(-1), o::splat(0));
                        // Latch detection: per-lane compare idx vs next.
                        let mut lat = [0i64; W];
                        for (j, slot) in lat.iter_mut().enumerate() {
                            let idx = ((w[j] >> crate::limits::FIXED_Q) as usize)
                                .min(region_last as usize);
                            let nxt = if idx == region_last as usize {
                                if wrap_to == usize::MAX { idx } else { wrap_to }
                            } else {
                                idx + 1
                            };
                            *slot = if nxt != idx { -1 } else { 0 };
                        }
                        let latch_mask = o::loadu(lat.as_ptr());
                        let use_b = o::band(take_b, latch_mask);
                        o::bor(
                            o::band(bv, use_b),
                            o::band(av, o::bxor(use_b, o::splat(-1))),
                        )
                    };
                    if ch == 0 {
                        o::storeu(obs_a.as_mut_ptr(), obs);
                    } else {
                        o::storeu(obs_b.as_mut_ptr(), obs);
                    }
                }
                if !ctx.read_stereo {
                    obs_b.fill(0);
                }
            }

            /// Render one whole `W`-frame content block.
            #[target_feature(enable = $feat)]
            pub(crate) unsafe fn content_block(
                ctx: &ContentCtx,
                kind: SegKind,
                anchor: i64,
                t0: i64,
                start_frame: i64,
                mixer: &mut Mixer,
            ) {
                let mut env = [0i64; W];
                let mut obs_a = [0i64; W];
                let mut obs_b = [0i64; W];
                env_block(kind, anchor, t0, &mut env);
                obs_content(ctx, t0, &mut obs_a, &mut obs_b);
                let envv = o::loadu(env.as_ptr());
                let ev = mul_q16_v(envv, o::splat(i64::from(ctx.gain_q16)));
                let obs_av = o::loadu(obs_a.as_ptr());
                match (ctx.mono_ch, ctx.stereo_base) {
                    (Some(ch), None) => {
                        let c = mul_q16_v(obs_av, ev);
                        scatter(c, ch, t0, start_frame, mixer);
                    }
                    (None, Some(base)) => {
                        let ml = mul_q16_v(ev, o::splat(i64::from(ctx.gl)));
                        let mr = mul_q16_v(ev, o::splat(i64::from(ctx.gr)));
                        let cl = mul_q16_v(obs_av, ml);
                        scatter(cl, base, t0, start_frame, mixer);
                        let cr = mul_q16_v(o::loadu(obs_b.as_ptr()), mr);
                        scatter(cr, base + 1, t0, start_frame, mixer);
                    }
                    _ => unreachable!("content route is mono or stereo pair"),
                }
            }

            /// Render one whole `W`-frame endless block (mono route).
            #[target_feature(enable = $feat)]
            pub(crate) unsafe fn endless_block(
                ctx: &EndlessCtx,
                kind: SegKind,
                anchor: i64,
                t0: i64,
                start_frame: i64,
                mixer: &mut Mixer,
            ) {
                let mut env = [0i64; W];
                let mut obs = [0i64; W];
                env_block(kind, anchor, t0, &mut env);
                match ctx.kind {
                    EndlessKind::Silence => obs.fill(0),
                    EndlessKind::Constant(level) => obs.fill(i64::from(level)),
                    EndlessKind::Noise { seed } => obs_noise(seed, t0, &mut obs),
                    EndlessKind::Oscillator { .. } => obs_osc(ctx.kind, ctx.t_on, t0, &mut obs),
                }
                let envv = o::loadu(env.as_ptr());
                let ev = mul_q16_v(envv, o::splat(i64::from(ctx.gain_q16)));
                let c = mul_q16_v(o::loadu(obs.as_ptr()), ev);
                scatter(c, ctx.out_ch, t0, start_frame, mixer);
            }
        }
    };
}

kernels!(kavx2, "avx2", x86_ops::v256, __m256i);
kernels!(kavx512, "avx512f,avx512dq,avx512vl", x86_ops::v512, __m512i);

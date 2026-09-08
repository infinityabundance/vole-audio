//! `nvptx64-nvidia-cuda` kernel entry (Phase G) — compiled only when this
//! package is cross-compiled for the PTX target (`--no-default-features`,
//! `-Z build-std=core`; see scripts/build-cuda-device.sh).
//!
//! Thin ABI wrapper only: all semantics live in `super::kernel_shared`, so
//! the device path shares the exact scalar/SIMD math and the host-side
//! `render_window` can serve as the CPU anchor for differential testing.
//!
//! Decomposition (first D0 diagnostic kernel): one thread per output sample
//! slot `(frame, channel)`; each thread accumulates the exact i64 mix over
//! the voice list in registers, saturates exactly once, and writes the final
//! i32 code coalesced with its neighbors. No global voice×frame sample
//! matrix exists; the only global sample-domain block is the final
//! observation (the D0 diagnostic block by definition). Voices are read from
//! global memory (read-mostly, cached); the shared-memory voice-tiling
//! variant is a later measured optimization, not assumed better.

#![cfg(target_arch = "nvptx64")]

use crate::device::entropy_shared::{EntropyJobDesc, FlatPage, FlatStream};
use crate::device::kernel_shared::{FlatState, FlatVoice, render_sample};
use crate::sampler::procedural::Partial;

/// Decode entropy pages into a bounded output arena (Phase H.2).
///
/// Grid/block mapping: one thread per page (grid-stride over the declared
/// page count); each thread decodes its page through the shared
/// `device::entropy_shared::decode_page` and writes its status (1 = exact
/// decode, 0 = malformed) to `status[page_id]`. The output arena holds the
/// window's samples; every page writes at its `out_off`. Pages decode only
/// the entropy payloads they reference — no full-object waveform is ever
/// materialized (H.2.17/H.2.18).
///
/// # Safety
/// All pointers must match the lengths declared in `*desc` (see
/// `EntropyJobDesc`). The host builds these buffers exactly
/// (`backend::cuda::EntropyWorld`); nothing here is user input.
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn vole_entropy_decode(
    desc: *const EntropyJobDesc,
    pages: *const FlatPage,
    streams: *const FlatStream,
    payload: *const u8,
    mvalues: *const u16,
    mstarts: *const u32,
    mfreqs: *const u32,
    mranges: *const u32,
    cycle: *const i32,
    out: *mut i32,
    scratch: *mut u8,
    status: *mut u8,
) {
    unsafe {
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let nthreads = core::arch::nvptx::_block_dim_x() as usize;
        let bid = core::arch::nvptx::_block_idx_x() as usize;
        let nblocks = core::arch::nvptx::_grid_dim_x() as usize;
        let d = &*desc;
        let stride = nthreads * nblocks;
        let mut g = bid * nthreads + tid;
        while g < d.page_count as usize {
            let ok = d.decode_page_raw(
                g as u32, pages, streams, payload, mvalues, mstarts, mfreqs, mranges, cycle, out,
                scratch,
            );
            *status.add(g) = u8::from(ok);
            g += stride;
        }
    }
}
/// Expand a mono sample arena into an interleaved multi-channel endpoint
/// region by duplication (L = R = ... = sample) — the sampler/mix transform
/// at the observation boundary for mono objects played on a multi-channel
/// endpoint (H.2.19: the fused entropy->D1 path must not materialize the
/// expansion on the host).
///
/// Grid/block mapping: one thread per source frame, grid-stride; each thread
/// writes `channels` identical interleaved codes at `dst[frame*channels..]`.
///
/// # Safety
/// `src` must hold at least `frames` i32 samples and `dst` at least
/// `frames * channels` i32 slots. The host builds both exactly
/// (`court entropy-d1`); nothing here is user input.
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn vole_upmix_mono_dup(
    src: *const i32,
    dst: *mut i32,
    frames: u64,
    channels: u64,
) {
    unsafe {
        let tid = core::arch::nvptx::_thread_idx_x() as u64;
        let nthreads = core::arch::nvptx::_block_dim_x() as u64;
        let bid = core::arch::nvptx::_block_idx_x() as u64;
        let nblocks = core::arch::nvptx::_grid_dim_x() as u64;
        let stride = nthreads * nblocks;
        let mut g = bid * nthreads + tid;
        while g < frames {
            let v = *src.add(g as usize);
            let base = (g * channels) as usize;
            for c in 0..channels as usize {
                *dst.add(base + c) = v;
            }
            g += stride;
        }
    }
}

/// Render one window into the final interleaved observation block (D0).
///
/// Grid/block mapping: `g = blockIdx.x * blockDim.x + threadIdx.x` indexes
/// output sample slots; grid-stride so any window size is legal.
///
/// # Safety
/// All five pointers must reference allocations that live for the kernel
/// duration and match the lengths recorded in `*state` (the voices,
/// samples and partials arenas, and an output block of exactly
/// `state.total_samples()` i32 slots). The host builds these buffers
/// exactly (`backend::cuda::KernelWorld`); nothing here is user input.
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn vole_render_d0(
    state: *const FlatState,
    voices: *const FlatVoice,
    samples: *const i32,
    partials: *const Partial,
    out: *mut i32,
) {
    unsafe {
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let nthreads = core::arch::nvptx::_block_dim_x() as usize;
        let bid = core::arch::nvptx::_block_idx_x() as usize;
        let nblocks = core::arch::nvptx::_grid_dim_x() as usize;

        let st = &*state;
        debug_assert!(st.check_layout().is_some());
        let total = st.total_samples();
        let voices_slice = core::slice::from_raw_parts(voices, st.voices_len as usize);
        let samples_slice = core::slice::from_raw_parts(samples, st.samples_len as usize);
        let partials_slice = core::slice::from_raw_parts(partials, st.partials_len as usize);

        let stride = nthreads * nblocks;
        let mut g = bid * nthreads + tid;
        while g < total {
            *out.add(g) = render_sample(&st, voices_slice, samples_slice, partials_slice, g);
            g += stride;
        }
    }
}

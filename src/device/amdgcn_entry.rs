//! `amdgcn-amd-amdhsa` kernel entry — Phase I (ROCm).
//!
//! Mirrors `nvptx_entry` on the AMD device target: thin ABI wrappers only;
//! all semantics live in `super::kernel_shared` / `super::entropy_shared`,
//! so the device path shares the exact scalar/SIMD math and differential
//! testing applies unchanged (`scalar == SIMD == CUDA == ROCm`).
//!
//! AMDGCN divergence from NVPTX (deliberate, documented):
//!
//! * **Launch geometry is passed as kernel parameters.** `core::arch::amdgpu`
//!   exposes the workitem id and workgroup id but not the workgroup count or
//!   workgroup size, so the host passes `blocks_x`/`threads_x` (the same
//!   values the NVPTX kernels read from the `blockIdx/blockDim/gridDim`
//!   builtins). They are launch geometry the host computed — never user
//!   input.
//! * **The artifact is a per-gfx code object, not portable text.** NVPTX
//!   emits PTX that the driver JITs for the installed GPU; AMDGPU code
//!   objects are ISA-specific (`-C target-cpu=gfx…`). The build script
//!   therefore builds a documented baseline `gfx` (env `VOLE_ROCM_GFX`) and
//!   records it in the artifact provenance; loadability on real ROCm
//!   hardware is Phase-J evidence, never assumed here.

#![cfg(target_arch = "amdgpu")]

use crate::device::entropy_shared::{EntropyJobDesc, FlatPage, FlatStream};
use crate::device::geom::Grid;
use crate::device::kernel_shared::{FlatState, FlatVoice, render_sample};
use crate::sampler::procedural::Partial;

/// Defensive entry guard shared by every kernel: invalid launch geometry
/// (a zero dimension) would make the grid-stride loops non-progressing
/// (`g += 0`), so the kernel returns immediately. The frozen launch contract
/// (see `device::geom`) requires `blocks_x > 0 && threads_x > 0` with the
/// declared values equal to the actual launch geometry.
#[inline]
fn guard(grid: Grid) -> Option<(u64, u64)> {
    if !grid.valid() {
        return None;
    }
    let tid = core::arch::amdgpu::workitem_id_x() as u64;
    let bid = core::arch::amdgpu::workgroup_id_x() as u64;
    grid.first(bid as u32, tid as u32)
        .map(|first| (first, grid.stride()))
}

/// Decode entropy pages into a bounded output arena (Phase H.2 semantics on
/// the AMD surface).
///
/// Grid/block mapping: one thread per page (grid-stride over the declared
/// page count); each thread decodes its page through the shared
/// `device::entropy_shared::decode_page` and writes its status (1 = exact
/// decode, 0 = malformed) to `status[page_id]`. The output arena holds the
/// window's samples; every page writes at its `out_off`. Pages decode only
/// the entropy payloads they reference — no full-object waveform is ever
/// materialized.
///
/// # Safety
/// All pointers must match the lengths declared in `*desc` (see
/// `EntropyJobDesc`). The host builds these buffers exactly; nothing here is
/// user input.
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn vole_entropy_decode(
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
    blocks_x: u32,
    threads_x: u32,
) {
    let grid = Grid::new(blocks_x, threads_x);
    // Defensive: zero geometry must not hang the grid-stride loop.
    let Some((mut g, stride)) = guard(grid) else {
        return;
    };
    unsafe {
        let d = &*desc;
        while g < d.page_count as u64 {
            let ok = d.decode_page_raw(
                g as u32, pages, streams, payload, mvalues, mstarts, mfreqs, mranges, cycle, out,
                scratch,
            );
            *status.add(g as usize) = u8::from(ok);
            g += stride;
        }
    }
}

/// Expand a mono sample arena into an interleaved multi-channel endpoint
/// region by duplication (L = R = ... = sample) — the sampler/mix transform
/// at the observation boundary (H.2.19 semantics on the AMD surface).
///
/// Grid/block mapping: one thread per source frame, grid-stride; each thread
/// writes `channels` identical interleaved codes at `dst[frame*channels..]`.
///
/// # Safety
/// `src` must hold at least `frames` i32 samples and `dst` at least
/// `frames * channels` i32 slots. The host builds both exactly; nothing here
/// is user input.
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn vole_upmix_mono_dup(
    src: *const i32,
    dst: *mut i32,
    frames: u64,
    channels: u64,
    blocks_x: u32,
    threads_x: u32,
) {
    let grid = Grid::new(blocks_x, threads_x);
    // Defensive: zero geometry must not hang the grid-stride loop.
    let Some((mut g, stride)) = guard(grid) else {
        return;
    };
    unsafe {
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
/// Grid/block mapping: the executing workitem's global index
/// (`workgroup_id_x * threads_x + workitem_id_x`, via `geom::Grid`) indexes
/// output sample slots; grid-stride so any window size is legal. Zero or
/// mismatched geometry is rejected defensively (`guard`); the frozen launch
/// contract is `device::geom`.
///
/// # Safety
/// All five pointers must reference allocations that live for the kernel
/// duration and match the lengths recorded in `*state` (the voices, samples
/// and partials arenas, and an output block of exactly
/// `state.total_samples()` i32 slots). The host builds these buffers
/// exactly; nothing here is user input.
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn vole_render_d0(
    state: *const FlatState,
    voices: *const FlatVoice,
    samples: *const i32,
    partials: *const Partial,
    out: *mut i32,
    blocks_x: u32,
    threads_x: u32,
) {
    let grid = Grid::new(blocks_x, threads_x);
    // Defensive: zero geometry must not hang the grid-stride loop.
    let Some((first, stride)) = guard(grid) else {
        return;
    };
    unsafe {
        let st = &*state;
        debug_assert!(st.check_layout().is_some());
        let total = st.total_samples();
        let voices_slice = core::slice::from_raw_parts(voices, st.voices_len as usize);
        let samples_slice = core::slice::from_raw_parts(samples, st.samples_len as usize);
        let partials_slice = core::slice::from_raw_parts(partials, st.partials_len as usize);

        let mut g = first;
        while g < total as u64 {
            *out.add(g as usize) =
                render_sample(&st, voices_slice, samples_slice, partials_slice, g as usize);
            g += stride;
        }
    }
}

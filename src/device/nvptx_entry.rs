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

use crate::device::kernel_shared::{FlatState, FlatVoice, render_sample};
use crate::sampler::procedural::Partial;

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

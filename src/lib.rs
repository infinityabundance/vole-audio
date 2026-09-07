//! VOLE-Audio — procedural sampling and direct audio materialization from
//! deterministic state.
//!
//! Architectural invariant (paper v1.0, DOI 10.5281/zenodo.22649073):
//!
//! > A `SampleObject` is the authoritative media object. PCM/sample-domain
//! > values are **observation surfaces**. A literal representation is valid
//! > and mandatory as a universal fallback.
//!
//! Non-negotiable consequences implemented across this crate:
//!
//! * one Cargo package (`vole-audio`), no workspace, no internal crates;
//! * exact deterministic semantics in the `vole.audio.u1` universe — no float
//!   in the normative exact evaluator, explicit fixed-point/integer rules;
//! * one shared semantic core compiled for scalar, SIMD, and GPU device
//!   targets (the evaluator logic is never duplicated per backend);
//! * every result classified (see [`status::Verdict`]) — negative results are
//!   results, and unsupported paths stay visible as evidence;
//! * evidence receipts are immutable and versioned
//!   (`vole.audio.evidence.v1`, see [`evidence`]).
//!
//! Host builds enable the default `std` feature. Device builds compile this
//! same crate with `--no-default-features` for `nvptx64-nvidia-cuda` or
//! `amdgcn-amd-amdhsa`; see `scripts/build-cuda-device.sh` and
//! `scripts/build-rocm-device.sh`.

#![cfg_attr(not(feature = "std"), no_std)]
// Device ABI feature gates: active only when compiling for a GPU target.
// They become *used* once device/nvptx_entry.rs and device/amdgcn_entry.rs
// (Phase G/I) are part of the device build; until then they are declared so the
// pinned-toolchain contract is explicit and reproducible.
#![cfg_attr(
    any(target_arch = "nvptx64", target_arch = "amdgpu"),
    allow(unused_features)
)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx))]
#![cfg_attr(target_arch = "amdgpu", feature(abi_gpu_kernel))]
#![cfg_attr(
    any(target_arch = "nvptx64", target_arch = "amdgpu"),
    feature(asm_experimental_arch)
)]

// ---------------------------------------------------------------------------
// Shared (no_std-clean) core modules — compiled for host scalar/SIMD and for
// GPU device targets without modification.
// ---------------------------------------------------------------------------

pub mod hash;
pub mod limits;
pub mod status;

pub mod eval;
pub mod sampler;
pub mod universe;

#[cfg(feature = "std")]
pub mod audio;
#[cfg(feature = "std")]
pub mod courts;
#[cfg(feature = "std")]
pub mod error;
#[cfg(feature = "std")]
pub mod evidence;
#[cfg(feature = "std")]
pub mod object;
#[cfg(feature = "std")]
pub use error::{Error, Kind, Result};

// ---------------------------------------------------------------------------
// Device panic handler: kernels are no_std and must not unwind.
// ---------------------------------------------------------------------------

#[cfg(all(
    not(feature = "std"),
    any(target_arch = "nvptx64", target_arch = "amdgpu")
))]
#[panic_handler]
fn device_panic(_info: &core::panic::PanicInfo) -> ! {
    // A kernel panic on device is a programming error; park the lane/thread.
    // Normative evaluator code is infallible by construction, so reaching this
    // handler is a bug that should be caught by differential testing.
    loop {}
}

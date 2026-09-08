//! GPU device surface (Phase G): flat kernel semantics + device ABI entries.
//!
//! Everything in this module is compiled from the **same package** for the
//! host (parity tests and the CPU reference) and for `nvptx64-nvidia-cuda`
//! (`extern "ptx-kernel"`) / `amdgcn-amd-amdhsa` (`extern "gpu-kernel"`)
//! device targets. There is no per-backend semantic duplication: the kernel
//! entries are thin ABI wrappers over `kernel_shared`, and the host side of
//! the flat state is produced by `backend::flatten`.
//!
//! Compilation model (see scripts/build-cuda-device.sh):
//!
//! ```text
//! cargo build --target nvptx64-nvidia-cuda --no-default-features -Z build-std=core
//!   -> target/nvptx64-nvidia-cuda/<profile>/vole_audio.ptx
//! ```
//!
//! The produced PTX module is loaded through the CUDA driver API
//! (`backend::cuda`) and every launch is differential-tested against the
//! scalar oracle (`court cuda`).

pub mod entropy_shared;
pub mod kernel_shared;

// Device ABI entries exist only when actually compiling for a GPU target.
// The host build never sees them (they need the device ABI feature gates).
#[cfg(target_arch = "amdgpu")]
pub mod amdgcn_entry;
#[cfg(target_arch = "nvptx64")]
pub mod nvptx_entry;

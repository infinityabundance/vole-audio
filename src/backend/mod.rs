//! Host GPU backend orchestration (Phase G+).
//!
//! `backend::flatten` is backend-neutral: it converts a validated
//! (`store`, `world`) into the flat plain-data state the device kernels
//! consume (`device::kernel_shared`), with per-class capability accounting
//! (`CUDA_NATIVE` vs `CUDA_FALLBACK`). `backend::cuda` is the native-Rust
//! CUDA driver-API runtime (dlopen'd, opt-in, runtime-probed) that uploads
//! the flat world, loads the PTX module produced from this same package, and
//! renders D0 diagnostic observations with exact scalar parity.
//!
//! This module is host-only (`std`); device builds never see it.

pub mod flatten;

#[cfg(feature = "std")]
pub mod cuda;
#[cfg(feature = "std")]
pub mod entropy_flat;
#[cfg(feature = "std")]
pub mod rocm;

pub use flatten::{FlattenedWorld, flatten};

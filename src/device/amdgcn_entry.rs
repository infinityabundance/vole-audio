//! `amdgcn-amd-amdhsa` kernel entry — Phase I (ROCm), not Phase G.
//!
//! Phase G implements CUDA only. This module exists so the device surface
//! (`device::`) is symmetric and the pinned toolchain's `abi_gpu_kernel`
//! contract is exercised in Phase I. Compiled only when this package is
//! cross-compiled for `amdgcn-amd-amdhsa`; until Phase I the build script
//! (scripts/build-rocm-device.sh) refuses to emit a code object.

#![cfg(target_arch = "amdgpu")]

// The Phase I entry (`extern "gpu-kernel" fn vole_render_d0_amdgcn(...)`)
// lands here with its differential court. Nothing in Phase G emits this
// artifact; the module is empty-by-design so the crate compiles for the
// target the day the toolchain/scripts are exercised.

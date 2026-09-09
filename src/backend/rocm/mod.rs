//! Native-Rust ROCm host surface (Phase I + J).
//!
//! Phase I scope (hardware-unavailable evidence + clean amdgcn build): the
//! ROCm runtime surface exists as a *loader probe* (`loader`, `probe`) that
//! walks the exact chain Phase J needs (AMD GPU -> amdgpu driver -> KFD ->
//! HIP/HSA dlopen + symbols) and records where it breaks, plus the
//! code-object artifact contract (`scripts/build-rocm-device.sh`) with
//! self-defending ELF validation (`elf`).
//!
//! Phase J scope (ROCm D1) adds the executable runtime on top of the frozen
//! Phase-I ABI contract: `ffi` (dlopen'd HIP bindings over exactly the
//! `HIP_D0_REQUIRED` / `HIP_D1_ADDITIONAL` tables), `runtime` (RAII session/
//! module/function/buffers/host registration), and `kernel` (the D0 render,
//! entropy decode, and mono-upmix launches mirroring `backend::cuda`). The
//! differential scalar == ROCm battery and the D1 endpoint experiment are
//! `court rocm-d0` / `court rocm-d1`; on hosts without an AMD device they
//! emit typed negatives — the runtime is never pretended to have executed.
//!
//! GPU support is opt-in and runtime-probed; a CPU-only build never touches
//! this module (it is only reachable through court/probe paths that first
//! classify the probe).

pub mod elf;
pub mod ffi;
pub mod kernel;
pub mod loader;
pub mod probe;
pub mod runtime;

pub use elf::{ElfInfo, inspect_amdgcn_code_object};
pub use loader::Lib;
pub use probe::{AmdGpu, KfdState, RocmProbe, RuntimeAttempt};
pub use runtime::{Arg, DeviceBuffer, Function, Module, RegistrationAttempt, Rocm};

//! Native-Rust ROCm host surface (Phase I) — presence probe + artifact.
//!
//! Phase I scope (hardware-unavailable evidence + clean amdgcn build): the
//! ROCm runtime surface exists as a *loader probe* (`loader`, `probe`) that
//! walks the exact chain Phase J needs (AMD GPU -> amdgpu driver -> KFD ->
//! HIP/HSA dlopen + symbols) and records where it breaks, plus the
//! code-object artifact contract (`scripts/build-rocm-device.sh`) with
//! self-defending ELF validation (`elf`). The launch/direct-path runtime
//! (module load, kernel launch, D0/D1 worlds mirroring `backend::cuda`) is
//! Phase J scope, where ROCm hardware can validate it; nothing here executes
//! kernels or claims device execution.
//!
//! GPU support is opt-in and runtime-probed; a CPU-only build never touches
//! this module (it is only reachable through `court rocm`/probe paths that
//! first classify the probe).

pub mod elf;
pub mod loader;
pub mod probe;

pub use elf::{ElfInfo, inspect_amdgcn_code_object};
pub use loader::Lib;
pub use probe::{AmdGpu, KfdState, RocmProbe, RuntimeAttempt};

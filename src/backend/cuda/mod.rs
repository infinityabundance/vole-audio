//! Native-Rust CUDA host runtime (Phase G) — driver API, dlopen'd.
//!
//! * `ffi` — audited minimal driver-API bindings (dynamic loading);
//! * `driver` — RAII context/module/stream/event/buffer/graph wrappers;
//! * `kernel` — D0 buffered-diagnostic renders (GPU-resident flat world,
//!   standard / high-priority stream / captured-graph submission);
//! * `probe` — driver/device/capability evidence for receipts.
//!
//! GPU support is opt-in and runtime-probed; a CPU-only build never touches
//! this module (it is only reachable through `court cuda`/probe paths that
//! first attempt `Cuda::open`).

pub mod direct;
pub mod driver;
pub mod ffi;
pub mod kernel;
pub mod probe;

pub use direct::{HostRegistration, PointerEvidence, RegisterRange};
pub use driver::{Cuda, DeviceBuffer, DeviceInfo, Event, Function, GraphExec, Module, Stream};
pub use kernel::KernelWorld;

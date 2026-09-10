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
pub mod entropy;
pub mod ffi;
pub mod kernel;
pub mod probe;
pub mod search;

pub use direct::{HostRegistration, PointerEvidence, RegisterRange};
pub use driver::{Cuda, DeviceBuffer, DeviceInfo, Event, Function, GraphExec, Module, Stream};
pub use kernel::KernelWorld;
pub use search::SearchWorld;

/// CUDA configuration observed from the process environment, for receipts.
///
/// The library never mutates the process environment (`Cuda::open` is a safe
/// public API and cannot assume a single-threaded process); a runner that wants
/// deterministic JIT and no on-disk JIT cache exports `CUDA_MODULE_LOADING=EAGER`
/// and `CUDA_CACHE_DISABLE=1` before `exec`. This records what was in effect, so
/// configuration is evidence rather than hidden mutation.
pub fn environment_evidence() -> serde_json::Value {
    let (module_loading, cache_disable) = driver::module_loading_policy();
    serde_json::json!({
        "module_loading": module_loading,
        "cache_disable": cache_disable,
        "observation": "read from the process environment (CUDA_MODULE_LOADING / \
                        CUDA_CACHE_DISABLE); the library never calls set_var",
    })
}

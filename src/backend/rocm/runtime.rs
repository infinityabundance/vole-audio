//! Native-Rust HIP runtime session (Phase J) — RAII over the resolved
//! `ffi::Fns`, mirroring `backend::cuda::driver` for the AMD surface.
//!
//! Ownership (structural, not comments): one `HipApi` owns the dlopen'd
//! `Lib` + the resolved `Fns`, shared through `Arc`. Every resource that
//! can outlive the opening session — `Module`, `Function`, `DeviceBuffer`,
//! `HostRegistration` — retains an `Arc` clone, so Rust itself proves the
//! library cannot be unloaded while any resolved function pointer is still
//! reachable, and a module cannot be unloaded while one of its `Function`s
//! is alive (`Function` retains the module owner):
//!
//! ```text
//! DeviceBuffer ─────┐
//! HostRegistration ─┤
//! Module ───────────┤──> HipApi ──> Lib
//! Function ─> Module┘
//! ```
//!
//! * `Rocm` — one HIP session: init, device selection, synchronized launch
//!   context (legacy default stream; every Phase-J launch is fully
//!   synchronized before the host reads anything);
//! * `Module`/`Function` — an AMDGPU code object loaded from bytes
//!   (`hipModuleLoadData`) and its kernel entries;
//! * `DeviceBuffer` — device memory with host<->device copies;
//! * `HostRegistration` — D1: pin an existing host range (the ALSA mmap
//!   endpoint region) with `hipHostRegister(hipHostRegisterMapped)`.
//!
//! Nothing here executes without a device; every path returns typed errors
//! that the courts record exactly.

use crate::backend::rocm::ffi::{self, Fns};
use crate::backend::rocm::loader::Lib;
use crate::device::geom::Grid;
use crate::error::{Error, Kind, Result};
use std::os::raw::{c_char, c_void};
use std::sync::Arc;

/// The dlopen'd HIP library + its resolved function table, owned together.
/// All runtime resources hold an `Arc` to this, so the library outlives
/// every pointer resolved from it (structural lifetime).
pub struct HipApi {
    /// The dlopen handle (dlclose on drop).
    pub lib: Lib,
    /// Resolved function pointers (valid while `lib` is alive).
    pub fns: Fns,
}

impl std::fmt::Debug for HipApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipApi")
            .field("lib", &self.lib.name)
            .finish()
    }
}

/// HIP session (RAII over one `Arc<HipApi>`).
pub struct Rocm {
    pub api: Arc<HipApi>,
    /// Ordinal the session is bound to.
    pub ordinal: i32,
    /// Number of visible HIP devices at open time.
    pub device_count: i32,
    /// Device name (evidence; `None` when hipDeviceGetName is absent).
    pub device_name: Option<String>,
    /// Runtime/driver version integers (evidence; absent when unavailable).
    pub runtime_version: Option<i32>,
    pub driver_version: Option<i32>,
}

impl std::fmt::Debug for Rocm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rocm")
            .field("ordinal", &self.ordinal)
            .field("device_count", &self.device_count)
            .field("device_name", &self.device_name)
            .field("runtime_version", &self.runtime_version)
            .field("driver_version", &self.driver_version)
            .finish()
    }
}

impl Rocm {
    /// Open a HIP session on `ordinal`: resolve the frozen surface from a
    /// freshly dlopen'd HIP library, init, verify the device exists, select
    /// it, and record identity evidence. Fails with the exact missing-symbol
    /// list when the D0 surface does not resolve.
    pub fn open(ordinal: i32) -> Result<Rocm> {
        let lib = open_hip_lib()?;
        let fns = ffi::resolve(&lib);
        let missing = fns.d0_missing();
        if !missing.is_empty() {
            return Err(Error::new(
                Kind::Unavailable,
                format!(
                    "HIP D0 surface incomplete (missing: {})",
                    missing.join(", ")
                ),
            ));
        }
        let api = Arc::new(HipApi { lib, fns });
        // SAFETY: hipInit(0) is the documented initialization call.
        let f = api.fns.hipInit.expect("bound");
        ffi::check(&api.fns, "hipInit", unsafe { f(0) })?;
        let mut count: i32 = 0;
        // SAFETY: out-param writes one int.
        ffi::check(&api.fns, "hipGetDeviceCount", unsafe {
            (api.fns.hipGetDeviceCount.expect("bound"))(&mut count)
        })?;
        if count <= 0 {
            return Err(Error::new(
                Kind::Unavailable,
                "HIP reports no visible devices (hipGetDeviceCount == 0)",
            ));
        }
        if ordinal < 0 || ordinal >= count {
            return Err(Error::new(
                Kind::Unavailable,
                format!("HIP device ordinal {ordinal} out of range [0, {count})"),
            ));
        }
        // SAFETY: hipSetDevice selects the current device for this thread.
        ffi::check(&api.fns, "hipSetDevice", unsafe {
            (api.fns.hipSetDevice.expect("bound"))(ordinal)
        })?;
        let mut rocm = Rocm {
            api,
            ordinal,
            device_count: count,
            device_name: None,
            runtime_version: None,
            driver_version: None,
        };
        rocm.device_name = rocm.device_name_evidence();
        rocm.runtime_version = rocm.version_evidence(false);
        rocm.driver_version = rocm.version_evidence(true);
        Ok(rocm)
    }

    fn device_name_evidence(&self) -> Option<String> {
        let f = self.api.fns.hipDeviceGetName?;
        let mut buf = vec![0u8; 256];
        // SAFETY: hipDeviceGetName writes at most `len` bytes into buf.
        let rc = unsafe {
            f(
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                self.ordinal,
            )
        };
        if rc != 0 {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Some(String::from_utf8_lossy(&buf[..end]).into_owned())
    }

    fn version_evidence(&self, driver: bool) -> Option<i32> {
        let f = if driver {
            self.api.fns.hipDriverGetVersion?
        } else {
            self.api.fns.hipRuntimeGetVersion?
        };
        let mut v: i32 = 0;
        // SAFETY: out-param writes one int.
        let rc = unsafe { f(&mut v) };
        if rc == 0 { Some(v) } else { None }
    }

    /// Full-device synchronization (the legacy default stream is used for
    /// every Phase-J launch; this makes device writes visible to the host
    /// before any readback).
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: hipDeviceSynchronize blocks until all preceding work on
        // this device completes.
        ffi::check(&self.api.fns, "hipDeviceSynchronize", unsafe {
            (self.api.fns.hipDeviceSynchronize.expect("bound"))()
        })
    }

    /// Load an AMDGPU code object from its exact bytes.
    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        Module::load(&self.api, image)
    }
}

/// Open the HIP library through the same candidate chain the probe uses
/// (single-sourced: `probe::HIP_SONAMES` + explicit ROCm lib dir scans).
pub fn open_hip_lib() -> Result<Lib> {
    let sonames = crate::backend::rocm::probe::HIP_SONAMES;
    let candidates = crate::backend::rocm::loader::lib_candidates(sonames);
    let names = candidates.iter().map(String::as_str).collect::<Vec<_>>();
    Lib::open_candidates(names).map_err(|e| Error::new(Kind::Unavailable, e))
}

/// The module handle + its owning API. `hipModuleUnload` runs when the last
/// `Arc<ModuleInner>` drops — i.e. only after every `Function` resolved from
/// the module is gone (structural lifetime).
pub struct ModuleInner {
    pub api: Arc<HipApi>,
    pub handle: ffi::hipModule_t,
}

// SAFETY: mirrors `loader::Lib`: the HIP module handle is an opaque runtime
// handle; the runtime is used from one thread (the courts) and every use
// happens while the owning API is alive. The impl exists so the module can
// be shared with its `Function`s through `Arc` (structural lifetime).
unsafe impl Send for ModuleInner {}
unsafe impl Sync for ModuleInner {}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: the handle was returned by hipModuleLoadData and is
            // still owned (this is the last Arc); no Function outlives it by
            // construction.
            unsafe {
                (self.api.fns.hipModuleUnload.expect("bound"))(self.handle);
            }
        }
    }
}

/// A loaded AMDGPU code object.
pub struct Module {
    pub inner: Arc<ModuleInner>,
}

impl Module {
    /// Load an AMDGPU code object from its exact bytes.
    pub fn load(api: &Arc<HipApi>, image: &[u8]) -> Result<Module> {
        if image.is_empty() {
            return Err(Error::new(Kind::Malformed, "empty code object image"));
        }
        let mut handle: ffi::hipModule_t = std::ptr::null_mut();
        // SAFETY: hipModuleLoadData parses the image bytes; handle is an
        // out-param. The image lives for the call.
        ffi::check(&api.fns, "hipModuleLoadData", unsafe {
            (api.fns.hipModuleLoadData.expect("bound"))(
                &mut handle,
                image.as_ptr() as *const c_void,
            )
        })?;
        Ok(Module {
            inner: Arc::new(ModuleInner {
                api: api.clone(),
                handle,
            }),
        })
    }

    /// Resolve a kernel entry by its exact exported name. The returned
    /// `Function` retains this module, so the module stays loaded (and the
    /// API stays open) for as long as the function exists.
    pub fn function(&self, entry: &str) -> Result<Function> {
        Function::get(self.inner.clone(), entry)
    }

    /// The owning API (shared).
    pub fn api(&self) -> &Arc<HipApi> {
        &self.inner.api
    }

    /// The module handle.
    pub fn handle(&self) -> ffi::hipModule_t {
        self.inner.handle
    }
}

/// One resolved kernel entry; retains its module owner.
pub struct Function {
    module: Arc<ModuleInner>,
    pub handle: ffi::hipFunction_t,
    /// Entry name (evidence + marshalling identity).
    pub entry: String,
}

impl Function {
    /// # SAFETY
    /// `module` must outlive the returned `Function` (it does: the function
    /// retains the `Arc`). The module must be a live, loaded module.
    pub fn get(module: Arc<ModuleInner>, entry: &str) -> Result<Function> {
        let cname = std::ffi::CString::new(entry)
            .map_err(|_| Error::new(Kind::Malformed, "kernel entry name contains NUL"))?;
        let mut handle: ffi::hipFunction_t = std::ptr::null_mut();
        // SAFETY: out-param; cname is NUL-terminated and lives for the call;
        // the module handle is owned by `module` and alive.
        ffi::check(&module.api.fns, "hipModuleGetFunction", unsafe {
            (module.api.fns.hipModuleGetFunction.expect("bound"))(
                &mut handle,
                module.handle,
                cname.as_ptr(),
            )
        })?;
        Ok(Function {
            module,
            handle,
            entry: entry.to_string(),
        })
    }

    /// Launch with the frozen grid-stride geometry contract: `grid` must be
    /// valid and `threads` must equal the actual workgroup size (the AMD
    /// kernels take `blocks_x`/`threads_x` as parameters AND are launched
    /// with those exact dimensions — `device::geom`).
    ///
    /// `args` are the kernel argument values in declaration order; pointer
    /// arguments carry the device address (see [`Arg`]).
    pub fn launch(&self, grid: Grid, args: &[Arg]) -> Result<()> {
        if !grid.valid() {
            return Err(Error::new(
                Kind::Malformed,
                "launch geometry must satisfy the frozen contract (blocks_x > 0 && threads_x > 0)",
            ));
        }
        let slots = marshal(args);
        let params = slots
            .iter()
            .map(|s| s as *const u64 as *const c_void)
            .collect::<Vec<_>>();
        // SAFETY: hipModuleLaunchKernel launches `f` with the exact grid/
        // block dims declared to the kernel; `params` points to the argument
        // value slots (each slot aligned and sized per its Arg kind), which
        // live for the call. sharedMemBytes 0, stream = legacy default.
        let rc = unsafe {
            (self.module.api.fns.hipModuleLaunchKernel.expect("bound"))(
                self.handle,
                grid.blocks_x,
                1,
                1,
                grid.threads_x,
                1,
                1,
                0,
                std::ptr::null_mut(), // legacy default stream
                params.as_ptr() as *mut *const c_void,
                std::ptr::null_mut(),
            )
        };
        ffi::check(&self.module.api.fns, "hipModuleLaunchKernel", rc)
    }

    /// The owning API (shared).
    pub fn api(&self) -> &Arc<HipApi> {
        &self.module.api
    }
}

/// One kernel argument value, in kernel declaration order. Pointer
/// arguments carry the device address (a machine word); scalar arguments
/// carry their exact value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arg {
    /// Device pointer / machine word (8 bytes).
    Ptr(u64),
    /// 64-bit scalar (e.g. `frames: u64`).
    U64(u64),
    /// 32-bit scalar (e.g. `blocks_x: u32`).
    U32(u32),
}

/// Materialize argument value slots and return one pointer per argument
/// (the `kernelParams` array of `hipModuleLaunchKernel`). Each slot is an
/// 8-byte cell; u32 args occupy the low 4 bytes of their cell and are read
/// as u32 by the driver. Slots must outlive the launch call (they do: the
/// returned Vec owns them and callers launch within the same scope).
pub fn marshal(args: &[Arg]) -> Vec<u64> {
    let mut slots = vec![0u64; args.len()];
    for (i, a) in args.iter().enumerate() {
        slots[i] = match a {
            Arg::Ptr(v) | Arg::U64(v) => *v,
            Arg::U32(v) => u64::from(*v),
        };
    }
    slots
}

/// Device memory (RAII; freed on drop). Retains the owning `Arc<HipApi>`,
/// so `hipFree` can never run through an unloaded library.
pub struct DeviceBuffer {
    pub api: Arc<HipApi>,
    /// Device pointer (0 = invalid).
    pub ptr: u64,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn alloc(api: &Arc<HipApi>, bytes: usize) -> Result<DeviceBuffer> {
        if bytes == 0 {
            return Err(Error::new(Kind::Malformed, "zero-byte device allocation"));
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-param; hipMalloc returns a device pointer.
        ffi::check(&api.fns, "hipMalloc", unsafe {
            (api.fns.hipMalloc.expect("bound"))(&mut ptr, bytes)
        })?;
        Ok(DeviceBuffer {
            api: api.clone(),
            ptr: ptr as u64,
            bytes,
        })
    }

    /// Host -> device copy of exactly `bytes` (checked).
    pub fn upload(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.bytes {
            return Err(Error::new(
                Kind::Malformed,
                "upload exceeds device buffer size",
            ));
        }
        // SAFETY: dst is `self.bytes` of device memory; src is `bytes.len()`
        // live host bytes; kind H2D.
        ffi::check(&self.api.fns, "hipMemcpy(H2D)", unsafe {
            (self.api.fns.hipMemcpy.expect("bound"))(
                self.ptr as *mut c_void,
                bytes.as_ptr() as *const c_void,
                bytes.len(),
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
            )
        })
    }

    /// Device -> host copy of exactly `out.len()` bytes (checked).
    pub fn download(&self, out: &mut [u8]) -> Result<()> {
        if out.len() > self.bytes {
            return Err(Error::new(
                Kind::Malformed,
                "download exceeds device buffer size",
            ));
        }
        // SAFETY: src is `self.bytes` of device memory; dst is `out.len()`
        // live host bytes; kind D2H.
        ffi::check(&self.api.fns, "hipMemcpy(D2H)", unsafe {
            (self.api.fns.hipMemcpy.expect("bound"))(
                out.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                out.len(),
                ffi::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        })
    }

    /// Device pointer as a kernel argument word.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: the pointer was returned by hipMalloc and is still
            // owned by self; the API (library) is alive because self retains
            // it.
            unsafe {
                (self.api.fns.hipFree.expect("bound"))(self.ptr as *mut c_void);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D1: host-memory registration (the ALSA mmap endpoint region)
// ---------------------------------------------------------------------------

/// Registration flags used by the D1 attempt: `hipHostRegisterMapped` (map
/// into the HIP address space so the device can write the region) — the HIP
/// analogue of CUDA `CU_MEMHOSTREGISTER_DEVICEMAP`. PORTABLE is deliberately
/// NOT set: the registration is process-local by design.
pub const D1_REGISTER_FLAGS: u32 = ffi::HIP_HOST_REGISTER_MAPPED;

/// A registered host-memory range (RAII): unregistered on drop. Retains the
/// owning `Arc<HipApi>`.
pub struct HostRegistration {
    pub api: Arc<HipApi>,
    /// Exact host pointer passed to hipHostRegister.
    pub host_ptr: u64,
    /// Exact byte length passed to hipHostRegister.
    pub bytes: usize,
    /// Flags passed (D1_REGISTER_FLAGS).
    pub flags: u32,
    /// Device-visible pointer for `host_ptr` (hipHostGetDevicePointer),
    /// valid when the registration is alive.
    pub device_ptr: Option<u64>,
}

/// Result of one registration attempt: success carries the RAII handle;
/// failure carries the exact HIP rc + message (never silently swallowed).
#[allow(clippy::large_enum_variant)] // the success payload is the RAII handle itself
pub enum RegistrationAttempt {
    Registered(HostRegistration),
    /// The HIP surface is missing a required D1 symbol.
    MissingSymbol(String),
    Failed {
        rc: i32,
        message: String,
    },
}

/// Attempt to register exactly `[base, base+len)` of an existing mapping.
///
/// # SAFETY
/// `base..base+len` must be a live, mapped, writable host range for the whole
/// call and for the lifetime of the returned `HostRegistration`; the caller
/// (the D1 court) holds the ALSA mapping open for that window.
pub unsafe fn attempt_register(api: &Arc<HipApi>, base: u64, len: usize) -> RegistrationAttempt {
    let fns = &api.fns;
    if !fns.d1_ready() {
        let missing = fns.d1_missing();
        return RegistrationAttempt::MissingSymbol(missing.join(", "));
    }
    let host_ptr = base as *mut c_void;
    // SAFETY: caller guarantees the range is live and writable for the
    // registration (and unregister below).
    let rc = unsafe { (fns.hipHostRegister.expect("bound"))(host_ptr, len, D1_REGISTER_FLAGS) };
    if rc != 0 {
        return RegistrationAttempt::Failed {
            rc,
            message: ffi::error_string(fns, rc),
        };
    }
    // hipHostGetDevicePointer(flags = 0) — the only legal value.
    let mut dptr: *mut c_void = std::ptr::null_mut();
    // SAFETY: out-param; the pointer was just registered.
    let rc2 = unsafe { (fns.hipHostGetDevicePointer.expect("bound"))(&mut dptr, host_ptr, 0) };
    if rc2 != 0 {
        // Registration succeeded but the device pointer is unobtainable: the
        // region is not usable from the device. Unregister (best effort) and
        // report the exact failure.
        // SAFETY: unregister the pointer registered above.
        unsafe { (fns.hipHostUnregister.expect("bound"))(host_ptr) };
        return RegistrationAttempt::Failed {
            rc: rc2,
            message: ffi::error_string(fns, rc2),
        };
    }
    RegistrationAttempt::Registered(HostRegistration {
        api: api.clone(),
        host_ptr: base,
        bytes: len,
        flags: D1_REGISTER_FLAGS,
        device_ptr: Some(dptr as u64),
    })
}

impl HostRegistration {
    /// Classify a registration failure rc into the evidence vocabulary.
    /// `d1_ready` comes from the probe; the classifier never invents support
    /// that was not probed.
    pub fn classify(rc: i32, d1_ready: bool) -> (crate::status::Verdict, String) {
        use crate::status::Verdict as V;
        if !d1_ready {
            return (
                V::UnsupportedByApi,
                "HIP D1 surface (hipHostRegister/hipHostGetDevicePointer/hipHostUnregister) \
                 does not resolve in this runtime"
                    .into(),
            );
        }
        match rc {
            ffi::HIP_ERROR_NOT_SUPPORTED => (
                V::UnsupportedByApi,
                format!(
                    "rc {rc} (HIP_ERROR_NOT_SUPPORTED): this memory class is not registerable; \
                     recorded exactly, no substitute buffer"
                ),
            ),
            ffi::HIP_ERROR_NO_DEVICE => (
                V::UnsupportedByHardware,
                format!("rc {rc} (HIP_ERROR_NO_DEVICE): no device is available to map the region"),
            ),
            _ => (
                V::Inconclusive,
                "registration failed with an unclassified rc; exact rc + HIP string recorded \
                 in the receipt"
                    .into(),
            ),
        }
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        if self.host_ptr != 0 {
            // SAFETY: unregister exactly what was registered; the caller
            // guarantees the mapping outlives this drop; the API (library)
            // is alive because self retains it.
            unsafe {
                (self.api.fns.hipHostUnregister.expect("bound"))(self.host_ptr as *mut c_void);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marshal_slots_match_arg_order_and_width() {
        let slots = marshal(&[
            Arg::Ptr(0x1234),
            Arg::U32(64),
            Arg::U64(512),
            Arg::U32(0xffff_ffff),
        ]);
        assert_eq!(slots, vec![0x1234, 64, 512, 0xffff_ffff]);
        assert_eq!(slots.len(), 4);
    }

    #[test]
    fn invalid_geometry_is_rejected_before_launch() {
        let g = Grid::new(0, 0);
        assert!(!g.valid());
    }

    #[test]
    fn classifier_distinguishes_api_from_hardware() {
        let (v, _) = HostRegistration::classify(1, false);
        assert_eq!(v, crate::status::Verdict::UnsupportedByApi);
        let (v, _) = HostRegistration::classify(ffi::HIP_ERROR_NOT_SUPPORTED, true);
        assert_eq!(v, crate::status::Verdict::UnsupportedByApi);
        let (v, _) = HostRegistration::classify(777, true);
        assert_eq!(v, crate::status::Verdict::Inconclusive);
    }

    #[test]
    fn resources_hold_the_api_arc() {
        // Hostile-lifetime guard without a device: `Rocm`/resources are
        // built around one `Arc<HipApi>`; the `Arc` is the structural
        // guarantee that dropping the session cannot unload the library
        // while a resource exists. We exercise the graph with a libc-backed
        // placeholder (no HIP calls happen: nothing is allocated/launched).
        let lib = Lib::open("libc.so.6").expect("libc dlopens");
        let api = Arc::new(HipApi {
            lib,
            fns: Fns {
                hipInit: None,
                hipGetDeviceCount: None,
                hipSetDevice: None,
                hipDeviceSynchronize: None,
                hipStreamSynchronize: None,
                hipModuleLoadData: None,
                hipModuleUnload: None,
                hipModuleGetFunction: None,
                hipModuleLaunchKernel: None,
                hipMalloc: None,
                hipFree: None,
                hipMemcpy: None,
                hipHostRegister: None,
                hipHostUnregister: None,
                hipHostGetDevicePointer: None,
                hipGetErrorString: None,
                hipRuntimeGetVersion: None,
                hipDriverGetVersion: None,
                hipDeviceGetName: None,
            },
        });
        assert_eq!(Arc::strong_count(&api), 1);
        // A device buffer (conceptually) retains the api.
        let buf = DeviceBuffer {
            api: api.clone(),
            ptr: 0, // never freed (drop guards on ptr != 0)
            bytes: 4096,
        };
        assert_eq!(Arc::strong_count(&api), 2);
        drop(api);
        // The buffer still holds the library alive.
        assert_eq!(Arc::strong_count(&buf.api), 1);
        drop(buf);
    }
}

//! Native-Rust HIP runtime session (Phase J) — RAII over the resolved
//! `ffi::Fns`, mirroring `backend::cuda::driver` for the AMD surface.
//!
//! Ownership (structural, not comments): one `HipApi` owns the dlopen'd
//! `Lib` + the resolved `Fns`, and one `HipDevice` binds that API to a
//! **device ordinal** — HIP's device choice is thread-local, so resource
//! identity is API lifetime **plus** device affinity:
//!
//! ```text
//!                  HipApi ──> Lib (dlopen) + Fns
//!                     │
//!                HipDevice { api, ordinal }
//!       ┌─────────────┼─────────────┐
//!       ↓             ↓             ↓
//!    Module      DeviceBuffer   HostRegistration
//!       ↓
//!    Function
//! ```
//!
//! Every current-device-dependent operation first re-establishes its owner's
//! device (`make_current()` → `hipSetDevice`, already part of the frozen D0
//! surface), so a resource can never silently act on whichever device a
//! later `Rocm::open` left current (review-2 finding).
//!
//! * `Rocm` — one HIP session: init, device count/identity evidence, and
//!   synchronized launch context (legacy default stream; every Phase-J
//!   launch is fully synchronized before the host reads anything);
//! * `Module`/`Function` — an AMDGPU code object loaded from bytes
//!   (`hipModuleLoadData`) and its kernel entries (a `Function` retains its
//!   module owner);
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

/// The API bound to one HIP device ordinal. HIP device selection is
/// thread-local, so carrying the ordinal is what makes affinity structural:
/// every operation on a resource built from this device first makes this
/// ordinal current.
pub struct HipDevice {
    pub api: Arc<HipApi>,
    pub ordinal: i32,
}

impl std::fmt::Debug for HipDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipDevice")
            .field("ordinal", &self.ordinal)
            .field("api", &self.api.lib.name)
            .finish()
    }
}

impl HipDevice {
    /// Select this device as the current device for this host thread.
    /// `hipSetDevice` is part of the frozen D0 surface, so this needs no ABI
    /// expansion.
    pub fn make_current(&self) -> Result<()> {
        let f = self.api.fns.hipSetDevice.ok_or_else(|| {
            Error::new(
                Kind::Unavailable,
                "hipSetDevice not resolved; cannot establish device affinity",
            )
        })?;
        // SAFETY: hipSetDevice selects the thread-local current device.
        ffi::check(&self.api.fns, "hipSetDevice", unsafe { f(self.ordinal) })
    }

    /// Best-effort re-selection for `Drop` paths: an error here must not
    /// panic during unwinding, but a stale current device must never be
    /// allowed to turn a free/unregister into a call on the wrong device.
    fn make_current_best_effort(&self) {
        let _ = self.make_current();
    }
}

/// HIP session (RAII over one `Arc<HipDevice>`).
pub struct Rocm {
    /// The device this session is bound to (API + ordinal).
    pub device: Arc<HipDevice>,
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
            .field("device", &self.device)
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
        let device = Arc::new(HipDevice { api, ordinal });
        device.make_current()?;
        let mut rocm = Rocm {
            device,
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
        let f = self.device.api.fns.hipDeviceGetName?;
        let mut buf = vec![0u8; 256];
        // SAFETY: hipDeviceGetName writes at most `len` bytes into buf.
        let rc = unsafe {
            f(
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                self.device.ordinal,
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
            self.device.api.fns.hipDriverGetVersion?
        } else {
            self.device.api.fns.hipRuntimeGetVersion?
        };
        let mut v: i32 = 0;
        // SAFETY: out-param writes one int.
        let rc = unsafe { f(&mut v) };
        if rc == 0 { Some(v) } else { None }
    }

    /// Full-device synchronization **on this session's device** (the legacy
    /// default stream is used for every Phase-J launch; this makes device
    /// writes visible to the host before any readback).
    pub fn synchronize(&self) -> Result<()> {
        self.device.make_current()?;
        // SAFETY: hipDeviceSynchronize blocks until all preceding work on
        // the current device completes.
        ffi::check(&self.device.api.fns, "hipDeviceSynchronize", unsafe {
            (self.device.api.fns.hipDeviceSynchronize.expect("bound"))()
        })
    }

    /// Load an AMDGPU code object from its exact bytes onto this device.
    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        Module::load(&self.device, image)
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

/// The module handle + its owning device. `hipModuleUnload` runs when the
/// last `Arc<ModuleInner>` drops — i.e. only after every `Function` resolved
/// from the module is gone (structural lifetime) — and only after
/// re-establishing the owning device.
pub struct ModuleInner {
    pub device: Arc<HipDevice>,
    pub handle: ffi::hipModule_t,
}

// SAFETY: mirrors `loader::Lib`: the HIP module handle is an opaque runtime
// handle; the runtime is used from one thread (the courts) and every use
// re-establishes the owning device. The impl exists so the module can be
// shared with its `Function`s through `Arc` (structural lifetime).
unsafe impl Send for ModuleInner {}
unsafe impl Sync for ModuleInner {}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // Best-effort device re-selection: unload must target the device
            // the module was loaded on.
            self.device.make_current_best_effort();
            // SAFETY: the handle was returned by hipModuleLoadData and is
            // still owned (this is the last Arc); no Function outlives it by
            // construction.
            unsafe {
                (self.device.api.fns.hipModuleUnload.expect("bound"))(self.handle);
            }
        }
    }
}

/// A loaded AMDGPU code object (device-affine).
pub struct Module {
    pub inner: Arc<ModuleInner>,
}

impl Module {
    /// Load an AMDGPU code object from its exact bytes onto `device`.
    pub fn load(device: &Arc<HipDevice>, image: &[u8]) -> Result<Module> {
        if image.is_empty() {
            return Err(Error::new(Kind::Malformed, "empty code object image"));
        }
        device.make_current()?;
        let mut handle: ffi::hipModule_t = std::ptr::null_mut();
        // SAFETY: hipModuleLoadData parses the image bytes on the current
        // device; handle is an out-param; the image lives for the call.
        ffi::check(&device.api.fns, "hipModuleLoadData", unsafe {
            (device.api.fns.hipModuleLoadData.expect("bound"))(
                &mut handle,
                image.as_ptr() as *const c_void,
            )
        })?;
        Ok(Module {
            inner: Arc::new(ModuleInner {
                device: device.clone(),
                handle,
            }),
        })
    }

    /// Resolve a kernel entry by its exact exported name. The returned
    /// `Function` retains this module (and therefore the device + API).
    pub fn function(&self, entry: &str) -> Result<Function> {
        Function::get(self.inner.clone(), entry)
    }

    /// The owning device (shared).
    pub fn device(&self) -> &Arc<HipDevice> {
        &self.inner.device
    }

    /// The module handle.
    pub fn handle(&self) -> ffi::hipModule_t {
        self.inner.handle
    }
}

/// One resolved kernel entry; retains its module owner (and thus the device
/// and API).
pub struct Function {
    module: Arc<ModuleInner>,
    pub handle: ffi::hipFunction_t,
    /// Entry name (evidence + marshalling identity).
    pub entry: String,
}

impl Function {
    /// Resolve `entry` from `module`. The module must be live (it is
    /// retained by the returned `Function`).
    pub fn get(module: Arc<ModuleInner>, entry: &str) -> Result<Function> {
        let cname = std::ffi::CString::new(entry)
            .map_err(|_| Error::new(Kind::Malformed, "kernel entry name contains NUL"))?;
        module.device.make_current()?;
        let mut handle: ffi::hipFunction_t = std::ptr::null_mut();
        // SAFETY: out-param; cname is NUL-terminated and lives for the call;
        // the module handle is owned by `module` and alive.
        ffi::check(&module.device.api.fns, "hipModuleGetFunction", unsafe {
            (module.device.api.fns.hipModuleGetFunction.expect("bound"))(
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
    /// The launch always targets this function's owning device (a previous
    /// `Rocm::open` on another ordinal cannot redirect it).
    pub fn launch(&self, grid: Grid, args: &[Arg]) -> Result<()> {
        if !grid.valid() {
            return Err(Error::new(
                Kind::Malformed,
                "launch geometry must satisfy the frozen contract (blocks_x > 0 && threads_x > 0)",
            ));
        }
        self.module.device.make_current()?;
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
            (self
                .module
                .device
                .api
                .fns
                .hipModuleLaunchKernel
                .expect("bound"))(
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
        ffi::check(&self.module.device.api.fns, "hipModuleLaunchKernel", rc)
    }

    /// The owning device (shared).
    pub fn device(&self) -> &Arc<HipDevice> {
        &self.module.device
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

/// Device memory (RAII; freed on drop). Retains the owning device, so
/// `hipFree` targets the device the buffer was allocated on and cannot run
/// through an unloaded library.
pub struct DeviceBuffer {
    pub device: Arc<HipDevice>,
    /// Device pointer (0 = invalid).
    pub ptr: u64,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn alloc(device: &Arc<HipDevice>, bytes: usize) -> Result<DeviceBuffer> {
        if bytes == 0 {
            return Err(Error::new(Kind::Malformed, "zero-byte device allocation"));
        }
        device.make_current()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-param; hipMalloc returns a device pointer.
        ffi::check(&device.api.fns, "hipMalloc", unsafe {
            (device.api.fns.hipMalloc.expect("bound"))(&mut ptr, bytes)
        })?;
        Ok(DeviceBuffer {
            device: device.clone(),
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
        self.device.make_current()?;
        // SAFETY: dst is `self.bytes` of device memory; src is `bytes.len()`
        // live host bytes; kind H2D.
        ffi::check(&self.device.api.fns, "hipMemcpy(H2D)", unsafe {
            (self.device.api.fns.hipMemcpy.expect("bound"))(
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
        self.device.make_current()?;
        // SAFETY: src is `self.bytes` of device memory; dst is `out.len()`
        // live host bytes; kind D2H.
        ffi::check(&self.device.api.fns, "hipMemcpy(D2H)", unsafe {
            (self.device.api.fns.hipMemcpy.expect("bound"))(
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
            self.device.make_current_best_effort();
            // SAFETY: the pointer was returned by hipMalloc on this device
            // and is still owned by self; the API (library) is alive because
            // self retains it.
            unsafe {
                (self.device.api.fns.hipFree.expect("bound"))(self.ptr as *mut c_void);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D1: host-memory registration (the ALSA mmap endpoint region)
// ---------------------------------------------------------------------------

/// Registration flags used by the D1 attempt: `hipHostRegisterMapped` (map
/// into the current device's address space so the device can write the
/// region) — the HIP analogue of CUDA `CU_MEMHOSTREGISTER_DEVICEMAP`.
/// PORTABLE is deliberately NOT set: the registration is process-local by
/// design.
pub const D1_REGISTER_FLAGS: u32 = ffi::HIP_HOST_REGISTER_MAPPED;

/// A registered host-memory range (RAII): unregistered on drop. Retains the
/// owning device (registration is per current device).
pub struct HostRegistration {
    pub device: Arc<HipDevice>,
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

/// `rc` recorded when a veol-side pre-call step failed (e.g. re-selecting
/// the owner device) — never a HIP error code.
pub const PRE_CALL_FAILURE_RC: i32 = -1;

/// Attempt to register exactly `[base, base+len)` of an existing mapping
/// **on the owner device** (hipHostRegister maps into the current device's
/// address space, so the ordinal must be established first).
///
/// # SAFETY
/// `base..base+len` must be a live, mapped, writable host range for the whole
/// call and for the lifetime of the returned `HostRegistration`; the caller
/// (the D1 court) holds the ALSA mapping open for that window.
pub unsafe fn attempt_register(
    device: &Arc<HipDevice>,
    base: u64,
    len: usize,
) -> RegistrationAttempt {
    let fns = &device.api.fns;
    if !fns.d1_ready() {
        let missing = fns.d1_missing();
        return RegistrationAttempt::MissingSymbol(missing.join(", "));
    }
    if let Err(e) = device.make_current() {
        return RegistrationAttempt::Failed {
            rc: PRE_CALL_FAILURE_RC,
            message: format!("could not select the owner device: {e}"),
        };
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
        device: device.clone(),
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
            PRE_CALL_FAILURE_RC => (
                V::Inconclusive,
                "veol-side pre-call failure (device re-selection) before hipHostRegister; no \
                 HIP rc exists for this step"
                    .into(),
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
            self.device.make_current_best_effort();
            // SAFETY: unregister exactly what was registered, on the device
            // it was registered for; the caller guarantees the mapping
            // outlives this drop; the API (library) is alive because self
            // retains it.
            unsafe {
                (self.device.api.fns.hipHostUnregister.expect("bound"))(
                    self.host_ptr as *mut c_void,
                );
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
        let (v, _) = HostRegistration::classify(PRE_CALL_FAILURE_RC, true);
        assert_eq!(v, crate::status::Verdict::Inconclusive);
        let (v, _) = HostRegistration::classify(777, true);
        assert_eq!(v, crate::status::Verdict::Inconclusive);
    }

    #[test]
    fn resources_hold_the_device_arc() {
        // Hostile-lifetime guard without a device: resources are built
        // around one `Arc<HipDevice>` (which owns `Arc<HipApi>` which owns
        // `Lib`); dropping the session cannot unload the library while a
        // resource exists.
        let lib = Lib::open("libc.so.6").expect("libc dlopens");
        let api = Arc::new(HipApi {
            lib,
            fns: empty_fns(),
        });
        let device = Arc::new(HipDevice {
            api: api.clone(),
            ordinal: 0,
        });
        assert_eq!(Arc::strong_count(&device), 1);
        let buf = DeviceBuffer {
            device: device.clone(),
            ptr: 0, // never freed (drop guards on ptr != 0)
            bytes: 4096,
        };
        assert_eq!(Arc::strong_count(&device), 2);
        drop(device);
        // The buffer still holds the device (and therefore the library).
        assert_eq!(Arc::strong_count(&buf.device), 1);
        drop(buf);
    }

    /// An `Fns` with nothing resolved (no call is made with it).
    pub(super) fn empty_fns() -> Fns {
        Fns {
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
        }
    }
}

/// Device-affinity hostile tests: these use a fake HIP function table that
/// records every `hipSetDevice` + operation, proving through the real
/// resource types that each current-device-dependent operation
/// re-establishes its owner's ordinal. One test owns the recorder, so the
/// log can never interleave with another test.
#[cfg(test)]
mod affinity_tests {
    use super::tests::empty_fns;
    use super::*;
    use std::os::raw::{c_int, c_uint};
    use std::sync::Mutex;

    static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());
    /// Serializes the tests that share `LOG` (the recorder is process-wide,
    /// so two concurrent tests would clear each other's entries).
    static SERIAL: Mutex<()> = Mutex::new(());

    fn record(s: &str) {
        LOG.lock().expect("log").push(s.to_string());
    }

    fn take_log() -> Vec<String> {
        std::mem::take(&mut *LOG.lock().expect("log"))
    }

    unsafe extern "C" fn f_set_device(d: c_int) -> ffi::hipError_t {
        record(&format!("set:{d}"));
        0
    }
    unsafe extern "C" fn f_malloc(p: *mut *mut c_void, _n: usize) -> ffi::hipError_t {
        record("malloc");
        // SAFETY: out-param for a fake device pointer.
        unsafe { *p = 0x1000 as *mut c_void };
        0
    }
    unsafe extern "C" fn f_free(_p: *mut c_void) -> ffi::hipError_t {
        record("free");
        0
    }
    unsafe extern "C" fn f_memcpy(
        _d: *mut c_void,
        _s: *const c_void,
        _n: usize,
        kind: ffi::hipMemcpyKind,
    ) -> ffi::hipError_t {
        record(if kind == ffi::HIP_MEMCPY_HOST_TO_DEVICE {
            "copy:h2d"
        } else {
            "copy:d2h"
        });
        0
    }
    unsafe extern "C" fn f_module_load(
        m: *mut ffi::hipModule_t,
        _img: *const c_void,
    ) -> ffi::hipError_t {
        record("module_load");
        // SAFETY: out-param for a fake module handle.
        unsafe { *m = 0x2000 as ffi::hipModule_t };
        0
    }
    unsafe extern "C" fn f_module_unload(_m: ffi::hipModule_t) -> ffi::hipError_t {
        record("module_unload");
        0
    }
    unsafe extern "C" fn f_get_function(
        f: *mut ffi::hipFunction_t,
        _m: ffi::hipModule_t,
        _n: *const c_char,
    ) -> ffi::hipError_t {
        record("get_function");
        // SAFETY: out-param for a fake function handle.
        unsafe { *f = 0x3000 as ffi::hipFunction_t };
        0
    }
    #[allow(clippy::too_many_arguments)]
    unsafe extern "C" fn f_launch(
        _f: ffi::hipFunction_t,
        _gx: c_uint,
        _gy: c_uint,
        _gz: c_uint,
        _bx: c_uint,
        _by: c_uint,
        _bz: c_uint,
        _shared: c_uint,
        _stream: ffi::hipStream_t,
        _params: *mut *const c_void,
        _extra: *mut *const c_void,
    ) -> ffi::hipError_t {
        record("launch");
        0
    }
    unsafe extern "C" fn f_sync() -> ffi::hipError_t {
        record("sync");
        0
    }
    unsafe extern "C" fn f_host_register(
        _p: *mut c_void,
        _n: usize,
        _flags: c_uint,
    ) -> ffi::hipError_t {
        record("register");
        0
    }
    unsafe extern "C" fn f_host_unregister(_p: *mut c_void) -> ffi::hipError_t {
        record("unregister");
        0
    }
    unsafe extern "C" fn f_host_devptr(
        d: *mut *mut c_void,
        _p: *mut c_void,
        _flags: c_uint,
    ) -> ffi::hipError_t {
        record("get_devptr");
        // SAFETY: out-param for a fake device pointer.
        unsafe { *d = 0x4000 as *mut c_void };
        0
    }

    unsafe extern "C" fn f_init(_flags: c_uint) -> ffi::hipError_t {
        0
    }
    unsafe extern "C" fn f_count(p: *mut c_int) -> ffi::hipError_t {
        // SAFETY: out-param writes one int.
        unsafe { *p = 2 };
        0
    }
    unsafe extern "C" fn f_stream_sync(_s: ffi::hipStream_t) -> ffi::hipError_t {
        0
    }
    unsafe extern "C" fn f_set_device_fail(_d: c_int) -> ffi::hipError_t {
        // rc 100 = hipErrorNoDevice (as if the ordinal vanished).
        100
    }

    fn fake_fns() -> Fns {
        let mut f = empty_fns();
        f.hipInit = Some(f_init);
        f.hipGetDeviceCount = Some(f_count);
        f.hipStreamSynchronize = Some(f_stream_sync);
        f.hipSetDevice = Some(f_set_device);
        f.hipMalloc = Some(f_malloc);
        f.hipFree = Some(f_free);
        f.hipMemcpy = Some(f_memcpy);
        f.hipModuleLoadData = Some(f_module_load);
        f.hipModuleUnload = Some(f_module_unload);
        f.hipModuleGetFunction = Some(f_get_function);
        f.hipModuleLaunchKernel = Some(f_launch);
        f.hipDeviceSynchronize = Some(f_sync);
        f.hipHostRegister = Some(f_host_register);
        f.hipHostUnregister = Some(f_host_unregister);
        f.hipHostGetDevicePointer = Some(f_host_devptr);
        f
    }

    fn fake_device(ordinal: i32) -> Arc<HipDevice> {
        let lib = Lib::open("libc.so.6").expect("libc dlopens");
        let api = Arc::new(HipApi {
            lib,
            fns: fake_fns(),
        });
        Arc::new(HipDevice { api, ordinal })
    }

    #[test]
    fn every_operation_restores_its_owner_device() {
        let _serial = SERIAL.lock().expect("serial");
        let d0 = fake_device(0);
        let d1 = fake_device(1);
        let _ = take_log();

        // Module + function on device 0; then device 1 becomes current.
        let module = Module::load(&d0, b"\x7fELFfake").expect("module load");
        let function = module.function("vole_render_d0").expect("function");
        d1.make_current().expect("select 1");
        // Launch must restore device 0 (owner) before launching.
        function
            .launch(
                Grid::new(2, 256),
                &[Arg::Ptr(0x1000), Arg::U32(2), Arg::U32(256)],
            )
            .expect("launch");
        let log = take_log();
        assert_eq!(
            log,
            vec![
                "set:0",
                "module_load",
                "set:0",
                "get_function",
                "set:1",
                "set:0",
                "launch"
            ]
        );

        // Buffers: alloc/upload/download/free all target the owner ordinal.
        let _ = take_log();
        let _ = d1.make_current();
        let buf = DeviceBuffer::alloc(&d0, 64).expect("alloc");
        buf.upload(&[0u8; 64]).expect("upload");
        let mut out = [0u8; 64];
        buf.download(&mut out).expect("download");
        assert_eq!(
            take_log(),
            vec![
                "set:1", "set:0", "malloc", "set:0", "copy:h2d", "set:0", "copy:d2h"
            ]
        );
        // Device 1 becomes current again; dropping the buffer still frees on
        // device 0.
        let _ = take_log();
        let _ = d1.make_current();
        drop(buf);
        assert_eq!(take_log(), vec!["set:1", "set:0", "free"]);

        // Registration always targets the owner ordinal (hipHostRegister
        // maps into the *current* device's address space).
        let _ = take_log();
        let _ = d1.make_current();
        // SAFETY: test-only; the fake never dereferences base..base+len.
        let reg = match unsafe { attempt_register(&d0, 0x1000, 4096) } {
            RegistrationAttempt::Registered(r) => r,
            _ => panic!("fake registration must succeed"),
        };
        assert!(reg.device_ptr.is_some());
        assert_eq!(take_log(), vec!["set:1", "set:0", "register", "get_devptr"]);
        let _ = take_log();
        let _ = d1.make_current();
        drop(reg);
        assert_eq!(take_log(), vec!["set:1", "set:0", "unregister"]);

        // Synchronize targets the session's owner ordinal.
        let _ = take_log();
        let _ = d1.make_current();
        let session = Rocm {
            device: d0.clone(),
            device_count: 2,
            device_name: None,
            runtime_version: None,
            driver_version: None,
        };
        session.synchronize().expect("sync");
        assert_eq!(take_log(), vec!["set:1", "set:0", "sync"]);

        // Module drop unloads on its owner device, after any function died.
        let _ = take_log();
        let _ = d1.make_current();
        drop(function);
        drop(module);
        assert_eq!(take_log(), vec!["set:1", "set:0", "module_unload"]);
    }

    #[test]
    fn registration_reports_pre_call_failure_distinctly() {
        let _serial = SERIAL.lock().expect("serial");
        // A device whose hipSetDevice fails cannot silently register against
        // whatever device is current: the attempt fails with the explicit
        // veol-side sentinel rc (never a fabricated HIP rc), and no register
        // call is made.
        let lib = Lib::open("libc.so.6").expect("libc dlopens");
        let mut fns = fake_fns();
        fns.hipSetDevice = Some(f_set_device_fail);
        let api = Arc::new(HipApi { lib, fns });
        let device = Arc::new(HipDevice { api, ordinal: 3 });
        let _ = take_log();
        // SAFETY: test-only; no call reaches the fake register.
        match unsafe { attempt_register(&device, 0x1000, 4096) } {
            RegistrationAttempt::Failed { rc, message } => {
                assert_eq!(rc, PRE_CALL_FAILURE_RC);
                assert!(message.contains("owner device"), "{message}");
            }
            _ => panic!("must fail before the HIP call"),
        }
        let log = take_log();
        assert!(!log.contains(&"register".to_string()), "log: {log:?}");
    }
}

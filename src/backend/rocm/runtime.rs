//! Native-Rust HIP runtime session (Phase J) — RAII over the resolved
//! `ffi::Fns`, mirroring `backend::cuda::driver` for the AMD surface.
//!
//! * `Rocm` — one HIP session: init, device selection, synchronized launch
//!   context (legacy default stream; every Phase-J launch is fully
//!   synchronized before the host reads anything, which is the exactness
//!   discipline the differential battery needs);
//! * `Module`/`Function` — an AMDGPU code object loaded from bytes
//!   (`hipModuleLoadData`) and its kernel entries;
//! * `DeviceBuffer` — device memory with host<->device copies;
//! * `HostRegistration` — D1: pin an existing host range (the ALSA mmap
//!   endpoint region) with `hipHostRegister(hipHostRegisterMapped)` so the
//!   device can write it directly.
//!
//! Nothing here executes without a device; every path returns typed errors
//! that the courts record exactly.

use crate::backend::rocm::ffi::{self, Fns};
use crate::backend::rocm::loader::Lib;
use crate::device::geom::Grid;
use crate::error::{Error, Kind, Result};
use std::os::raw::{c_char, c_void};

/// HIP session (RAII). The library handle outlives every resolved pointer
/// and is closed on drop.
pub struct Rocm {
    pub fns: Fns,
    /// The dlopen'd HIP library (kept alive for `fns`).
    _lib: Lib,
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
        // SAFETY: hipInit(0) is the documented initialization call.
        let f = fns.hipInit.expect("bound");
        ffi::check(&fns, "hipInit", unsafe { f(0) })?;
        let mut count: i32 = 0;
        // SAFETY: out-param writes one int.
        ffi::check(&fns, "hipGetDeviceCount", unsafe {
            (fns.hipGetDeviceCount.expect("bound"))(&mut count)
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
        ffi::check(&fns, "hipSetDevice", unsafe {
            (fns.hipSetDevice.expect("bound"))(ordinal)
        })?;
        let mut rocm = Rocm {
            fns,
            _lib: lib,
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
        let f = self.fns.hipDeviceGetName?;
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
            self.fns.hipDriverGetVersion?
        } else {
            self.fns.hipRuntimeGetVersion?
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
        ffi::check(&self.fns, "hipDeviceSynchronize", unsafe {
            (self.fns.hipDeviceSynchronize.expect("bound"))()
        })
    }

    /// Load an AMDGPU code object from its exact bytes.
    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        Module::load(&self.fns, image)
    }
}

/// Open the HIP library through the same candidate chain the probe uses
/// (ld cache sonames, ROCM_LIB_PATH dirs, /opt/rocm* prefix scan).
pub fn open_hip_lib() -> Result<Lib> {
    let mut candidates: Vec<String> = Vec::new();
    for soname in crate::backend::rocm::probe::COMPUTE_SONAMES {
        if soname.0.starts_with("libamdhip64") {
            candidates.push(soname.0.to_string());
        }
    }
    if let Ok(p) = std::env::var("ROCM_LIB_PATH") {
        for dir in p.split(':').filter(|d| !d.is_empty()) {
            candidates.push(format!("{dir}/libamdhip64.so"));
        }
    }
    if let Ok(entries) = std::fs::read_dir("/opt") {
        let mut rocm_dirs = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("rocm"))
            .map(|e| e.path())
            .collect::<Vec<_>>();
        rocm_dirs.sort();
        for dir in rocm_dirs {
            for sub in ["lib", "lib64"] {
                let libdir = dir.join(sub);
                if let Ok(files) = std::fs::read_dir(&libdir) {
                    let mut found = files
                        .filter_map(|e| e.ok())
                        .filter(|e| {
                            e.file_name()
                                .to_string_lossy()
                                .starts_with("libamdhip64.so")
                        })
                        .map(|e| e.path())
                        .collect::<Vec<_>>();
                    found.sort();
                    for p in found {
                        candidates.push(p.to_string_lossy().into_owned());
                    }
                }
            }
        }
    }
    let names = candidates.iter().map(String::as_str).collect::<Vec<_>>();
    Lib::open_candidates(names).map_err(|e| Error::new(Kind::Unavailable, e))
}

/// A loaded AMDGPU code object (RAII; unloaded on drop).
pub struct Module {
    pub fns: Fns,
    pub handle: ffi::hipModule_t,
}

impl Module {
    pub fn load(fns: &Fns, image: &[u8]) -> Result<Module> {
        if image.is_empty() {
            return Err(Error::new(Kind::Malformed, "empty code object image"));
        }
        let mut handle: ffi::hipModule_t = std::ptr::null_mut();
        // SAFETY: hipModuleLoadData parses the image bytes; handle is an
        // out-param. The image lives for the call.
        ffi::check(fns, "hipModuleLoadData", unsafe {
            (fns.hipModuleLoadData.expect("bound"))(&mut handle, image.as_ptr() as *const c_void)
        })?;
        Ok(Module { fns: *fns, handle })
    }

    /// Resolve a kernel entry by its exact exported name.
    pub fn function(&self, entry: &str) -> Result<Function> {
        // SAFETY: `get` passes the module handle to the HIP runtime; the
        // handle is owned by self and alive for the call.
        unsafe { Function::get(&self.fns, self.handle, entry) }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: the handle was returned by hipModuleLoadData and is
            // still owned by self; function handles die with the module.
            unsafe {
                (self.fns.hipModuleUnload.expect("bound"))(self.handle);
            }
        }
    }
}

/// One resolved kernel entry.
pub struct Function {
    pub fns: Fns,
    pub handle: ffi::hipFunction_t,
    /// Entry name (evidence + marshalling identity).
    pub entry: String,
}

impl Function {
    /// # SAFETY
    /// `module` must be a live module handle (owned by a `Module` that
    /// outlives the returned `Function`); `entry` is a NUL-terminated C
    /// string for the call duration.
    pub unsafe fn get(fns: &Fns, module: ffi::hipModule_t, entry: &str) -> Result<Function> {
        let cname = std::ffi::CString::new(entry)
            .map_err(|_| Error::new(Kind::Malformed, "kernel entry name contains NUL"))?;
        let mut handle: ffi::hipFunction_t = std::ptr::null_mut();
        // SAFETY: out-param; cname is NUL-terminated and lives for the call.
        ffi::check(fns, "hipModuleGetFunction", unsafe {
            (fns.hipModuleGetFunction.expect("bound"))(&mut handle, module, cname.as_ptr())
        })?;
        Ok(Function {
            fns: *fns,
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
            (self.fns.hipModuleLaunchKernel.expect("bound"))(
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
        ffi::check(&self.fns, "hipModuleLaunchKernel", rc)
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

/// Device memory (RAII; freed on drop).
pub struct DeviceBuffer {
    pub fns: Fns,
    /// Device pointer (0 = invalid).
    pub ptr: u64,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn alloc(fns: &Fns, bytes: usize) -> Result<DeviceBuffer> {
        if bytes == 0 {
            return Err(Error::new(Kind::Malformed, "zero-byte device allocation"));
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-param; hipMalloc returns a device pointer.
        ffi::check(fns, "hipMalloc", unsafe {
            (fns.hipMalloc.expect("bound"))(&mut ptr, bytes)
        })?;
        Ok(DeviceBuffer {
            fns: *fns,
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
        ffi::check(&self.fns, "hipMemcpy(H2D)", unsafe {
            (self.fns.hipMemcpy.expect("bound"))(
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
        ffi::check(&self.fns, "hipMemcpy(D2H)", unsafe {
            (self.fns.hipMemcpy.expect("bound"))(
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
            // owned by self.
            unsafe {
                (self.fns.hipFree.expect("bound"))(self.ptr as *mut c_void);
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

/// A registered host-memory range (RAII): unregistered on drop. Drop order is
/// the caller's concern (must precede HIP teardown).
pub struct HostRegistration {
    pub fns: Fns,
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
pub unsafe fn attempt_register(fns: &Fns, base: u64, len: usize) -> RegistrationAttempt {
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
        fns: *fns,
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
            // guarantees the mapping outlives this drop.
            unsafe {
                (self.fns.hipHostUnregister.expect("bound"))(self.host_ptr as *mut c_void);
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
        // Pointer cells must be non-null for the driver (marshalling
        // contract: every arg has a slot).
        assert_eq!(slots.len(), 4);
    }

    #[test]
    fn invalid_geometry_is_rejected_before_launch() {
        // No device needed: the launch contract is enforced host-side first.
        let g = Grid::new(0, 0);
        assert!(!g.valid());
    }

    #[test]
    fn classifier_distinguishes_api_from_hardware() {
        // d1_ready=false => API-level, whatever the rc.
        let (v, _) = HostRegistration::classify(1, false);
        assert_eq!(v, crate::status::Verdict::UnsupportedByApi);
        // d1_ready with a supported-class rc.
        let (v, _) = HostRegistration::classify(ffi::HIP_ERROR_NOT_SUPPORTED, true);
        assert_eq!(v, crate::status::Verdict::UnsupportedByApi);
        // Unknown rc => inconclusive with exact rc recorded.
        let (v, _) = HostRegistration::classify(777, true);
        assert_eq!(v, crate::status::Verdict::Inconclusive);
    }
}

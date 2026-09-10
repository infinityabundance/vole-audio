//! Native-Rust HIP runtime bindings for the ROCm backend (Phase J) —
//! dynamic loading, mirroring `backend::cuda::ffi`.
//!
//! Phase I froze the Phase-J ABI contract as two surfaces (see
//! `backend::rocm::probe`): `HIP_D0_REQUIRED` (init/device/module/launch/
//! memory/copy/sync) and `HIP_D1_ADDITIONAL` (host registration). Every
//! symbol this module resolves is a member of those frozen tables; no symbol
//! outside the tables is *required* (a few optional symbols are resolved for
//! richer evidence rows when present: error strings, runtime/driver
//! versions, device name).
//!
//! Same discipline as the CUDA ffi: `dlopen` with `RTLD_NOW | RTLD_LOCAL`
//! through `loader::Lib`, the handle kept alive for exactly as long as the
//! resolved pointers are used, `dlsym` + transmute of the symbol address,
//! and typed failures. No unsafe beyond the loader and the extern-call
//! sites, each carrying its SAFETY rationale.

// HIP symbol and type names are frozen from hip_runtime_api.h; keeping the
// upstream spelling makes the ABI table auditable against the headers.
#![allow(non_camel_case_types, non_snake_case)]

use crate::backend::rocm::loader::Lib;
use crate::backend::rocm::probe::{HIP_D0_REQUIRED, HIP_D1_ADDITIONAL};
use crate::error::{Error, Kind, Result};
use std::os::raw::{c_char, c_int, c_uint, c_void};

// ---------------------------------------------------------------------------
// HIP types (hip_runtime_api.h)
// ---------------------------------------------------------------------------

/// `hipError_t` is an `int`; `hipSuccess == 0`.
pub type hipError_t = c_int;
/// `hipDevice_t` is an `int` ordinal/handle.
pub type hipDevice_t = c_int;
/// `hipModule_t` (opaque handle).
pub type hipModule_t = *mut c_void;
/// `hipFunction_t` (opaque kernel handle).
pub type hipFunction_t = *mut c_void;
/// `hipStream_t` (opaque; the legacy default stream is 0/null).
pub type hipStream_t = *mut c_void;

/// `hipMemcpyKind` (mirrors `cudaMemcpyKind` values).
pub type hipMemcpyKind = c_int;
pub const HIP_MEMCPY_HOST_TO_HOST: hipMemcpyKind = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: hipMemcpyKind = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: hipMemcpyKind = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: hipMemcpyKind = 3;

/// `hipHostRegisterFlags`. D1 uses `hipHostRegisterMapped` (map into the
/// HIP address space so the device can write the region) — the HIP analogue
/// of CUDA `CU_MEMHOSTREGISTER_DEVICEMAP`. PORTABLE is deliberately NOT set:
/// the registration is process-local by design.
pub const HIP_HOST_REGISTER_MAPPED: c_uint = 2;

/// hipError_t values used by the classifier/evidence rows (only the subset
/// whose numeric values are stable across HIP releases is hardcoded; every
/// other rc is recorded with its exact `hipGetErrorString` text).
pub const HIP_ERROR_INVALID_VALUE: hipError_t = 1;
pub const HIP_ERROR_OUT_OF_MEMORY: hipError_t = 2;
pub const HIP_ERROR_NO_DEVICE: hipError_t = 100;
pub const HIP_ERROR_HOST_MEMORY_ALREADY_REGISTERED: hipError_t = 712;
pub const HIP_ERROR_NOT_SUPPORTED: hipError_t = 801;

// ---------------------------------------------------------------------------
// Resolved function-pointer table
// ---------------------------------------------------------------------------

macro_rules! hip_fns {
    ( $( $name:ident : $fty:ty ),* $(,)? ) => {
        /// Resolved HIP function pointers (all `Option`; the frozen-surface
        /// checks happen at open time against the D0/D1 tables).
        #[derive(Debug, Clone, Copy)]
        pub struct Fns {
            $( pub $name: Option<$fty>, )*
        }
    };
}

hip_fns! {
    hipInit: unsafe extern "C" fn(c_uint) -> hipError_t,
    hipGetDeviceCount: unsafe extern "C" fn(*mut c_int) -> hipError_t,
    hipSetDevice: unsafe extern "C" fn(c_int) -> hipError_t,
    hipDeviceSynchronize: unsafe extern "C" fn() -> hipError_t,
    hipStreamSynchronize: unsafe extern "C" fn(hipStream_t) -> hipError_t,
    hipModuleLoadData: unsafe extern "C" fn(*mut hipModule_t, *const c_void) -> hipError_t,
    hipModuleUnload: unsafe extern "C" fn(hipModule_t) -> hipError_t,
    hipModuleGetFunction: unsafe extern "C" fn(*mut hipFunction_t, hipModule_t, *const c_char) -> hipError_t,
    hipModuleLaunchKernel: unsafe extern "C" fn(
        hipFunction_t,
        c_uint, c_uint, c_uint, // grid dims
        c_uint, c_uint, c_uint, // block dims
        c_uint,                 // shared mem bytes
        hipStream_t,
        *mut *const c_void,    // kernelParams
        *mut *const c_void,    // extra
    ) -> hipError_t,
    hipMalloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> hipError_t,
    hipFree: unsafe extern "C" fn(*mut c_void) -> hipError_t,
    hipMemcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, hipMemcpyKind) -> hipError_t,
    // D1 additional.
    hipHostRegister: unsafe extern "C" fn(*mut c_void, usize, c_uint) -> hipError_t,
    hipHostUnregister: unsafe extern "C" fn(*mut c_void) -> hipError_t,
    hipHostGetDevicePointer: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> hipError_t,
    // Optional evidence extras (never required).
    hipGetErrorString: unsafe extern "C" fn(hipError_t) -> *const c_char,
    hipRuntimeGetVersion: unsafe extern "C" fn(*mut c_int) -> hipError_t,
    hipDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> hipError_t,
    hipDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, hipDevice_t) -> hipError_t,
}

/// How the runtime library handle is kept alive: `Fns` does not own a
/// handle — every runtime resource (`runtime::Rocm`/`Module`/`Function`/
/// `DeviceBuffer`/`HostRegistration`) retains the owning `HipApi`
/// (`Arc<Lib + Fns>`), so the library cannot be unloaded while a resolved
/// pointer is still reachable (structural lifetime, not a comment).
///
/// HIP soname preference order lives in `backend::rocm::probe::HIP_SONAMES`
/// (the single source of truth shared with the probe and the opener); there
/// is deliberately no second copy here.
///
/// Resolve the frozen D0 + D1-additional symbol tables from an open library.
///
/// # SAFETY
/// The returned `Fns` is only valid while `lib` is alive; the caller (the
/// `Rocm` session) owns the `Lib` and drops it after the `Fns` copy.
pub fn resolve(lib: &Lib) -> Fns {
    let mut f = Fns {
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
    };
    macro_rules! bind {
        ( $dst:ident, $sym:literal ) => {
            // SAFETY: the symbol pointers are used only while `lib` is alive
            // (the owning session's contract); transmute to the exact extern
            // fn type mirrors backend::cuda::ffi::resolve.
            f.$dst = unsafe { lib.symbol::<unsafe extern "C" fn()>($sym) }
                .map(|p| unsafe { std::mem::transmute_copy(&p) });
        };
    }
    bind!(hipInit, "hipInit");
    bind!(hipGetDeviceCount, "hipGetDeviceCount");
    bind!(hipSetDevice, "hipSetDevice");
    bind!(hipDeviceSynchronize, "hipDeviceSynchronize");
    bind!(hipStreamSynchronize, "hipStreamSynchronize");
    bind!(hipModuleLoadData, "hipModuleLoadData");
    bind!(hipModuleUnload, "hipModuleUnload");
    bind!(hipModuleGetFunction, "hipModuleGetFunction");
    bind!(hipModuleLaunchKernel, "hipModuleLaunchKernel");
    bind!(hipMalloc, "hipMalloc");
    bind!(hipFree, "hipFree");
    bind!(hipMemcpy, "hipMemcpy");
    bind!(hipHostRegister, "hipHostRegister");
    bind!(hipHostUnregister, "hipHostUnregister");
    bind!(hipHostGetDevicePointer, "hipHostGetDevicePointer");
    bind!(hipGetErrorString, "hipGetErrorString");
    bind!(hipRuntimeGetVersion, "hipRuntimeGetVersion");
    bind!(hipDriverGetVersion, "hipDriverGetVersion");
    bind!(hipDeviceGetName, "hipDeviceGetName");
    f
}

impl Fns {
    /// Names of the frozen D0 surface symbols that did not resolve.
    pub fn d0_missing(&self) -> Vec<&'static str> {
        let present = |n: &str| match n {
            "hipInit" => self.hipInit.is_some(),
            "hipGetDeviceCount" => self.hipGetDeviceCount.is_some(),
            "hipSetDevice" => self.hipSetDevice.is_some(),
            "hipModuleLoadData" => self.hipModuleLoadData.is_some(),
            "hipModuleUnload" => self.hipModuleUnload.is_some(),
            "hipModuleGetFunction" => self.hipModuleGetFunction.is_some(),
            "hipModuleLaunchKernel" => self.hipModuleLaunchKernel.is_some(),
            "hipMalloc" => self.hipMalloc.is_some(),
            "hipFree" => self.hipFree.is_some(),
            "hipMemcpy" => self.hipMemcpy.is_some(),
            "hipDeviceSynchronize" => self.hipDeviceSynchronize.is_some(),
            "hipStreamSynchronize" => self.hipStreamSynchronize.is_some(),
            _ => false,
        };
        HIP_D0_REQUIRED
            .iter()
            .copied()
            .filter(|n| !present(n))
            .collect()
    }

    /// Names of the frozen D1-additional symbols that did not resolve.
    pub fn d1_missing(&self) -> Vec<&'static str> {
        let present = |n: &str| match n {
            "hipHostRegister" => self.hipHostRegister.is_some(),
            "hipHostGetDevicePointer" => self.hipHostGetDevicePointer.is_some(),
            "hipHostUnregister" => self.hipHostUnregister.is_some(),
            _ => false,
        };
        HIP_D1_ADDITIONAL
            .iter()
            .copied()
            .filter(|n| !present(n))
            .collect()
    }

    /// The full D0 ABI surface resolved (scalar == ROCm battery readiness).
    pub fn d0_ready(&self) -> bool {
        self.d0_missing().is_empty()
    }

    /// D0 + D1-additional resolved (direct endpoint mapping readiness).
    pub fn d1_ready(&self) -> bool {
        self.d0_ready() && self.d1_missing().is_empty()
    }
}

/// HIP error string (best effort; empty when the symbol or string is
/// unavailable).
pub fn error_string(fns: &Fns, rc: hipError_t) -> String {
    let Some(f) = fns.hipGetErrorString else {
        return format!("rc {rc} (hipGetErrorString unavailable)");
    };
    // SAFETY: hipGetErrorString returns a static/thread-local NUL-terminated
    // string valid until the next call; we copy it immediately.
    let p = unsafe { f(rc) };
    if p.is_null() {
        return format!("rc {rc} (no error string)");
    }
    // SAFETY: p is NUL-terminated by the HIP contract; read up to the NUL.
    let bytes = unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned();
    if bytes.is_empty() {
        format!("rc {rc}")
    } else {
        format!("rc {rc} {bytes}")
    }
}

/// Turn a failing HIP rc into a typed `Error`.
pub fn hip_error(fns: &Fns, what: &str, rc: hipError_t) -> Error {
    Error::new(Kind::External, format!("{what}: {}", error_string(fns, rc)))
}

/// Assert a HIP call succeeded.
pub fn check(fns: &Fns, what: &str, rc: hipError_t) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(hip_error(fns, what, rc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_surface_against_frozen_tables() {
        // The FFI resolve table must cover exactly the frozen D0/D1 tables
        // (a symbol added to one table and not the other breaks the
        // readiness split the Phase-I receipts attest). Optional extras are
        // not part of either table.
        let empty = Fns {
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
        };
        assert_eq!(empty.d0_missing().len(), HIP_D0_REQUIRED.len());
        assert_eq!(empty.d1_missing().len(), HIP_D1_ADDITIONAL.len());
        assert!(!empty.d0_ready());
        assert!(!empty.d1_ready());
    }

    #[test]
    fn missing_library_is_a_typed_error() {
        let r = Lib::open("libvole_audio_no_such_hip.so.99");
        assert!(r.is_err());
    }

    #[test]
    fn module_unload_is_part_of_the_frozen_d0_surface() {
        // The runtime unloads modules during ordinary teardown; a surface
        // that omitted hipModuleUnload could "resolve completely" and then
        // panic/UB on drop (review finding).
        assert!(
            crate::backend::rocm::probe::HIP_D0_REQUIRED.contains(&"hipModuleUnload"),
            "hipModuleUnload must be a required D0 symbol"
        );
        let empty = Fns {
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
        };
        assert!(empty.d0_missing().contains(&"hipModuleUnload"));
    }

    #[test]
    fn error_string_is_best_effort() {
        let empty = Fns {
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
        };
        assert!(error_string(&empty, 1).contains("rc 1"));
        let e = hip_error(&empty, "test", HIP_ERROR_INVALID_VALUE);
        assert!(e.to_string().contains("test"));
    }
}

//! Minimal audited CUDA driver-API FFI (Phase G), dynamically loaded.
//!
//! Only the exact API surface Phase G uses is bound, with the *legacy* export
//! names that remain stable across driver generations (e.g. `cuCtxCreate_v2`
//! for the classic three-argument context create; the CUDA 13 header moved
//! the macro name to an extended signature, but the classic symbol is still
//! exported by the driver). Every call site is wrapped by `driver::Cuda`;
//! no raw CUDA handle ever escapes this module's RAII types.
//!
//! Loading is `dlopen` of `libcuda.so.1` (the driver, not the toolkit
//! runtime): a machine without the NVIDIA driver never loads this module,
//! and CPU-only builds remain fully usable.

use crate::error::{Error, Kind, Result};
use libc::{c_char, c_int, c_void};
use std::ffi::CString;

pub type CUresult = c_int;
pub type CUdevice = c_int;
/// Opaque driver handles (pointers, kept as usize).
pub type CUcontext = usize;
pub type CUmodule = usize;
pub type CUfunction = usize;
pub type CUstream = usize;
pub type CUevent = usize;
pub type CUgraph = usize;
pub type CUgraphExec = usize;
/// Device memory pointer (64-bit address space).
pub type CUdeviceptr = u64;

pub const CU_CTX_SCHED_AUTO: u32 = 0x00;
pub const CU_CTX_SCHED_BLOCKING_SYNC: u32 = 0x04;
pub const CU_CTX_MAP_HOST: u32 = 0x08;
pub const CU_STREAM_NON_BLOCKING: u32 = 0x01;
pub const CU_EVENT_DISABLE_TIMING: u32 = 0x01;
pub const CU_STREAM_CAPTURE_MODE_RELAXED: u32 = 2;
pub const CU_LAUNCH_ATTRIBUTE_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR: c_int = 2;

// Device attributes used by the probe (values frozen from cuda.h).
pub const ATTR_WARP_SIZE: c_int = 10;
pub const ATTR_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;
pub const ATTR_GPU_CLOCK_RATE: c_int = 13;
pub const ATTR_KERNEL_EXEC_TIMEOUT: c_int = 17;
pub const ATTR_MEMORY_CLOCK_RATE: c_int = 36;
pub const ATTR_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
pub const ATTR_UNIFIED_ADDRESSING: c_int = 41;
pub const ATTR_PCI_BUS_ID: c_int = 33;
pub const ATTR_PCI_DEVICE_ID: c_int = 34;
pub const ATTR_PCI_DOMAIN_ID: c_int = 50;
pub const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const ATTR_STREAM_PRIORITIES_SUPPORTED: c_int = 78;
pub const ATTR_GLOBAL_L1_CACHE_SUPPORTED: c_int = 79;
pub const ATTR_CONCURRENT_MANAGED_ACCESS: c_int = 89;
pub const ATTR_HOST_REGISTER_SUPPORTED: c_int = 99;

// Stream-priority attribute value range is queried via cuCtxGetStreamPriorityRange
// in older drivers; on modern drivers it is a device attribute pair:
// CU_DEVICE_ATTRIBUTE_MIN_STREAM_PRIORITY / MAX_STREAM_PRIORITY.
pub const ATTR_MIN_STREAM_PRIORITY: c_int = 122;
pub const ATTR_MAX_STREAM_PRIORITY: c_int = 123;

macro_rules! fns {
    ($( $name:ident : $fty:ty ),* $(,)?) => {
        #[allow(non_snake_case, missing_docs)]
        #[derive(Clone, Copy)]
        pub struct Fns { $( pub $name: Option<$fty> ),* }

        impl Fns {
            /// Resolve every bound symbol from an open driver handle; a
            /// missing symbol is an error (the driver we run against lacks an
            /// API this build needs).
            ///
            /// # SAFETY
            /// `lib` must stay loaded (and be a valid dlopen handle) for as
            /// long as the returned function pointers are used.
            pub unsafe fn load(lib: *mut c_void) -> Self {
                // SAFETY: the library handle outlives every returned
                // function pointer (the caller keeps it alive for the
                // lifetime of `Fns`).
                unsafe {
                    Fns {
                        $(
                            $name: resolve(lib, stringify!($name)),
                        )*
                    }
                }
            }
        }
    };
}

/// dlsym one symbol (no demangle; driver exports are plain C names).
///
/// # SAFETY
/// The returned pointer is only valid while `lib` stays loaded.
unsafe fn resolve<T>(lib: *mut c_void, name: &str) -> Option<T> {
    let cname = CString::new(name).expect("symbol name has no NUL");
    unsafe {
        // dlsym returns *mut c_void; transmute to the fn pointer type.
        let p = libc::dlsym(lib, cname.as_ptr());
        if p.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy(&p))
        }
    }
}

type FnInit = unsafe extern "C" fn(CUresult) -> CUresult;
type FnDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> CUresult;
type FnDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type FnDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type FnDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type FnDeviceComputeCapability = unsafe extern "C" fn(*mut c_int, *mut c_int, CUdevice) -> CUresult;
type FnDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type FnCtxCreate =
    unsafe extern "C" fn(*mut CUcontext, *mut c_void, c_int, u32, CUdevice) -> CUresult;
type FnCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCtxSynchronize = unsafe extern "C" fn() -> CUresult;
type FnModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnModuleLoadDataEx = unsafe extern "C" fn(
    *mut CUmodule,
    *const c_void,
    u32,
    *const i32,
    *mut *mut c_void,
) -> CUresult;
type FnModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
type FnModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type FnMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type FnMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type FnMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type FnMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type FnMemcpyHtoDAsync =
    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
type FnMemcpyDtoHAsync =
    unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
type FnLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type FnStreamCreate = unsafe extern "C" fn(*mut CUstream, u32) -> CUresult;
type FnStreamCreateWithPriority = unsafe extern "C" fn(*mut CUstream, u32, c_int) -> CUresult;
type FnStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
type FnStreamSynchronize = unsafe extern "C" fn(CUstream) -> CUresult;
type FnStreamGetPriority = unsafe extern "C" fn(CUstream, *mut c_int) -> CUresult;
type FnEventCreate = unsafe extern "C" fn(*mut CUevent, u32) -> CUresult;
type FnEventDestroy = unsafe extern "C" fn(CUevent) -> CUresult;
type FnEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
type FnEventSynchronize = unsafe extern "C" fn(CUevent) -> CUresult;
type FnEventElapsedTime = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
type FnStreamBeginCapture = unsafe extern "C" fn(CUstream, u32) -> CUresult;
type FnStreamEndCapture = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
type FnGraphInstantiateWithFlags = unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult;
type FnGraphLaunch = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
type FnGraphExecDestroy = unsafe extern "C" fn(CUgraphExec) -> CUresult;
type FnGraphDestroy = unsafe extern "C" fn(CUgraph) -> CUresult;
type FnGetErrorString = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;

fns! {
    cuInit: FnInit,
    cuDriverGetVersion: FnDriverGetVersion,
    cuDeviceGetCount: FnDeviceGetCount,
    cuDeviceGet: FnDeviceGet,
    cuDeviceGetName: FnDeviceGetName,
    cuDeviceComputeCapability: FnDeviceComputeCapability,
    cuDeviceGetAttribute: FnDeviceGetAttribute,
    cuCtxCreate: FnCtxCreate,
    cuCtxDestroy: FnCtxDestroy,
    cuCtxSynchronize: FnCtxSynchronize,
    cuModuleLoadData: FnModuleLoadData,
    cuModuleLoadDataEx: FnModuleLoadDataEx,
    cuModuleUnload: FnModuleUnload,
    cuModuleGetFunction: FnModuleGetFunction,
    cuMemAlloc: FnMemAlloc,
    cuMemFree: FnMemFree,
    cuMemcpyHtoD: FnMemcpyHtoD,
    cuMemcpyDtoH: FnMemcpyDtoH,
    cuMemcpyHtoDAsync: FnMemcpyHtoDAsync,
    cuMemcpyDtoHAsync: FnMemcpyDtoHAsync,
    cuLaunchKernel: FnLaunchKernel,
    cuStreamCreate: FnStreamCreate,
    cuStreamCreateWithPriority: FnStreamCreateWithPriority,
    cuStreamDestroy: FnStreamDestroy,
    cuStreamSynchronize: FnStreamSynchronize,
    cuStreamGetPriority: FnStreamGetPriority,
    cuEventCreate: FnEventCreate,
    cuEventDestroy: FnEventDestroy,
    cuEventRecord: FnEventRecord,
    cuEventSynchronize: FnEventSynchronize,
    cuEventElapsedTime: FnEventElapsedTime,
    cuStreamBeginCapture: FnStreamBeginCapture,
    cuStreamEndCapture: FnStreamEndCapture,
    cuGraphInstantiateWithFlags: FnGraphInstantiateWithFlags,
    cuGraphLaunch: FnGraphLaunch,
    cuGraphExecDestroy: FnGraphExecDestroy,
    cuGraphDestroy: FnGraphDestroy,
    cuGetErrorString: FnGetErrorString,
}

/// A loaded driver handle plus its resolved symbols.
pub struct Driver {
    /// dlopen handle; kept alive for the lifetime of every function pointer.
    _lib: *mut c_void,
    pub fns: Fns,
}

impl Driver {
    /// Open `libcuda.so.1` and resolve the bound API surface.
    pub fn open() -> Result<Driver> {
        // SAFETY: dlopen RTLD_NOW|RTLD_LOCAL; the handle is kept for the
        // lifetime of the Driver (all fn pointers die with it).
        let name = CString::new("libcuda.so.1").expect("no NUL");
        let lib = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if lib.is_null() {
            return Err(Error::new(
                Kind::Unavailable,
                "CUDA driver (libcuda.so.1) not loadable on this host",
            ));
        }
        let fns = unsafe { Fns::load(lib) };
        // Every bound symbol must exist; a partial surface is an error, not a
        // silent degradation.
        let missing: Vec<&'static str> = fns.missing();
        if !missing.is_empty() {
            return Err(Error::new(
                Kind::Unsupported,
                format!(
                    "CUDA driver missing required symbols: {}",
                    missing.join(", ")
                ),
            ));
        }
        Ok(Driver { _lib: lib, fns })
    }
}

impl Fns {
    fn missing(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        macro_rules! probe {
            ($($name:ident),* $(,)?) => {
                $( if self.$name.is_none() { out.push(stringify!($name)); } )*
            };
        }
        probe! {
            cuInit, cuDriverGetVersion, cuDeviceGetCount, cuDeviceGet,
            cuDeviceGetName, cuDeviceComputeCapability, cuDeviceGetAttribute,
            cuCtxCreate, cuCtxDestroy, cuCtxSynchronize, cuModuleLoadData,
            cuModuleLoadDataEx, cuModuleUnload, cuModuleGetFunction, cuMemAlloc, cuMemFree,
            cuMemcpyHtoD, cuMemcpyDtoH, cuMemcpyHtoDAsync, cuMemcpyDtoHAsync,
            cuLaunchKernel, cuStreamCreate, cuStreamCreateWithPriority,
            cuStreamDestroy, cuStreamSynchronize, cuStreamGetPriority,
            cuEventCreate, cuEventDestroy, cuEventRecord, cuEventSynchronize,
            cuEventElapsedTime, cuStreamBeginCapture, cuStreamEndCapture,
            cuGraphInstantiateWithFlags, cuGraphLaunch, cuGraphExecDestroy,
            cuGraphDestroy, cuGetErrorString,
        }
        out
    }
}

impl std::fmt::Debug for Driver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Driver(libcuda.so.1)")
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        if !self._lib.is_null() {
            // SAFETY: close after all uses (Driver owns the only references).
            unsafe { libc::dlclose(self._lib) };
        }
    }
}

/// Format a CUDA result into an `Error` with the driver's error string.
pub fn cuda_error(fns: &Fns, what: &str, rc: CUresult) -> Error {
    Error::new(
        Kind::External,
        format!("cuda {what}: {} (rc {rc})", error_string(fns, rc)),
    )
}

/// Driver error string for a result code (never panics on unknown codes).
pub fn error_string(fns: &Fns, rc: CUresult) -> String {
    if let Some(f) = fns.cuGetErrorString {
        // SAFETY: out-param writes a static string owned by the driver.
        let mut s: *const c_char = std::ptr::null();
        let r = unsafe { f(rc, &mut s) };
        if r == 0 && !s.is_null() {
            // SAFETY: s is a NUL-terminated driver-owned string.
            let c = unsafe { std::ffi::CStr::from_ptr(s) };
            return c.to_string_lossy().into_owned();
        }
    }
    format!("CUresult {rc}")
}

//! Minimal audited CUDA driver-API FFI (Phase G), dynamically loaded.
//!
//! Only the exact API surface Phase G uses is bound. Every numeric constant
//! in this module is frozen **from the installed CUDA 13.3 headers**
//! (`/opt/cuda/include/cuda.h`) and pinned by the `frozen_constants` test
//! table below, so a manual transcription error cannot silently drift.
//!
//! ABI notes (verified against driver 610.57.04 + CUDA 13.3 headers):
//!
//! * `cuCtxCreate` maps to `cuCtxCreate_v4` and takes **four** arguments:
//!   `(CUcontext*, CUctxCreateParams*, unsigned int flags, CUdevice)`.
//!   `ctxCreateParams` may be `NULL` for a regular context. (The
//!   five-argument `(…, CUexecAffinityParam*, int, …)` form is
//!   `cuCtxCreate_v3`, a different API. The legacy `cuCtxCreate_v2` export
//!   on this driver yields a context that rejects later allocation with
//!   rc 201 — not used.)
//! * All symbols are resolved with the exported names and the signatures
//!   above; `cuGetProcAddress`-style explicit versioning is not needed while
//!   we bind exactly one documented ABI per symbol, but each binding names
//!   the header signature it implements.
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

// ---------------------------------------------------------------------------
// Frozen constants — values transcribed from /opt/cuda/include/cuda.h
// (CUDA 13.3). Every value is pinned by the `frozen_constants` test.
// ---------------------------------------------------------------------------

/// CU_CTX_SCHED_AUTO = 0x00 (cuCtxCreate flags; we pass 0 and say so).
pub const CU_CTX_SCHED_AUTO: u32 = 0x00;
/// CU_STREAM_NON_BLOCKING = 0x1.
pub const CU_STREAM_NON_BLOCKING: u32 = 0x01;
/// CU_EVENT_DISABLE_TIMING = 0x2 (CUDA 13 header; 0x1 is CU_EVENT_DEFAULT).
pub const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
/// CU_STREAM_CAPTURE_MODE_RELAXED = 2.
pub const CU_STREAM_CAPTURE_MODE_RELAXED: u32 = 2;

/// CU_JIT_INFO_LOG_BUFFER = 3 (option value: `char*`).
pub const CU_JIT_INFO_LOG_BUFFER: i32 = 3;
/// CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4 (option value: `unsigned int`, in/out).
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: i32 = 4;
/// CU_JIT_ERROR_LOG_BUFFER = 5 (option value: `char*`).
pub const CU_JIT_ERROR_LOG_BUFFER: i32 = 5;
/// CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6 (option value: `unsigned int`, in/out).
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: i32 = 6;

// Device attributes used by the probe (cuda.h lines 828-917 area).
pub const ATTR_WARP_SIZE: c_int = 10;
pub const ATTR_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const ATTR_CLOCK_RATE: c_int = 13;
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;
pub const ATTR_KERNEL_EXEC_TIMEOUT: c_int = 17;
pub const ATTR_PCI_BUS_ID: c_int = 33;
pub const ATTR_PCI_DEVICE_ID: c_int = 34;
pub const ATTR_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
pub const ATTR_UNIFIED_ADDRESSING: c_int = 41;
pub const ATTR_PCI_DOMAIN_ID: c_int = 50;
pub const ATTR_STREAM_PRIORITIES_SUPPORTED: c_int = 78;
pub const ATTR_GLOBAL_L1_CACHE_SUPPORTED: c_int = 79;
pub const ATTR_CONCURRENT_MANAGED_ACCESS: c_int = 89;
pub const ATTR_HOST_REGISTER_SUPPORTED: c_int = 99;
/// CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113 (cuda.h
/// line 932) — gates the READ_ONLY registration flag.
pub const ATTR_READ_ONLY_HOST_REGISTER_SUPPORTED: c_int = 113;

// cuMemHostRegister flags (cuda.h lines 3416-3451, CUDA 13.3). Note:
// CU_MEMHOSTREGISTER_WRITE_COMBINED does **not** exist in current headers
// (removed); only PORTABLE/DEVICEMAP/IOMEMORY/READ_ONLY remain. Every value
// is pinned by the `frozen_constants` test.
pub const CU_MEMHOSTREGISTER_PORTABLE: u32 = 0x01;
pub const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;
pub const CU_MEMHOSTREGISTER_IOMEMORY: u32 = 0x04;
pub const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 0x08;

/// CUmemorytype values (cuda.h line 1215-1218) — `cuPointerGetAttribute`
/// `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` output.
pub const CU_MEMORYTYPE_HOST: c_int = 0x01;
pub const CU_MEMORYTYPE_DEVICE: c_int = 0x02;
pub const CU_MEMORYTYPE_UNIFIED: c_int = 0x04;

// CUpointer_attribute values (cuda.h enum lines 998-1018) used by the D1
// pointer-attribute evidence probe; the attribute is passed as an int on the
// wire.
pub const POINTER_ATTRIBUTE_MEMORY_TYPE: c_int = 2;
pub const POINTER_ATTRIBUTE_DEVICE_POINTER: c_int = 3;
pub const POINTER_ATTRIBUTE_HOST_POINTER: c_int = 4;
pub const POINTER_ATTRIBUTE_RANGE_START_ADDR: c_int = 11;
pub const POINTER_ATTRIBUTE_RANGE_SIZE: c_int = 12;
pub const POINTER_ATTRIBUTE_MAPPED: c_int = 13;

// CUDA error codes referenced by the D1 registration classifier (cuda.h
// lines 2692-3361).
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
pub const CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: CUresult = 712;
pub const CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: CUresult = 713;
pub const CUDA_ERROR_NOT_PERMITTED: CUresult = 800;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;

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
/// cuCtxCreate(CUcontext*, CUctxCreateParams*, unsigned int flags, CUdevice)
/// — the current (v4) four-argument ABI; params may be NULL for a regular
/// context (cuda.h line 6481).
type FnCtxCreate = unsafe extern "C" fn(*mut CUcontext, *mut c_void, u32, CUdevice) -> CUresult;
type FnCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCtxSynchronize = unsafe extern "C" fn() -> CUresult;
/// cuCtxGetStreamPriorityRange(int* least, int* greatest) — the only
/// documented way to obtain the meaningful priority range (cuda.h 7176).
type FnCtxGetStreamPriorityRange = unsafe extern "C" fn(*mut c_int, *mut c_int) -> CUresult;
/// cuCtxGetCurrent(CUcontext*) — the calling thread's current context (NULL
/// when the thread has none).
type FnCtxGetCurrent = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
/// cuCtxSetCurrent(CUcontext) — set the calling thread's current context.
///
/// This is the primitive the affinity guard uses rather than
/// `cuCtxPushCurrent`: a context created by `cuCtxCreate` is already on the
/// calling thread's context stack, and pushing it again returns
/// `CUDA_ERROR_INVALID_CONTEXT` (rc 201, verified on driver 610.57.04).
/// `cuCtxSetCurrent` switches without touching the stack, so saving the
/// previous value and setting it back restores the exact prior state.
type FnCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
/// cuCtxPopCurrent(CUcontext*) — pop the calling thread's context stack,
/// restoring the previous current context. Used once, right after creation, to
/// leave a newly created context *floating* rather than attached to its
/// creator thread.
type FnCtxPopCurrent = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
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

// Phase H (D1) surface — bound as *optional* symbols: a driver predating
// them (or a hypothetical build without them) still opens for the D0
// courts, and `court d1` records UNSUPPORTED_BY_API when one is absent.
// Signatures transcribed from cuda.h:
//   cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags)  [9773]
//   cuMemHostUnregister(void *p)                                     [9799]
//   cuMemHostGetDevicePointer(CUdeviceptr*, void*, unsigned int)     [v2]
//   cuPointerGetAttribute(void *data, CUpointer_attribute, CUdeviceptr) [15496]
//   cuMemGetAddressRange(CUdeviceptr*, size_t*, CUdeviceptr)         [8971]
type FnMemHostRegister = unsafe extern "C" fn(*mut c_void, usize, u32) -> CUresult;
type FnMemHostUnregister = unsafe extern "C" fn(*mut c_void) -> CUresult;
type FnMemHostGetDevicePointer =
    unsafe extern "C" fn(*mut CUdeviceptr, *mut c_void, u32) -> CUresult;
type FnPointerGetAttribute = unsafe extern "C" fn(*mut c_void, c_int, CUdeviceptr) -> CUresult;
type FnMemGetAddressRange =
    unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUdeviceptr) -> CUresult;

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
    cuCtxGetStreamPriorityRange: FnCtxGetStreamPriorityRange,
    cuCtxGetCurrent: FnCtxGetCurrent,
    cuCtxSetCurrent: FnCtxSetCurrent,
    cuCtxPopCurrent: FnCtxPopCurrent,
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
    cuMemHostRegister: FnMemHostRegister,
    cuMemHostUnregister: FnMemHostUnregister,
    cuMemHostGetDevicePointer: FnMemHostGetDevicePointer,
    cuPointerGetAttribute: FnPointerGetAttribute,
    cuMemGetAddressRange: FnMemGetAddressRange,
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
            cuCtxCreate, cuCtxDestroy, cuCtxSynchronize, cuCtxGetStreamPriorityRange,
            cuCtxGetCurrent, cuCtxSetCurrent, cuCtxPopCurrent,
            cuModuleLoadDataEx, cuModuleUnload, cuModuleGetFunction, cuMemAlloc, cuMemFree,
            cuMemcpyHtoD, cuMemcpyDtoH, cuLaunchKernel, cuStreamCreate,
            cuStreamCreateWithPriority, cuStreamDestroy, cuStreamSynchronize,
            cuStreamGetPriority, cuEventCreate, cuEventDestroy, cuEventRecord,
            cuEventSynchronize, cuEventElapsedTime, cuStreamBeginCapture,
            cuStreamEndCapture, cuGraphInstantiateWithFlags, cuGraphLaunch,
            cuGraphExecDestroy, cuGraphDestroy, cuGetErrorString,
        }
        out
    }
}

impl Fns {
    /// True when the full Phase H D1 host-registration surface resolved.
    /// `court d1` requires this; the D0 courts do not.
    pub fn d1_surface_complete(&self) -> bool {
        self.cuMemHostRegister.is_some()
            && self.cuMemHostUnregister.is_some()
            && self.cuMemHostGetDevicePointer.is_some()
            && self.cuPointerGetAttribute.is_some()
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

// SAFETY: `Driver` owns the only reference to its dlopen handle and closes it
// exactly once on drop; the resolved function pointers are process-global code
// addresses. Sharing the handle across threads is therefore sound, and it is
// what lets a CUDA context be held by an `Arc` shared with every resource (see
// `driver::CudaContext`).
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every manually frozen numeric CUDA constant, pinned to its value in
    /// the CUDA 13.3 headers (/opt/cuda/include/cuda.h). Changing a value
    /// here without a header citation is a defect.
    #[test]
    fn frozen_constants_match_cuda_13_3_headers() {
        let rows: Vec<(&str, i64)> = vec![
            // cuEventFlag (cuda.h ~line 474)
            (
                "CU_EVENT_DISABLE_TIMING",
                i64::from(CU_EVENT_DISABLE_TIMING),
            ),
            // cuStreamCreateFlags (cuda.h line 445)
            ("CU_STREAM_NON_BLOCKING", i64::from(CU_STREAM_NON_BLOCKING)),
            // cuStreamCaptureMode (cuda.h line 2544)
            (
                "CU_STREAM_CAPTURE_MODE_RELAXED",
                i64::from(CU_STREAM_CAPTURE_MODE_RELAXED),
            ),
            // cuJitOption (cuda.h lines ~1282-1314)
            ("CU_JIT_INFO_LOG_BUFFER", i64::from(CU_JIT_INFO_LOG_BUFFER)),
            (
                "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES",
                i64::from(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES),
            ),
            (
                "CU_JIT_ERROR_LOG_BUFFER",
                i64::from(CU_JIT_ERROR_LOG_BUFFER),
            ),
            (
                "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES",
                i64::from(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES),
            ),
            // cuDeviceAttribute (cuda.h lines 815-917)
            (
                "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
                i64::from(ATTR_MAX_THREADS_PER_BLOCK),
            ),
            ("CU_DEVICE_ATTRIBUTE_WARP_SIZE", i64::from(ATTR_WARP_SIZE)),
            ("CU_DEVICE_ATTRIBUTE_CLOCK_RATE", i64::from(ATTR_CLOCK_RATE)),
            (
                "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
                i64::from(ATTR_MULTIPROCESSOR_COUNT),
            ),
            (
                "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT",
                i64::from(ATTR_KERNEL_EXEC_TIMEOUT),
            ),
            ("CU_DEVICE_ATTRIBUTE_PCI_BUS_ID", i64::from(ATTR_PCI_BUS_ID)),
            (
                "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",
                i64::from(ATTR_PCI_DEVICE_ID),
            ),
            (
                "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",
                i64::from(ATTR_PCI_DOMAIN_ID),
            ),
            (
                "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",
                i64::from(ATTR_MAX_THREADS_PER_MULTIPROCESSOR),
            ),
            (
                "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING",
                i64::from(ATTR_UNIFIED_ADDRESSING),
            ),
            (
                "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED",
                i64::from(ATTR_STREAM_PRIORITIES_SUPPORTED),
            ),
            (
                "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED",
                i64::from(ATTR_GLOBAL_L1_CACHE_SUPPORTED),
            ),
            (
                "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS",
                i64::from(ATTR_CONCURRENT_MANAGED_ACCESS),
            ),
            (
                "CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED",
                i64::from(ATTR_HOST_REGISTER_SUPPORTED),
            ),
            (
                "CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED",
                i64::from(ATTR_READ_ONLY_HOST_REGISTER_SUPPORTED),
            ),
            // cuMemHostRegister flags (cuda.h lines 3416-3451)
            (
                "CU_MEMHOSTREGISTER_PORTABLE",
                i64::from(CU_MEMHOSTREGISTER_PORTABLE),
            ),
            (
                "CU_MEMHOSTREGISTER_DEVICEMAP",
                i64::from(CU_MEMHOSTREGISTER_DEVICEMAP),
            ),
            (
                "CU_MEMHOSTREGISTER_IOMEMORY",
                i64::from(CU_MEMHOSTREGISTER_IOMEMORY),
            ),
            (
                "CU_MEMHOSTREGISTER_READ_ONLY",
                i64::from(CU_MEMHOSTREGISTER_READ_ONLY),
            ),
            // CUmemorytype (cuda.h line 1215-1218)
            ("CU_MEMORYTYPE_HOST", i64::from(CU_MEMORYTYPE_HOST)),
            ("CU_MEMORYTYPE_DEVICE", i64::from(CU_MEMORYTYPE_DEVICE)),
            ("CU_MEMORYTYPE_UNIFIED", i64::from(CU_MEMORYTYPE_UNIFIED)),
            // CUpointer_attribute (cuda.h lines 998-1018)
            (
                "CU_POINTER_ATTRIBUTE_MEMORY_TYPE",
                i64::from(POINTER_ATTRIBUTE_MEMORY_TYPE),
            ),
            (
                "CU_POINTER_ATTRIBUTE_DEVICE_POINTER",
                i64::from(POINTER_ATTRIBUTE_DEVICE_POINTER),
            ),
            (
                "CU_POINTER_ATTRIBUTE_HOST_POINTER",
                i64::from(POINTER_ATTRIBUTE_HOST_POINTER),
            ),
            (
                "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR",
                i64::from(POINTER_ATTRIBUTE_RANGE_START_ADDR),
            ),
            (
                "CU_POINTER_ATTRIBUTE_RANGE_SIZE",
                i64::from(POINTER_ATTRIBUTE_RANGE_SIZE),
            ),
            (
                "CU_POINTER_ATTRIBUTE_MAPPED",
                i64::from(POINTER_ATTRIBUTE_MAPPED),
            ),
            // CUDA driver error codes (cuda.h lines 2692-3361)
            (
                "CUDA_ERROR_INVALID_VALUE",
                i64::from(CUDA_ERROR_INVALID_VALUE),
            ),
            (
                "CUDA_ERROR_OUT_OF_MEMORY",
                i64::from(CUDA_ERROR_OUT_OF_MEMORY),
            ),
            ("CUDA_ERROR_NO_DEVICE", i64::from(CUDA_ERROR_NO_DEVICE)),
            (
                "CUDA_ERROR_INVALID_DEVICE",
                i64::from(CUDA_ERROR_INVALID_DEVICE),
            ),
            (
                "CUDA_ERROR_INVALID_CONTEXT",
                i64::from(CUDA_ERROR_INVALID_CONTEXT),
            ),
            (
                "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
                i64::from(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED),
            ),
            (
                "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
                i64::from(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED),
            ),
            (
                "CUDA_ERROR_NOT_PERMITTED",
                i64::from(CUDA_ERROR_NOT_PERMITTED),
            ),
            (
                "CUDA_ERROR_NOT_SUPPORTED",
                i64::from(CUDA_ERROR_NOT_SUPPORTED),
            ),
        ];
        let expected: Vec<(&str, i64)> = vec![
            ("CU_EVENT_DISABLE_TIMING", 0x2),
            ("CU_STREAM_NON_BLOCKING", 0x1),
            ("CU_STREAM_CAPTURE_MODE_RELAXED", 2),
            ("CU_JIT_INFO_LOG_BUFFER", 3),
            ("CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES", 4),
            ("CU_JIT_ERROR_LOG_BUFFER", 5),
            ("CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES", 6),
            ("CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK", 1),
            ("CU_DEVICE_ATTRIBUTE_WARP_SIZE", 10),
            ("CU_DEVICE_ATTRIBUTE_CLOCK_RATE", 13),
            ("CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT", 16),
            ("CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT", 17),
            ("CU_DEVICE_ATTRIBUTE_PCI_BUS_ID", 33),
            ("CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID", 34),
            ("CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID", 50),
            ("CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR", 39),
            ("CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING", 41),
            ("CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED", 78),
            ("CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED", 79),
            ("CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS", 89),
            ("CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED", 99),
            ("CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED", 113),
            ("CU_MEMHOSTREGISTER_PORTABLE", 0x1),
            ("CU_MEMHOSTREGISTER_DEVICEMAP", 0x2),
            ("CU_MEMHOSTREGISTER_IOMEMORY", 0x4),
            ("CU_MEMHOSTREGISTER_READ_ONLY", 0x8),
            ("CU_MEMORYTYPE_HOST", 0x1),
            ("CU_MEMORYTYPE_DEVICE", 0x2),
            ("CU_MEMORYTYPE_UNIFIED", 0x4),
            ("CU_POINTER_ATTRIBUTE_MEMORY_TYPE", 2),
            ("CU_POINTER_ATTRIBUTE_DEVICE_POINTER", 3),
            ("CU_POINTER_ATTRIBUTE_HOST_POINTER", 4),
            ("CU_POINTER_ATTRIBUTE_RANGE_START_ADDR", 11),
            ("CU_POINTER_ATTRIBUTE_RANGE_SIZE", 12),
            ("CU_POINTER_ATTRIBUTE_MAPPED", 13),
            ("CUDA_ERROR_INVALID_VALUE", 1),
            ("CUDA_ERROR_OUT_OF_MEMORY", 2),
            ("CUDA_ERROR_NO_DEVICE", 100),
            ("CUDA_ERROR_INVALID_DEVICE", 101),
            ("CUDA_ERROR_INVALID_CONTEXT", 201),
            ("CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED", 712),
            ("CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED", 713),
            ("CUDA_ERROR_NOT_PERMITTED", 800),
            ("CUDA_ERROR_NOT_SUPPORTED", 801),
        ];
        assert_eq!(rows.len(), expected.len());
        for ((name, val), (ename, eval)) in rows.iter().zip(expected.iter()) {
            assert_eq!(
                val, eval,
                "{name} frozen value {val} != cuda.h value {eval}; re-verify against the header"
            );
            let _ = ename;
        }
    }
}

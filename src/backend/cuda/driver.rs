//! CUDA driver-API wrapper (Phase G): context, module, streams, events,
//! device buffers — all RAII, all through the audited `ffi::Fns`.
//!
//! Threading rule: driver API calls are bound to the *current* context of
//! the calling thread. `CudaContext::create` makes the new context current
//! for this thread, and every subsequent call in this module happens on the
//! same thread (courts run single-threaded), so no context juggling is
//! needed. Drop order matters: streams/events/buffers must die before the
//! module and context (Rust drops fields in declaration order; the context is
//! declared last and is also the last field to drop).

use crate::error::{Error, Result};
use std::ffi::{CString, c_void};
use std::path::Path;

use super::ffi::{
    ATTR_CONCURRENT_MANAGED_ACCESS, ATTR_GLOBAL_L1_CACHE_SUPPORTED, ATTR_GPU_CLOCK_RATE,
    ATTR_HOST_REGISTER_SUPPORTED, ATTR_KERNEL_EXEC_TIMEOUT, ATTR_MAX_STREAM_PRIORITY,
    ATTR_MAX_THREADS_PER_BLOCK, ATTR_MAX_THREADS_PER_MULTIPROCESSOR, ATTR_MIN_STREAM_PRIORITY,
    ATTR_MULTIPROCESSOR_COUNT, ATTR_PCI_BUS_ID, ATTR_PCI_DEVICE_ID, ATTR_PCI_DOMAIN_ID,
    ATTR_STREAM_PRIORITIES_SUPPORTED, ATTR_UNIFIED_ADDRESSING, CUdeviceptr, CUgraph, CUgraphExec,
    CUmodule, CUresult, CUstream, Driver, Fns, cuda_error,
};

// ---------------------------------------------------------------------------
// Result checking helper
// ---------------------------------------------------------------------------

fn check(fns: &Fns, what: &str, rc: CUresult) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(cuda_error(fns, what, rc))
    }
}

// ---------------------------------------------------------------------------
// Device + context
// ---------------------------------------------------------------------------

/// One CUDA device (probed facts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub ordinal: i32,
    pub name: String,
    pub major: i32,
    pub minor: i32,
    /// sm_XY label, e.g. "sm_89".
    pub sm: String,
    pub pci_bus_id: i32,
    pub pci_device_id: i32,
    pub pci_domain_id: i32,
    pub multiprocessor_count: i32,
    pub clock_rate_khz: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_multiprocessor: i32,
    pub unified_addressing: i32,
    pub stream_priorities_supported: i32,
    pub min_stream_priority: i32,
    pub max_stream_priority: i32,
    pub concurrent_managed_access: i32,
    pub host_register_supported: i32,
    pub global_l1_cache_supported: i32,
    pub kernel_exec_timeout: i32,
}

impl DeviceInfo {
    pub fn probe(fns: &Fns, ordinal: i32) -> Result<DeviceInfo> {
        let get = |attr: std::ffi::c_int| -> Result<i32> {
            let mut v = 0;
            // SAFETY: device attribute out-param.
            let rc = unsafe { (fns.cuDeviceGetAttribute.expect("bound"))(&mut v, attr, ordinal) };
            check(fns, "cuDeviceGetAttribute", rc)?;
            Ok(v)
        };
        let name = {
            let mut buf = vec![0u8; 256];
            // SAFETY: name buffer of 256 bytes.
            let rc = unsafe {
                (fns.cuDeviceGetName.expect("bound"))(
                    buf.as_mut_ptr() as *mut std::ffi::c_char,
                    buf.len() as i32,
                    ordinal,
                )
            };
            check(fns, "cuDeviceGetName", rc)?;
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            String::from_utf8_lossy(&buf[..end]).into_owned()
        };
        let (mut major, mut minor) = (0, 0);
        // SAFETY: compute-capability out params.
        let rc = unsafe {
            (fns.cuDeviceComputeCapability.expect("bound"))(&mut major, &mut minor, ordinal)
        };
        check(fns, "cuDeviceComputeCapability", rc)?;
        let info = DeviceInfo {
            ordinal,
            sm: format!("sm_{major}{minor}"),
            name,
            major,
            minor,
            pci_bus_id: get(ATTR_PCI_BUS_ID).unwrap_or(-1),
            pci_device_id: get(ATTR_PCI_DEVICE_ID).unwrap_or(-1),
            pci_domain_id: get(ATTR_PCI_DOMAIN_ID).unwrap_or(-1),
            multiprocessor_count: get(ATTR_MULTIPROCESSOR_COUNT)?,
            clock_rate_khz: get(ATTR_GPU_CLOCK_RATE)?,
            max_threads_per_block: get(ATTR_MAX_THREADS_PER_BLOCK)?,
            max_threads_per_multiprocessor: get(ATTR_MAX_THREADS_PER_MULTIPROCESSOR)?,
            unified_addressing: get(ATTR_UNIFIED_ADDRESSING)?,
            stream_priorities_supported: get(ATTR_STREAM_PRIORITIES_SUPPORTED)?,
            min_stream_priority: get(ATTR_MIN_STREAM_PRIORITY)?,
            max_stream_priority: get(ATTR_MAX_STREAM_PRIORITY)?,
            concurrent_managed_access: get(ATTR_CONCURRENT_MANAGED_ACCESS)?,
            host_register_supported: get(ATTR_HOST_REGISTER_SUPPORTED)?,
            global_l1_cache_supported: get(ATTR_GLOBAL_L1_CACHE_SUPPORTED)?,
            kernel_exec_timeout: get(ATTR_KERNEL_EXEC_TIMEOUT)?,
        };
        let _ = (major, minor);
        Ok(info)
    }
}

/// A CUDA driver handle + context + device (owns the whole session).
pub struct Cuda {
    pub fns: Fns,
    /// Driver version (e.g. 13030 for 13.3).
    pub driver_version: i32,
    pub device: DeviceInfo,
    /// Current context for this thread (created by `open`).
    pub context: usize,
    /// Keep the dlopen handle alive last.
    _driver: Driver,
}

impl std::fmt::Debug for Cuda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cuda")
            .field("driver_version", &self.driver_version)
            .field("device", &self.device.name)
            .field("sm", &self.device.sm)
            .finish()
    }
}

impl Cuda {
    /// Open the driver, pick `ordinal` (default 0), create a context.
    pub fn open(ordinal: i32) -> Result<Cuda> {
        // Eager module loading + no on-disk JIT cache: with the default LAZY
        // mode the driver defers PTX JIT past module load (deferred failures
        // surface confusingly), and the on-disk JIT cache can serve a stale
        // entry as a silent rc 218 (observed on driver 610.57.04). We JIT
        // deterministically at load — the "warm JIT at setup, never on a
        // real-time path" rule — with no hidden on-disk state. Process-wide
        // settings, applied only when unset; documented in PROJECT_STATE.
        if std::env::var_os("CUDA_MODULE_LOADING").is_none() {
            // SAFETY: single-threaded setup before any other driver use.
            unsafe { std::env::set_var("CUDA_MODULE_LOADING", "EAGER") };
        }
        if std::env::var_os("CUDA_CACHE_DISABLE").is_none() {
            // SAFETY: single-threaded setup before any other driver use.
            unsafe { std::env::set_var("CUDA_CACHE_DISABLE", "1") };
        }
        let driver = Driver::open()?;
        let fns = driver.fns;
        // SAFETY: driver API init.
        let rc = unsafe { (fns.cuInit.expect("bound"))(0) };
        check(&fns, "cuInit", rc)?;
        let mut version = 0;
        // SAFETY: out-param.
        let rc = unsafe { (fns.cuDriverGetVersion.expect("bound"))(&mut version) };
        check(&fns, "cuDriverGetVersion", rc)?;
        let mut count = 0;
        // SAFETY: out-param.
        let rc = unsafe { (fns.cuDeviceGetCount.expect("bound"))(&mut count) };
        check(&fns, "cuDeviceGetCount", rc)?;
        if count == 0 {
            return Err(Error::new(
                crate::error::Kind::Unavailable,
                "CUDA driver present but no CUDA device",
            ));
        }
        if ordinal >= count {
            return Err(Error::new(
                crate::error::Kind::Unavailable,
                format!(
                    "CUDA device ordinal {ordinal} out of range ({} devices)",
                    count
                ),
            ));
        }
        let device = DeviceInfo::probe(&fns, ordinal)?;
        // Context creation: bind the *exported* `cuCtxCreate` with its modern
        // driver signature `(CUcontext*, CUexecAffinityParam*, int numParams,
        // unsigned int flags, CUdevice)` — zero affinity params + default
        // scheduling. (The legacy `cuCtxCreate_v2` export on current drivers
        // yields a context that rejects later allocation; verified against
        // driver 610.57.04.) The created context becomes current for this
        // thread.
        let mut context: usize = 0;
        // SAFETY: out-param; no affinity params; default flags.
        let rc = unsafe {
            (fns.cuCtxCreate.expect("bound"))(&mut context, std::ptr::null_mut(), 0, 0, ordinal)
        };
        check(&fns, "cuCtxCreate", rc)?;
        Ok(Cuda {
            fns,
            driver_version: version,
            device,
            context,
            _driver: driver,
        })
    }

    /// Highest available stream priority (0 is the default; negative values
    /// are higher priority on NVIDIA).
    pub fn highest_priority(&self) -> i32 {
        if self.device.stream_priorities_supported != 0 {
            self.device.min_stream_priority
        } else {
            0
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: context synchronize.
        let rc = unsafe { (self.fns.cuCtxSynchronize.expect("bound"))() };
        check(&self.fns, "cuCtxSynchronize", rc)
    }

    /// Load a module from PTX/cubin bytes.
    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        Module::load(&self.fns, image)
    }

    /// Load a module from a file (PTX text or cubin).
    pub fn load_module_file(&self, path: &Path) -> Result<Module> {
        let bytes = std::fs::read(path)?;
        self.load_module(&bytes)
    }

    pub fn create_stream(&self) -> Result<Stream> {
        Stream::create(&self.fns, 0)
    }

    pub fn create_stream_priority(&self, priority: i32) -> Result<Stream> {
        Stream::create_priority(&self.fns, priority)
    }

    pub fn create_event(&self, disable_timing: bool) -> Result<Event> {
        Event::create(&self.fns, disable_timing)
    }

    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer> {
        DeviceBuffer::alloc(&self.fns, bytes)
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        // SAFETY: context destroy (best effort at teardown).
        if self.context != 0 {
            unsafe { (self.fns.cuCtxDestroy.expect("bound"))(self.context) };
        }
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Loaded CUDA module (PTX or cubin image).
pub struct Module {
    pub fns: Fns,
    pub handle: CUmodule,
}

impl Module {
    fn load(fns: &Fns, image: &[u8]) -> Result<Module> {
        // cuModuleLoadDataEx (not plain cuModuleLoadData): on the tested
        // driver (610.57.04) the plain entry fails PTX JIT with rc 218 while
        // the Ex entry with an error-log buffer succeeds, and the log makes
        // any genuine JIT failure diagnosable.
        let mut handle: CUmodule = 0;
        const CU_JIT_ERROR_LOG_BUFFER: i32 = 2;
        const LOG_BYTES: usize = 1 << 15;
        let mut log = vec![0u8; LOG_BYTES];
        let mut opts: [i32; 1] = [CU_JIT_ERROR_LOG_BUFFER];
        let mut vals: [*mut c_void; 1] = [log.as_mut_ptr() as *mut c_void];
        // SAFETY: image bytes live for the call (driver copies/JITs them);
        // the log buffer is written by the JIT and read below.
        let rc = unsafe {
            (fns.cuModuleLoadDataEx.expect("bound"))(
                &mut handle,
                image.as_ptr() as *const _,
                1,
                opts.as_mut_ptr(),
                vals.as_mut_ptr(),
            )
        };
        if rc != 0 {
            let end = log.iter().position(|&b| b == 0).unwrap_or(log.len());
            let jit_log = String::from_utf8_lossy(&log[..end]);
            let msg = if jit_log.is_empty() {
                format!("PTX JIT failed (rc {rc})")
            } else {
                format!("PTX JIT failed (rc {rc}): {}", jit_log.trim())
            };
            return Err(crate::error::Error::new(crate::error::Kind::External, msg));
        }
        Ok(Module { fns: *fns, handle })
    }

    /// Look up a kernel entry by (mangled-free) export name.
    pub fn function(&self, name: &str) -> Result<Function> {
        let cname = CString::new(name).map_err(|_| Error::malformed("kernel name has NUL"))?;
        let mut f: usize = 0;
        // SAFETY: out-param; cname lives for the call.
        let rc = unsafe {
            (self.fns.cuModuleGetFunction.expect("bound"))(&mut f, self.handle, cname.as_ptr())
        };
        check(&self.fns, "cuModuleGetFunction", rc)?;
        Ok(Function {
            fns: self.fns,
            handle: f,
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: module unload (best effort).
            unsafe { (self.fns.cuModuleUnload.expect("bound"))(self.handle) };
        }
    }
}

/// A kernel function handle.
#[derive(Clone, Copy)]
pub struct Function {
    pub fns: Fns,
    pub handle: usize,
}

impl Function {
    /// Launch with parameters given as 8-byte little-endian values in order
    /// (device pointers and 64-bit scalars; the kernel ABI takes one 64-bit
    /// parameter slot each).
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        stream: CUstream,
        params: &[u64],
    ) -> Result<()> {
        // The driver expects an array of pointers, each pointing at one
        // parameter's bytes. Build a local value buffer and point into it
        // (the call is synchronous in the sense that the driver copies the
        // parameter bytes before returning).
        let mut buf: Vec<u64> = params.to_vec();
        let mut kernel_params: Vec<*mut std::ffi::c_void> = buf
            .iter_mut()
            .map(|v| v as *mut u64 as *mut std::ffi::c_void)
            .collect();
        // SAFETY: kernel_params entries point at each parameter's bytes in
        // `buf` (kept alive for the call); `extra` is NULL; the kernel is
        // `vole_render_d0` (no shared memory requested).
        let rc = unsafe {
            (self.fns.cuLaunchKernel.expect("bound"))(
                self.handle,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                stream,
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        check(&self.fns, "cuLaunchKernel", rc)
    }
}

// ---------------------------------------------------------------------------
// Streams / events
// ---------------------------------------------------------------------------

/// A CUDA stream (RAII). `raw()` exposes the handle for launch calls.
pub struct Stream {
    pub fns: Fns,
    pub handle: CUstream,
}

impl Stream {
    fn create(fns: &Fns, flags: u32) -> Result<Stream> {
        let mut handle: CUstream = 0;
        // SAFETY: out-param.
        let rc = unsafe { (fns.cuStreamCreate.expect("bound"))(&mut handle, flags) };
        check(fns, "cuStreamCreate", rc)?;
        Ok(Stream { fns: *fns, handle })
    }

    fn create_priority(fns: &Fns, priority: i32) -> Result<Stream> {
        let mut handle: CUstream = 0;
        // SAFETY: out-param.
        let rc =
            unsafe { (fns.cuStreamCreateWithPriority.expect("bound"))(&mut handle, 0, priority) };
        check(fns, "cuStreamCreateWithPriority", rc)?;
        Ok(Stream { fns: *fns, handle })
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: stream synchronize.
        let rc = unsafe { (self.fns.cuStreamSynchronize.expect("bound"))(self.handle) };
        check(&self.fns, "cuStreamSynchronize", rc)
    }

    pub fn priority(&self) -> Result<i32> {
        let mut p = 0;
        // SAFETY: out-param.
        let rc = unsafe { (self.fns.cuStreamGetPriority.expect("bound"))(self.handle, &mut p) };
        check(&self.fns, "cuStreamGetPriority", rc)?;
        Ok(p)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: stream destroy (best effort).
            unsafe { (self.fns.cuStreamDestroy.expect("bound"))(self.handle) };
        }
    }
}

/// A CUDA event (RAII).
pub struct Event {
    pub fns: Fns,
    pub handle: usize,
}

impl Event {
    fn create(fns: &Fns, disable_timing: bool) -> Result<Event> {
        let flags = if disable_timing {
            super::ffi::CU_EVENT_DISABLE_TIMING
        } else {
            0
        };
        let mut handle: usize = 0;
        // SAFETY: out-param.
        let rc = unsafe { (fns.cuEventCreate.expect("bound"))(&mut handle, flags) };
        check(fns, "cuEventCreate", rc)?;
        Ok(Event { fns: *fns, handle })
    }

    pub fn record(&self, stream: CUstream) -> Result<()> {
        // SAFETY: event record.
        let rc = unsafe { (self.fns.cuEventRecord.expect("bound"))(self.handle, stream) };
        check(&self.fns, "cuEventRecord", rc)
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: event synchronize.
        let rc = unsafe { (self.fns.cuEventSynchronize.expect("bound"))(self.handle) };
        check(&self.fns, "cuEventSynchronize", rc)
    }

    /// Milliseconds between `start` and `self` (both timing-enabled).
    pub fn elapsed_ms(&self, start: &Event) -> Result<f64> {
        let mut ms = 0.0f32;
        // SAFETY: out-param.
        let rc = unsafe {
            (self.fns.cuEventElapsedTime.expect("bound"))(&mut ms, start.handle, self.handle)
        };
        check(&self.fns, "cuEventElapsedTime", rc)?;
        Ok(f64::from(ms))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: event destroy (best effort).
            unsafe { (self.fns.cuEventDestroy.expect("bound"))(self.handle) };
        }
    }
}

// ---------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------

/// A device memory allocation (RAII).
pub struct DeviceBuffer {
    pub fns: Fns,
    pub ptr: CUdeviceptr,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn alloc(fns: &Fns, bytes: usize) -> Result<DeviceBuffer> {
        if bytes == 0 {
            // The driver rejects zero-size allocations; allocate one byte so
            // empty arenas still have a valid (never-dereferenced) pointer.
            return DeviceBuffer::alloc(fns, 1);
        }
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: out-param.
        let rc = unsafe { (fns.cuMemAlloc.expect("bound"))(&mut ptr, bytes) };
        check(fns, "cuMemAlloc", rc)?;
        Ok(DeviceBuffer {
            fns: *fns,
            ptr,
            bytes,
        })
    }

    pub fn upload(&self, host: &[u8]) -> Result<()> {
        debug_assert_eq!(host.len(), self.bytes);
        // SAFETY: host bytes length == allocation size; synchronous copy.
        let rc = unsafe {
            (self.fns.cuMemcpyHtoD.expect("bound"))(self.ptr, host.as_ptr() as *const _, host.len())
        };
        check(&self.fns, "cuMemcpyHtoD", rc)
    }

    pub fn download(&self, host: &mut [u8]) -> Result<()> {
        debug_assert_eq!(host.len(), self.bytes);
        // SAFETY: host buffer length == allocation size.
        let rc = unsafe {
            (self.fns.cuMemcpyDtoH.expect("bound"))(
                host.as_mut_ptr() as *mut _,
                self.ptr,
                host.len(),
            )
        };
        check(&self.fns, "cuMemcpyDtoH", rc)
    }

    pub fn upload_async(&self, host: &[u8], stream: CUstream) -> Result<()> {
        debug_assert_eq!(host.len(), self.bytes);
        // SAFETY: as upload but stream-ordered.
        let rc = unsafe {
            (self.fns.cuMemcpyHtoDAsync.expect("bound"))(
                self.ptr,
                host.as_ptr() as *const _,
                host.len(),
                stream,
            )
        };
        check(&self.fns, "cuMemcpyHtoDAsync", rc)
    }

    pub fn download_async(&self, host: &mut [u8], stream: CUstream) -> Result<()> {
        debug_assert_eq!(host.len(), self.bytes);
        // SAFETY: as download but stream-ordered.
        let rc = unsafe {
            (self.fns.cuMemcpyDtoHAsync.expect("bound"))(
                host.as_mut_ptr() as *mut _,
                self.ptr,
                host.len(),
                stream,
            )
        };
        check(&self.fns, "cuMemcpyDtoHAsync", rc)
    }

    /// Raw device pointer for kernel parameters.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: device free (best effort).
            unsafe { (self.fns.cuMemFree.expect("bound"))(self.ptr) };
        }
    }
}

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

/// A captured+instantiated CUDA graph executable (RAII) for one fixed
/// (kernel, grid, buffers) launch shape.
pub struct GraphExec {
    pub fns: Fns,
    pub handle: CUgraphExec,
}

impl GraphExec {
    /// Capture the given closure's kernel launches on `stream` and
    /// instantiate. `stream` must be otherwise idle.
    pub fn capture<F>(fns: &Fns, stream: CUstream, launch: F) -> Result<GraphExec>
    where
        F: FnOnce() -> Result<()>,
    {
        // SAFETY: begin capture in relaxed mode.
        let rc = unsafe {
            (fns.cuStreamBeginCapture.expect("bound"))(
                stream,
                super::ffi::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        check(fns, "cuStreamBeginCapture", rc)?;
        launch()?;
        let mut graph: CUgraph = 0;
        // SAFETY: out-param.
        let rc = unsafe { (fns.cuStreamEndCapture.expect("bound"))(stream, &mut graph) };
        check(fns, "cuStreamEndCapture", rc)?;
        let mut exec: CUgraphExec = 0;
        // SAFETY: instantiate with default flags.
        let rc = unsafe { (fns.cuGraphInstantiateWithFlags.expect("bound"))(&mut exec, graph, 0) };
        // The captured graph is no longer needed once instantiated.
        // SAFETY: graph destroy.
        unsafe { (fns.cuGraphDestroy.expect("bound"))(graph) };
        check(fns, "cuGraphInstantiateWithFlags", rc)?;
        Ok(GraphExec {
            fns: *fns,
            handle: exec,
        })
    }

    pub fn launch(&self, stream: CUstream) -> Result<()> {
        // SAFETY: graph launch on a stream.
        let rc = unsafe { (self.fns.cuGraphLaunch.expect("bound"))(self.handle, stream) };
        check(&self.fns, "cuGraphLaunch", rc)
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: graph exec destroy (best effort).
            unsafe { (self.fns.cuGraphExecDestroy.expect("bound"))(self.handle) };
        }
    }
}

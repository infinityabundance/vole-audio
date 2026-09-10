//! CUDA driver-API wrapper (Phase G): context, module, streams, events,
//! device buffers — all RAII, all through the audited `ffi::Fns`.
//!
//! Threading rule: driver API calls are bound to the *current* context of
//! the calling thread. `Cuda::open` makes the new context current for this
//! thread, and every subsequent call in this module happens on the same
//! thread (courts run single-threaded), so no context juggling is needed.
//!
//! Ownership (structural, mirroring the HIP graph fixed in Phase J): the
//! driver library, the context, and the device identity live in one
//! [`CudaContext`], and every GPU resource (`Module`, `Function`, `Stream`,
//! `Event`, `DeviceBuffer`, `GraphExec`) retains an `Arc<CudaContext>`. The
//! context therefore cannot be destroyed — and the driver cannot be dlclosed —
//! while any dependent resource is still alive, whatever order the owner
//! drops its fields in. `Function` additionally retains its `Module`, so a
//! kernel handle cannot outlive its module.
//!
//! Affinity (the other half of resource identity): CUDA's driver API is
//! *thread-current-context* based, so lifetime alone is not enough — a buffer
//! from context A must not be used while context B happens to be current on
//! this thread. Every context-dependent operation enters its owner through
//! [`CudaContext::enter`], which makes the owning context current for the
//! duration of the call (`cuCtxSetCurrent`, restoring the previous value on
//! drop — only when a different context was current), and every API that takes
//! a stream takes a `&Stream` whose context identity is checked. Resource
//! identity is thus `driver lifetime + context identity + current-thread
//! affinity`.

use crate::error::{Error, Result};
use std::ffi::{CString, c_void};
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use super::ffi::{
    ATTR_CLOCK_RATE, ATTR_CONCURRENT_MANAGED_ACCESS, ATTR_GLOBAL_L1_CACHE_SUPPORTED,
    ATTR_HOST_REGISTER_SUPPORTED, ATTR_KERNEL_EXEC_TIMEOUT, ATTR_MAX_THREADS_PER_BLOCK,
    ATTR_MAX_THREADS_PER_MULTIPROCESSOR, ATTR_MULTIPROCESSOR_COUNT, ATTR_PCI_BUS_ID,
    ATTR_PCI_DEVICE_ID, ATTR_PCI_DOMAIN_ID, ATTR_STREAM_PRIORITIES_SUPPORTED,
    ATTR_UNIFIED_ADDRESSING, CUcontext, CUdeviceptr, CUevent, CUgraph, CUgraphExec, CUmodule,
    CUresult, CUstream, Driver, Fns, cuda_error,
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
            clock_rate_khz: get(ATTR_CLOCK_RATE)?,
            max_threads_per_block: get(ATTR_MAX_THREADS_PER_BLOCK)?,
            max_threads_per_multiprocessor: get(ATTR_MAX_THREADS_PER_MULTIPROCESSOR)?,
            unified_addressing: get(ATTR_UNIFIED_ADDRESSING)?,
            stream_priorities_supported: get(ATTR_STREAM_PRIORITIES_SUPPORTED)?,
            concurrent_managed_access: get(ATTR_CONCURRENT_MANAGED_ACCESS)?,
            host_register_supported: get(ATTR_HOST_REGISTER_SUPPORTED)?,
            global_l1_cache_supported: get(ATTR_GLOBAL_L1_CACHE_SUPPORTED)?,
            kernel_exec_timeout: get(ATTR_KERNEL_EXEC_TIMEOUT)?,
        };
        let _ = (major, minor);
        Ok(info)
    }
}

/// The CUDA driver + context + device identity, owned together.
///
/// Declared so that `Drop::drop` (which destroys the context) runs while
/// `fns` and the dlopen'd driver are still alive: for a struct with a manual
/// `Drop`, the body runs first and the fields are dropped afterwards in
/// declaration order.
pub struct CudaContext {
    pub fns: Fns,
    /// Driver version (e.g. 13030 for 13.3).
    pub driver_version: i32,
    pub device: DeviceInfo,
    /// Current context for this thread (created by `open`).
    pub context: usize,
    /// Least / greatest meaningful stream priority of this context
    /// (`cuCtxGetStreamPriorityRange`). Lower numbers = higher priority;
    /// out-of-range requests are clamped by the driver.
    pub priority_least: i32,
    pub priority_greatest: i32,
    /// Keep the dlopen handle alive last.
    _driver: Driver,
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext")
            .field("driver_version", &self.driver_version)
            .field("device", &self.device.name)
            .field("sm", &self.device.sm)
            .finish()
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // SAFETY: context destroy (best effort at teardown). Every resource
        // that needs this context holds an `Arc` to it, so this runs last.
        if self.context != 0 {
            unsafe { (self.fns.cuCtxDestroy.expect("bound"))(self.context) };
        }
    }
}

/// Makes one context current on the calling host thread for its lifetime.
///
/// CUDA's driver API operates on the calling thread's *current* context, so a
/// resource that is alive but not current cannot be used (the driver returns
/// `CUDA_ERROR_INVALID_CONTEXT`). Entering is cheap when the owning context is
/// already current, and otherwise switches to it with `cuCtxSetCurrent`,
/// restoring the exact previous value on drop. `cuCtxPushCurrent` is
/// deliberately not used: a context created by `cuCtxCreate` is already on the
/// calling thread's context stack, and pushing it again fails with rc 201
/// (verified on driver 610.57.04).
///
/// The guard is **neither `Send` nor `Sync`**: it represents thread-local
/// driver state, so moving it to another thread would restore the saved context
/// on the wrong thread and leave the entering thread switched. The
/// `PhantomData<Rc<()>>` marker makes that a compile-time property rather than
/// a comment.
pub(crate) struct CurrentContextGuard<'a> {
    ctx: &'a CudaContext,
    /// `None` when nothing had to change; otherwise the context to restore
    /// (`0` meaning "no current context").
    previous: Option<CUcontext>,
    /// `Rc<()>` is neither `Send` nor `Sync`, and is zero-sized.
    _thread_bound: PhantomData<Rc<()>>,
}

impl CudaContext {
    /// Make this context current on the calling thread until the guard drops,
    /// which restores whatever was current before.
    pub(crate) fn enter(&self) -> Result<CurrentContextGuard<'_>> {
        let mut current: CUcontext = 0;
        // SAFETY: out-param.
        let rc = unsafe { (self.fns.cuCtxGetCurrent.expect("bound"))(&mut current) };
        check(&self.fns, "cuCtxGetCurrent", rc)?;
        if current == self.context {
            return Ok(CurrentContextGuard {
                ctx: self,
                previous: None,
                _thread_bound: PhantomData,
            });
        }
        // SAFETY: make this context current on the calling thread.
        let rc = unsafe { (self.fns.cuCtxSetCurrent.expect("bound"))(self.context) };
        check(&self.fns, "cuCtxSetCurrent", rc)?;
        Ok(CurrentContextGuard {
            ctx: self,
            previous: Some(current),
            _thread_bound: PhantomData,
        })
    }

    /// True when this context is the calling thread's current context.
    ///
    /// Public because it is the query the affinity guard performs: callers may
    /// ask whether a resource's owner is current on this thread without
    /// entering it.
    pub fn is_current(&self) -> bool {
        let mut current: CUcontext = 0;
        // SAFETY: out-param; a failure leaves `current` at zero.
        let rc = unsafe { (self.fns.cuCtxGetCurrent.expect("bound"))(&mut current) };
        rc == 0 && current == self.context
    }
}

impl Drop for CurrentContextGuard<'_> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            // SAFETY: best effort; restores the exact context that was current
            // before this guard entered (0 = no current context). Runs on the
            // thread that entered, which is the thread the state belongs to.
            unsafe { (self.ctx.fns.cuCtxSetCurrent.expect("bound"))(previous) };
        }
    }
}

/// Owns a freshly created context for the window in which `Cuda::open` has not
/// yet built its RAII owner.
///
/// `cuCtxCreate` pushes the new context onto the calling thread's stack, and the
/// two setup queries that follow it are fallible. Without this guard an error
/// (or a panic) between creation and the `Arc<CudaContext>` would return from a
/// safe function having leaked a context and left the caller's context stack
/// altered. Disarmed once the real owner exists.
struct ProvisionalContext {
    fns: Fns,
    context: CUcontext,
    armed: bool,
}

impl Drop for ProvisionalContext {
    fn drop(&mut self) {
        if !self.armed || self.context == 0 {
            return;
        }
        // Detach it first when it is still current on this thread, then destroy.
        let mut current: CUcontext = 0;
        // SAFETY: out-param; a failure leaves `current` at zero and we simply
        // destroy below.
        let rc = unsafe { (self.fns.cuCtxGetCurrent.expect("bound"))(&mut current) };
        if rc == 0 && current == self.context {
            let mut popped: CUcontext = 0;
            // SAFETY: pops the context created for this guard.
            unsafe { (self.fns.cuCtxPopCurrent.expect("bound"))(&mut popped) };
        }
        // SAFETY: destroys the context this guard owns.
        unsafe { (self.fns.cuCtxDestroy.expect("bound"))(self.context) };
    }
}

// Compile-time assertion: the affinity guard must not be Send (identity of the
// current context is per-thread). If a future refactor made it Send, this
// instantiation becomes ambiguous and fails to compile.
const _: fn() = || {
    trait AmbiguousIfSend<A> {
        fn some_item() {}
    }
    impl<T: ?Sized> AmbiguousIfSend<()> for T {}
    impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
    let _ = <CurrentContextGuard<'static> as AmbiguousIfSend<_>>::some_item;
};

/// The CUDA module-loading / JIT-cache configuration **observed** from the
/// process environment, for receipts.
///
/// `Cuda::open` never mutates the process environment: it is a safe public
/// function and cannot assume a single-threaded process, where `set_var` would
/// race concurrent environment reads. Runners that want deterministic JIT
/// (`EAGER`) and no on-disk JIT cache (`CUDA_CACHE_DISABLE=1`) export those
/// before `exec`; this records what was actually in effect, so configuration is
/// evidence rather than hidden mutation. Values are `EAGER`/`LAZY`/`unset` and
/// `1`/`0`/`unset`.
pub fn module_loading_policy() -> (String, String) {
    fn read(key: &str) -> String {
        std::env::var(key)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "unset".to_string())
    }
    (read("CUDA_MODULE_LOADING"), read("CUDA_CACHE_DISABLE"))
}

/// Reject a cross-context pairing through the safe API (e.g. a kernel from
/// context A launched onto a stream from context B). Distinct contexts have
/// distinct address spaces, so such a call is never meaningful.
fn ensure_same_context(
    what: &str,
    owner: &Arc<CudaContext>,
    other: &Arc<CudaContext>,
) -> Result<()> {
    if Arc::ptr_eq(owner, other) {
        Ok(())
    } else {
        Err(Error::malformed(format!(
            "{what} belong to different CUDA contexts"
        )))
    }
}

/// A CUDA session: a shared handle to the driver/context/device identity.
///
/// `Cuda` is a thin owner; the heavy state lives in [`CudaContext`], which
/// every resource keeps alive. `Deref` keeps the historical `cuda.fns` /
/// `cuda.device` / `cuda.context` field reads meaningful while the ownership
/// graph underneath is structural.
#[derive(Clone)]
pub struct Cuda {
    pub ctx: Arc<CudaContext>,
}

impl Deref for Cuda {
    type Target = CudaContext;
    fn deref(&self) -> &CudaContext {
        &self.ctx
    }
}

impl std::fmt::Debug for Cuda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cuda").field("ctx", &*self.ctx).finish()
    }
}

impl Cuda {
    /// Open the driver, pick `ordinal` (default 0), create a context, then pop
    /// that context off the calling thread's context stack so it is *floating*
    /// on return.
    ///
    /// The library deliberately does **not** mutate the process environment
    /// (see [`module_loading_policy`]): `Cuda::open` is a safe public function
    /// and cannot assume a single-threaded process, where `set_var` would race
    /// arbitrary concurrent environment reads. A runner that wants deterministic
    /// JIT (EAGER) and no on-disk JIT cache exports `CUDA_MODULE_LOADING` /
    /// `CUDA_CACHE_DISABLE` before `exec`; receipts record what was in effect.
    pub fn open(ordinal: i32) -> Result<Cuda> {
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
        // Context creation: bind the exported `cuCtxCreate` (= cuCtxCreate_v4)
        // with its documented four-argument signature
        // `(CUcontext*, CUctxCreateParams*, unsigned int flags, CUdevice)`;
        // `NULL` ctxCreateParams creates a regular context (cuda.h 6481).
        // Legacy `cuCtxCreate_v2` (three-arg) on this driver yields a context
        // that rejects later allocation with rc 201 (verified, driver
        // 610.57.04). `cuCtxCreate` pushes the new context onto this thread's
        // context stack.
        let mut context: usize = 0;
        // SAFETY: out-param; NULL params (regular context); default flags.
        let rc = unsafe {
            (fns.cuCtxCreate.expect("bound"))(&mut context, std::ptr::null_mut(), 0, ordinal)
        };
        check(&fns, "cuCtxCreate", rc)?;
        // From here until the RAII owner exists, the created context must be
        // destroyed on any early return (and the caller's context stack left as
        // it was). `ProvisionalContext` owns it until disarmed below.
        let mut provisional = ProvisionalContext {
            fns,
            context,
            armed: true,
        };
        // Stream priority range: context query, the only documented source,
        // performed while the new context is still current.
        let (mut least, mut greatest) = (0, 0);
        // SAFETY: out-params.
        let rc =
            unsafe { (fns.cuCtxGetStreamPriorityRange.expect("bound"))(&mut least, &mut greatest) };
        check(&fns, "cuCtxGetStreamPriorityRange", rc)?;
        // Pop the new context immediately: on return it is *floating*, and the
        // calling thread's context stack is exactly as it was before. Two
        // consequences matter for embedding this library:
        //   * `open` does not permanently alter its caller's current context;
        //   * the final `Arc` may be dropped on any thread without leaving a
        //     destroyed context current on the creating thread (`cuCtxDestroy`
        //     does not detach a context that is current elsewhere).
        // Every operation re-attaches via `CudaContext::enter` anyway.
        let mut popped: usize = 0;
        // SAFETY: out-param; pops the context created just above.
        let rc = unsafe { (fns.cuCtxPopCurrent.expect("bound"))(&mut popped) };
        check(&fns, "cuCtxPopCurrent", rc)?;
        if popped != context {
            return Err(Error::new(
                crate::error::Kind::External,
                format!(
                    "cuCtxPopCurrent returned {popped:#x}, expected the created context \
                     {context:#x}"
                ),
            ));
        }
        // The context is now floating and fully set up: hand ownership to the
        // RAII context, disarming the provisional guard (nothing below can fail).
        let fns = provisional.fns;
        provisional.armed = false;
        Ok(Cuda {
            ctx: Arc::new(CudaContext {
                fns,
                driver_version: version,
                device,
                context,
                priority_least: least,
                priority_greatest: greatest,
                _driver: driver,
            }),
        })
    }

    /// The shared context owner (retained by every resource).
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Highest available stream priority. `cuCtxGetStreamPriorityRange`
    /// returns (least, greatest) where the *greatest* priority is the
    /// numerically lowest value (lower numbers = higher priority); 0 is the
    /// default. Verified on driver 610.57.04: range (0, -5); a stream
    /// created at -5 reads back -5, while +1 clamps to 0.
    pub fn highest_priority(&self) -> i32 {
        if self.device.stream_priorities_supported != 0 {
            self.priority_greatest
        } else {
            0
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        let _guard = self.ctx.enter()?;
        // SAFETY: context synchronize.
        let rc = unsafe { (self.fns.cuCtxSynchronize.expect("bound"))() };
        check(&self.fns, "cuCtxSynchronize", rc)
    }

    /// Load a module from PTX/cubin bytes.
    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        Module::load(&self.ctx, image)
    }

    /// Load a module from a file (PTX text or cubin).
    pub fn load_module_file(&self, path: &Path) -> Result<Module> {
        let bytes = std::fs::read(path)?;
        self.load_module(&bytes)
    }

    pub fn create_stream(&self) -> Result<Stream> {
        Stream::create(&self.ctx, 0)
    }

    pub fn create_stream_priority(&self, priority: i32) -> Result<Stream> {
        Stream::create_priority(&self.ctx, priority)
    }

    pub fn create_event(&self, disable_timing: bool) -> Result<Event> {
        Event::create(&self.ctx, disable_timing)
    }

    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer> {
        DeviceBuffer::alloc(&self.ctx, bytes)
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Loaded CUDA module (PTX or cubin image) — retains the context, and is
/// retained by every `Function` resolved from it.
pub struct Module {
    pub inner: Arc<ModuleInner>,
}

/// The module handle + its owning context. `cuModuleUnload` runs when the
/// last `Arc<ModuleInner>` drops — i.e. only after every `Function` resolved
/// from the module is gone, and only while the context is still alive.
pub struct ModuleInner {
    pub ctx: Arc<CudaContext>,
    pub handle: CUmodule,
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: module unload (best effort); the context is alive
            // because this struct retains it, and no `Function` outlives it
            // by construction. Enter the context so unload targets it.
            // Bind the guard to a named local: `let _ = ...` would drop it at
            // the end of this statement and unload with the wrong context.
            let _guard = self.ctx.enter();
            unsafe { (self.ctx.fns.cuModuleUnload.expect("bound"))(self.handle) };
        }
    }
}

impl Module {
    fn load(ctx: &Arc<CudaContext>, image: &[u8]) -> Result<Module> {
        let _guard = ctx.enter()?;
        let fns = &ctx.fns;
        // Module load goes through cuModuleLoadDataEx (never the plain entry,
        // which fails PTX JIT with rc 218 on the tested driver), with the
        // correct JIT options: CU_JIT_ERROR_LOG_BUFFER (=5) + its required
        // size option CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES (=6, type
        // unsigned int, in/out). Values frozen from cuda.h (ffi constants +
        // pinned by test).
        //
        // PTX text input must be NUL-terminated: cuModuleLoadDataEx treats the
        // image as a C string and ptxas parses past the buffer end otherwise.
        // The heap bytes that follow an unterminated Vec<u8> decide whether
        // the parse happens to succeed or fails with a tail-located ptxas
        // error ("Unexpected non-ASCII character" / "Parsing error near '['"
        // at a line near the module end) — process-state dependent and
        // intermittent. Terminate explicitly so the parse is deterministic.
        let mut terminated = image.to_vec();
        terminated.push(0);
        let mut handle: CUmodule = 0;
        let log_bytes: usize = 1 << 15;
        let mut log = vec![0u8; log_bytes];
        let mut log_used: u32 = log_bytes as u32;
        let mut opts: [i32; 2] = [
            super::ffi::CU_JIT_ERROR_LOG_BUFFER,
            super::ffi::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut vals: [*mut c_void; 2] = [
            log.as_mut_ptr() as *mut c_void,
            (&mut log_used as *mut u32) as *mut c_void,
        ];
        // SAFETY: terminated bytes live for the call (driver copies/JITs
        // them); the log buffer and its size slot are written by the JIT and
        // read below.
        let rc = unsafe {
            (fns.cuModuleLoadDataEx.expect("bound"))(
                &mut handle,
                terminated.as_ptr() as *const _,
                2,
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
        Ok(Module {
            inner: Arc::new(ModuleInner {
                ctx: ctx.clone(),
                handle,
            }),
        })
    }

    /// Look up a kernel entry by (mangled-free) export name. The returned
    /// `Function` retains this module (and therefore the context).
    pub fn function(&self, name: &str) -> Result<Function> {
        let _guard = self.inner.ctx.enter()?;
        let cname = CString::new(name).map_err(|_| Error::malformed("kernel name has NUL"))?;
        let mut f: usize = 0;
        // SAFETY: out-param; cname lives for the call; the module handle is
        // owned by `self.inner` and alive.
        let rc = unsafe {
            (self.inner.ctx.fns.cuModuleGetFunction.expect("bound"))(
                &mut f,
                self.inner.handle,
                cname.as_ptr(),
            )
        };
        check(&self.inner.ctx.fns, "cuModuleGetFunction", rc)?;
        Ok(Function {
            module: self.inner.clone(),
            handle: f,
            entry: name.to_string(),
        })
    }

    /// The context this module belongs to.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.inner.ctx
    }
}

/// A kernel function handle. Retains its module owner (and thereby the
/// context), so it cannot outlive either.
#[derive(Clone)]
pub struct Function {
    module: Arc<ModuleInner>,
    pub handle: usize,
    /// Export name (evidence).
    pub entry: String,
}

impl Function {
    /// The context this function belongs to.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.module.ctx
    }

    /// Launch with parameters given as 8-byte little-endian values in order
    /// (device pointers and 64-bit scalars; the kernel ABI takes one 64-bit
    /// parameter slot each).
    ///
    /// `stream` must belong to the same context as this kernel; both the
    /// context identity and the calling thread's current context are
    /// established here.
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        stream: &Stream,
        params: &[u64],
    ) -> Result<()> {
        ensure_same_context("a kernel and its stream", &self.module.ctx, stream.ctx())?;
        let _guard = self.module.ctx.enter()?;
        let fns = &self.module.ctx.fns;
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
            (fns.cuLaunchKernel.expect("bound"))(
                self.handle,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                stream.raw(),
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        check(fns, "cuLaunchKernel", rc)
    }
}

// ---------------------------------------------------------------------------
// Streams / events
// ---------------------------------------------------------------------------

/// A CUDA stream (RAII). Retains the context, so it cannot outlive it; the raw
/// handle is private so a stream from one context cannot be handed to a
/// resource owned by another.
pub struct Stream {
    ctx: Arc<CudaContext>,
    handle: CUstream,
}

impl Stream {
    fn create(ctx: &Arc<CudaContext>, flags: u32) -> Result<Stream> {
        let _guard = ctx.enter()?;
        let mut handle: CUstream = 0;
        // SAFETY: out-param.
        let rc = unsafe { (ctx.fns.cuStreamCreate.expect("bound"))(&mut handle, flags) };
        check(&ctx.fns, "cuStreamCreate", rc)?;
        Ok(Stream {
            ctx: ctx.clone(),
            handle,
        })
    }

    fn create_priority(ctx: &Arc<CudaContext>, priority: i32) -> Result<Stream> {
        let _guard = ctx.enter()?;
        let mut handle: CUstream = 0;
        // SAFETY: out-param.
        let rc = unsafe {
            (ctx.fns.cuStreamCreateWithPriority.expect("bound"))(&mut handle, 0, priority)
        };
        check(&ctx.fns, "cuStreamCreateWithPriority", rc)?;
        Ok(Stream {
            ctx: ctx.clone(),
            handle,
        })
    }

    /// The context this stream belongs to.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Raw stream handle. Private to this module: every public operation takes
    /// a `&Stream`, so context identity is checked rather than assumed.
    pub(crate) fn raw(&self) -> CUstream {
        self.handle
    }

    pub fn synchronize(&self) -> Result<()> {
        let _guard = self.ctx.enter()?;
        // SAFETY: stream synchronize.
        let rc = unsafe { (self.ctx.fns.cuStreamSynchronize.expect("bound"))(self.handle) };
        check(&self.ctx.fns, "cuStreamSynchronize", rc)
    }

    pub fn priority(&self) -> Result<i32> {
        let _guard = self.ctx.enter()?;
        let mut p = 0;
        // SAFETY: out-param.
        let rc = unsafe { (self.ctx.fns.cuStreamGetPriority.expect("bound"))(self.handle, &mut p) };
        check(&self.ctx.fns, "cuStreamGetPriority", rc)?;
        Ok(p)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: stream destroy (best effort); the context is alive
            // because this struct retains it. Enter it so destroy targets it.
            let _guard = self.ctx.enter();
            unsafe { (self.ctx.fns.cuStreamDestroy.expect("bound"))(self.handle) };
        }
    }
}

/// A CUDA event (RAII). Retains the context.
pub struct Event {
    ctx: Arc<CudaContext>,
    handle: CUevent,
}

impl Event {
    fn create(ctx: &Arc<CudaContext>, disable_timing: bool) -> Result<Event> {
        let _guard = ctx.enter()?;
        let flags = if disable_timing {
            super::ffi::CU_EVENT_DISABLE_TIMING
        } else {
            0
        };
        let mut handle: CUevent = 0;
        // SAFETY: out-param.
        let rc = unsafe { (ctx.fns.cuEventCreate.expect("bound"))(&mut handle, flags) };
        check(&ctx.fns, "cuEventCreate", rc)?;
        Ok(Event {
            ctx: ctx.clone(),
            handle,
        })
    }

    /// The context this event belongs to.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        ensure_same_context("an event and its stream", &self.ctx, stream.ctx())?;
        let _guard = self.ctx.enter()?;
        // SAFETY: event record.
        let rc = unsafe { (self.ctx.fns.cuEventRecord.expect("bound"))(self.handle, stream.raw()) };
        check(&self.ctx.fns, "cuEventRecord", rc)
    }

    pub fn synchronize(&self) -> Result<()> {
        let _guard = self.ctx.enter()?;
        // SAFETY: event synchronize.
        let rc = unsafe { (self.ctx.fns.cuEventSynchronize.expect("bound"))(self.handle) };
        check(&self.ctx.fns, "cuEventSynchronize", rc)
    }

    /// Milliseconds between `start` and `self` (both timing-enabled).
    pub fn elapsed_ms(&self, start: &Event) -> Result<f64> {
        ensure_same_context("two events", &self.ctx, start.ctx())?;
        let _guard = self.ctx.enter()?;
        let mut ms = 0.0f32;
        // SAFETY: out-param.
        let rc = unsafe {
            (self.ctx.fns.cuEventElapsedTime.expect("bound"))(&mut ms, start.handle, self.handle)
        };
        check(&self.ctx.fns, "cuEventElapsedTime", rc)?;
        Ok(f64::from(ms))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: event destroy (best effort); the context is alive
            // because this struct retains it.
            let _guard = self.ctx.enter();
            unsafe { (self.ctx.fns.cuEventDestroy.expect("bound"))(self.handle) };
        }
    }
}

// ---------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------

/// A device memory allocation (RAII). Retains the context, so `cuMemFree`
/// targets a live context and the driver cannot be dlclosed underneath it.
pub struct DeviceBuffer {
    ctx: Arc<CudaContext>,
    pub ptr: CUdeviceptr,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn alloc(ctx: &Arc<CudaContext>, bytes: usize) -> Result<DeviceBuffer> {
        if bytes == 0 {
            // The driver rejects zero-size allocations; allocate one byte so
            // empty arenas still have a valid (never-dereferenced) pointer.
            return DeviceBuffer::alloc(ctx, 1);
        }
        let _guard = ctx.enter()?;
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: out-param.
        let rc = unsafe { (ctx.fns.cuMemAlloc.expect("bound"))(&mut ptr, bytes) };
        check(&ctx.fns, "cuMemAlloc", rc)?;
        Ok(DeviceBuffer {
            ctx: ctx.clone(),
            ptr,
            bytes,
        })
    }

    /// The context this allocation belongs to.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn upload(&self, host: &[u8]) -> Result<()> {
        debug_assert_eq!(host.len(), self.bytes);
        let _guard = self.ctx.enter()?;
        // SAFETY: host bytes length == allocation size; synchronous copy.
        let rc = unsafe {
            (self.ctx.fns.cuMemcpyHtoD.expect("bound"))(
                self.ptr,
                host.as_ptr() as *const _,
                host.len(),
            )
        };
        check(&self.ctx.fns, "cuMemcpyHtoD", rc)
    }

    pub fn download(&self, host: &mut [u8]) -> Result<()> {
        debug_assert_eq!(host.len(), self.bytes);
        let _guard = self.ctx.enter()?;
        // SAFETY: host buffer length == allocation size.
        let rc = unsafe {
            (self.ctx.fns.cuMemcpyDtoH.expect("bound"))(
                host.as_mut_ptr() as *mut _,
                self.ptr,
                host.len(),
            )
        };
        check(&self.ctx.fns, "cuMemcpyDtoH", rc)
    }

    /// Download only the leading `host.len()` bytes of the allocation (the
    /// device block may be sized for a larger maximum quantum; the actual
    /// transfer size is what the counters must record).
    pub fn download_prefix(&self, host: &mut [u8]) -> Result<()> {
        debug_assert!(host.len() <= self.bytes);
        let _guard = self.ctx.enter()?;
        // SAFETY: host buffer length <= allocation size.
        let rc = unsafe {
            (self.ctx.fns.cuMemcpyDtoH.expect("bound"))(
                host.as_mut_ptr() as *mut _,
                self.ptr,
                host.len(),
            )
        };
        check(&self.ctx.fns, "cuMemcpyDtoH", rc)
    }

    /// Stream-ordered upload. `stream` must belong to this buffer's context.
    pub fn upload_async(&self, host: &[u8], stream: &Stream) -> Result<()> {
        debug_assert_eq!(host.len(), self.bytes);
        ensure_same_context("a buffer and its stream", &self.ctx, stream.ctx())?;
        let _guard = self.ctx.enter()?;
        // SAFETY: as upload but stream-ordered.
        let rc = unsafe {
            (self.ctx.fns.cuMemcpyHtoDAsync.expect("bound"))(
                self.ptr,
                host.as_ptr() as *const _,
                host.len(),
                stream.raw(),
            )
        };
        check(&self.ctx.fns, "cuMemcpyHtoDAsync", rc)
    }

    /// Stream-ordered download. `stream` must belong to this buffer's context.
    pub fn download_async(&self, host: &mut [u8], stream: &Stream) -> Result<()> {
        debug_assert_eq!(host.len(), self.bytes);
        ensure_same_context("a buffer and its stream", &self.ctx, stream.ctx())?;
        let _guard = self.ctx.enter()?;
        // SAFETY: as download but stream-ordered.
        let rc = unsafe {
            (self.ctx.fns.cuMemcpyDtoHAsync.expect("bound"))(
                host.as_mut_ptr() as *mut _,
                self.ptr,
                host.len(),
                stream.raw(),
            )
        };
        check(&self.ctx.fns, "cuMemcpyDtoHAsync", rc)
    }

    /// Raw device pointer for kernel parameters.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: device free (best effort); the context is alive because
            // this struct retains it. Enter it so the free targets it.
            let _guard = self.ctx.enter();
            unsafe { (self.ctx.fns.cuMemFree.expect("bound"))(self.ptr) };
        }
    }
}

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

/// A captured+instantiated CUDA graph executable (RAII) for one fixed
/// (kernel, grid, buffers) launch shape. Retains the context.
pub struct GraphExec {
    ctx: Arc<CudaContext>,
    handle: CUgraphExec,
}

impl GraphExec {
    /// Capture the given closure's kernel launches on `stream` and
    /// instantiate. `stream` must be otherwise idle, and belong to `ctx`.
    pub fn capture<F>(ctx: &Arc<CudaContext>, stream: &Stream, launch: F) -> Result<GraphExec>
    where
        F: FnOnce() -> Result<()>,
    {
        ensure_same_context("a graph and its stream", ctx, stream.ctx())?;
        let _guard = ctx.enter()?;
        let stream = stream.raw();
        let fns = &ctx.fns;
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
            ctx: ctx.clone(),
            handle: exec,
        })
    }

    /// Launch the instantiated graph on `stream` (same context required).
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        ensure_same_context("a graph and its stream", &self.ctx, stream.ctx())?;
        let _guard = self.ctx.enter()?;
        // SAFETY: graph launch on a stream.
        let rc = unsafe { (self.ctx.fns.cuGraphLaunch.expect("bound"))(self.handle, stream.raw()) };
        check(&self.ctx.fns, "cuGraphLaunch", rc)
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: graph exec destroy (best effort); the context is alive
            // because this struct retains it.
            let _guard = self.ctx.enter();
            unsafe { (self.ctx.fns.cuGraphExecDestroy.expect("bound"))(self.handle) };
        }
    }
}

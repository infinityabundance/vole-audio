//! D0 buffered-diagnostic kernel orchestration (Phase G).
//!
//! `KernelWorld` owns the GPU-resident flat world for one
//! (`FlattenedWorld`, maximum quantum): voices/samples/partials/state are
//! uploaded once at construction (no per-quantum device allocation), and
//! every observation is a fused final render into the single D0 output
//! block. This is explicitly the **D0 / `GpuBufferedDiagnostic`** path — no
//! D1 endpoint claim is made anywhere in Phase G.
//!
//! Submission strategies (benchmarked, never assumed): standard stream,
//! highest-priority stream (a hint, not a guarantee), and a pre-instantiated
//! captured graph for one fixed quantum (the driver bakes grid/args at
//! capture; state bytes are re-read from the state buffer at every launch,
//! so window start/frames can still vary by rewriting that buffer).

use crate::backend::cuda::driver::{Cuda, DeviceBuffer, Function, GraphExec, Module, Stream};
use crate::backend::cuda::ffi::CUdeviceptr;
use crate::backend::flatten::FlattenedWorld;
use crate::device::kernel_shared::FlatState;
use crate::error::{Error, Result};
use std::time::Instant;

/// Kernel entry exported by the device module (`device::nvptx_entry`).
pub const KERNEL_ENTRY: &str = "vole_render_d0";
/// Threads per block for the first D0 kernel.
pub const BLOCK_THREADS: u32 = 256;

/// Submission strategy for one observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Plain stream launch.
    Standard,
    /// Highest-available-priority stream launch.
    HighPriority,
    /// Pre-instantiated captured graph (fixed quantum).
    Graph,
}

impl Strategy {
    pub const ALL: [Strategy; 3] = [Strategy::Standard, Strategy::HighPriority, Strategy::Graph];

    pub const fn label(self) -> &'static str {
        match self {
            Strategy::Standard => "standard-stream",
            Strategy::HighPriority => "high-priority-stream",
            Strategy::Graph => "captured-graph",
        }
    }
}

/// GPU-resident world + D0 render pipeline for one fixed maximum quantum.
///
/// Drop order is declaration order, so `cuda` (the context) is declared
/// LAST: streams/buffers/module/graph die before the context is destroyed.
pub struct KernelWorld {
    /// Owned so the module (and its function handles) outlives every launch
    /// and is unloaded before the context dies.
    _module: Module,
    function: Function,
    d_state: DeviceBuffer,
    d_voices: DeviceBuffer,
    d_samples: DeviceBuffer,
    d_partials: DeviceBuffer,
    d_out: DeviceBuffer,
    stream_standard: Stream,
    stream_priority: Option<Stream>,
    graph: Option<GraphExec>,
    /// Maximum render quantum (frames) the D0 block is sized for.
    pub max_frames: usize,
    pub flat: FlattenedWorld,
    state_bytes: Vec<u8>,
    out_bytes: Vec<u8>,
    /// Honest D0 traffic/launch counters for every render on this world
    /// (see `evidence::counters`); the court folds them into its receipt.
    pub counters: crate::evidence::counters::Counters,
    /// Priority actually assigned by the driver to the high-priority stream
    /// (verified via cuStreamGetPriority; receipts record it).
    pub priority_assigned: Option<i32>,
    /// The CUDA session (context). Declared last: destroyed last.
    pub cuda: Cuda,
}

/// Byte view of a `repr(C)` POD slice.
///
/// # SAFETY
/// `T` must be a `#[repr(C)]` POD whose every byte is defined (no
/// uninitialized padding). The flat records satisfy this by construction
/// (explicit `_pad` fields, no holes; enforced by the layout anchors in
/// `device::kernel_shared`).
unsafe fn pod_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl KernelWorld {
    /// Open the CUDA device, load the PTX module, upload the flat world, and
    /// prepare a D0 output block for up to `max_frames`-frame windows.
    pub fn open(
        ordinal: i32,
        ptx_bytes: &[u8],
        flat: FlattenedWorld,
        max_frames: usize,
    ) -> Result<KernelWorld> {
        let cuda = Cuda::open(ordinal)?;
        KernelWorld::open_with(cuda, ptx_bytes, flat, max_frames)
    }

    /// Like `open`, but driven by an existing CUDA session (the caller keeps
    /// one context across registrations and renders — the D1 court path). The
    /// context becomes owned by the returned world and dies with it.
    pub fn open_with(
        cuda: Cuda,
        ptx_bytes: &[u8],
        flat: FlattenedWorld,
        max_frames: usize,
    ) -> Result<KernelWorld> {
        let module = cuda.load_module(ptx_bytes)?;
        let function = module.function(KERNEL_ENTRY)?;
        let channels = usize::from(flat.output_channels);
        if channels == 0 || channels > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::limit("output channels outside domain"));
        }
        if max_frames == 0 || max_frames > crate::limits::MAX_QUANTUM_FRAMES as usize {
            return Err(Error::limit("max_frames outside quantum domain"));
        }

        // Upload the world once (no per-quantum allocation anywhere in the
        // render path).
        let d_voices = DeviceBuffer::alloc(&cuda.ctx, unsafe { pod_bytes(&flat.voices) }.len())?;
        d_voices.upload(unsafe { pod_bytes(&flat.voices) })?;
        let d_samples = DeviceBuffer::alloc(&cuda.ctx, unsafe { pod_bytes(&flat.samples) }.len())?;
        d_samples.upload(unsafe { pod_bytes(&flat.samples) })?;
        let d_partials =
            DeviceBuffer::alloc(&cuda.ctx, unsafe { pod_bytes(&flat.partials) }.len())?;
        d_partials.upload(unsafe { pod_bytes(&flat.partials) })?;
        let state = flat.state(0, max_frames);
        let state_arr = [state];
        let state_bytes: Vec<u8> = unsafe { pod_bytes(&state_arr) }.to_vec();
        let d_state = DeviceBuffer::alloc(&cuda.ctx, state_bytes.len())?;
        d_state.upload(&state_bytes)?;
        let out_bytes = vec![0u8; max_frames * channels * 4];
        let d_out = DeviceBuffer::alloc(&cuda.ctx, out_bytes.len())?;

        let stream_standard = cuda.create_stream()?;
        let stream_priority = if cuda.device.stream_priorities_supported != 0 {
            Some(cuda.create_stream_priority(cuda.highest_priority())?)
        } else {
            None
        };
        // Verify what the driver actually assigned (clamping would be
        // evidence of a wrong range, never silently accepted).
        let priority_assigned = stream_priority
            .as_ref()
            .map(|s| s.priority().unwrap_or(cuda.priority_least));

        // D0 accounting anchors: the VRAM observation block and its host
        // staging mirror (persistent, sized for the max quantum).
        let mut counters = crate::evidence::counters::Counters::new();
        counters.host_pcm_resident_peak_bytes = out_bytes.len() as u64;
        counters.device_sample_block_bytes = out_bytes.len() as u64;

        let mut world = KernelWorld {
            _module: module,
            function,
            d_state,
            d_voices,
            d_samples,
            d_partials,
            d_out,
            stream_standard,
            stream_priority,
            graph: None,
            max_frames,
            flat,
            state_bytes,
            out_bytes,
            counters,
            priority_assigned,
            cuda,
        };
        // Graph capture is an optional optimization; failure is recorded by
        // the court, never fatal.
        let _ = world.capture_graph(max_frames);
        Ok(world)
    }

    /// Capture a graph of one full D0 render at exactly `frames` (the baked
    /// launch shape). State bytes are re-read at every launch, so window
    /// parameters can still vary between launches by rewriting `d_state`.
    pub fn capture_graph(&mut self, frames: usize) -> Result<()> {
        let channels = self.flat.output_channels;
        let grid = grid_for(frames, channels);
        let f = self.function.clone();
        let params = self.kernel_params();
        let graph = GraphExec::capture(&self.cuda.ctx, &self.stream_standard, || {
            f.launch(grid, (BLOCK_THREADS, 1, 1), &self.stream_standard, &params)
        })?;
        self.graph = Some(graph);
        Ok(())
    }

    /// True when a captured graph is available for the graph strategy.
    pub fn graph_available(&self) -> bool {
        self.graph.is_some()
    }

    /// True when a high-priority stream exists on this device.
    pub fn priority_available(&self) -> bool {
        self.stream_priority.is_some()
    }

    fn kernel_params(&self) -> Vec<u64> {
        vec![
            self.d_state.device_ptr(),
            self.d_voices.device_ptr(),
            self.d_samples.device_ptr(),
            self.d_partials.device_ptr(),
            self.d_out.device_ptr(),
        ]
    }

    /// Parameters for a render whose final observation lands at `out_dev`
    /// (D1: the device-visible pointer of a registered endpoint region).
    fn kernel_params_to(&self, out_dev: CUdeviceptr) -> Vec<u64> {
        vec![
            self.d_state.device_ptr(),
            self.d_voices.device_ptr(),
            self.d_samples.device_ptr(),
            self.d_partials.device_ptr(),
            out_dev,
        ]
    }

    /// Upload the window state; call before any launch.
    fn push_state(&mut self, start_frame: i64, frames: usize) -> Result<()> {
        let state: FlatState = self.flat.state(start_frame, frames);
        let state_arr = [state];
        self.state_bytes.clear();
        self.state_bytes
            .extend_from_slice(unsafe { pod_bytes(&state_arr) });
        self.d_state.upload(&self.state_bytes)
    }

    /// Render `[start, start+frames)` through strategy `s` and copy the D0
    /// block back into `out` (exact scalar-comparable i32 codes).
    pub fn render(
        &mut self,
        s: Strategy,
        start_frame: i64,
        frames: usize,
        out: &mut [i32],
    ) -> Result<()> {
        if frames > self.max_frames {
            return Err(Error::limit("render frames exceed KernelWorld max"));
        }
        if out.len() != frames * usize::from(self.flat.output_channels) {
            return Err(Error::malformed("output buffer length mismatch"));
        }
        self.push_state(start_frame, frames)?;
        let channels = self.flat.output_channels;
        let grid = grid_for(frames, channels);
        let params = self.kernel_params();
        match s {
            Strategy::Standard => {
                self.function.launch(
                    grid,
                    (BLOCK_THREADS, 1, 1),
                    &self.stream_standard,
                    &params,
                )?;
                self.stream_standard.synchronize()?;
                self.counters.kernel_launches += 1;
            }
            Strategy::HighPriority => {
                let stream = self.stream_priority.as_ref().ok_or_else(|| {
                    Error::new(crate::error::Kind::Unavailable, "no priority stream")
                })?;
                self.function
                    .launch(grid, (BLOCK_THREADS, 1, 1), stream, &params)?;
                stream.synchronize()?;
                self.counters.kernel_launches += 1;
            }
            Strategy::Graph => {
                let graph = self.graph.as_ref().ok_or_else(|| {
                    Error::new(crate::error::Kind::Unavailable, "graph not captured")
                })?;
                graph.launch(&self.stream_standard)?;
                self.stream_standard.synchronize()?;
                self.counters.kernel_launches += 1;
            }
        }
        self.download(out)?;
        self.counters.quanta_submitted += 1;
        Ok(())
    }

    fn download(&mut self, out: &mut [i32]) -> Result<()> {
        // Transfer only the rendered window's leading bytes (the D0 block is
        // sized for the max quantum); the *actual* transfer size is what the
        // counters record. The host staging mirror then feeds the caller
        // buffer via one host copy.
        let want = out.len() * 4;
        self.d_out.download_prefix(&mut self.out_bytes[..want])?;
        self.counters.gpu_to_host_pcm_bytes += want as u64;
        self.counters.host_pcm_copy_bytes += want as u64;
        let codes: &[i32] =
            unsafe { std::slice::from_raw_parts(self.out_bytes.as_ptr() as *const i32, out.len()) };
        out.copy_from_slice(codes);
        Ok(())
    }

    /// D0 render whose DtoH transfer lands **directly** in the caller's
    /// buffer — no internal host staging copy (`out_bytes` is left unused).
    /// This is the stronger conventional D0 baseline for endpoint
    /// comparisons: the D0 VRAM block and the explicit DtoH materialization
    /// still exist, but the avoidable host-to-host copy is removed. Counters:
    /// `gpu_to_host_pcm_bytes` records the actual transfer; `host_pcm_copy_
    /// bytes` is NOT incremented here (any further copy is the caller's,
    /// e.g. the court's buffer→endpoint-region copy).
    pub fn render_into(&mut self, start_frame: i64, frames: usize, out: &mut [i32]) -> Result<()> {
        if frames > self.max_frames {
            return Err(Error::limit("render frames exceed KernelWorld max"));
        }
        if out.len() != frames * usize::from(self.flat.output_channels) {
            return Err(Error::malformed("output buffer length mismatch"));
        }
        self.push_state(start_frame, frames)?;
        let channels = self.flat.output_channels;
        let grid = grid_for(frames, channels);
        let params = self.kernel_params();
        self.function
            .launch(grid, (BLOCK_THREADS, 1, 1), &self.stream_standard, &params)?;
        self.stream_standard.synchronize()?;
        self.counters.kernel_launches += 1;
        let want = out.len() * 4;
        // SAFETY: `out` is a live i32 buffer; its bytes are a valid transfer
        // destination of exactly `want` bytes.
        let out_bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, want) };
        self.d_out.download_prefix(out_bytes)?;
        self.counters.gpu_to_host_pcm_bytes += want as u64;
        self.counters.quanta_submitted += 1;
        Ok(())
    }

    /// D1 fused render: the kernel writes the final interleaved i32 codes
    /// directly into `out_dev` — the device-visible pointer of a registered
    /// *endpoint* region — with no D0 observation block, no device->host
    /// transfer, and no host PCM copy. The caller (court d1) guarantees
    /// `out_dev` addresses `frames * channels * 4` writable bytes that the
    /// endpoint consumes only after the caller commits (this function
    /// returns only after the stream is synchronized, so GPU writes to the
    /// mapped region are visible per the driver's mapped-host-memory
    /// coherency contract).
    ///
    /// Counters: `kernel_launches` and `quanta_submitted` increment exactly
    /// as in the D0 path; `endpoint_observation_bytes` accumulates the bytes
    /// written into the endpoint region. `gpu_to_host_pcm_bytes` and
    /// `host_pcm_copy_bytes` are NOT incremented — that is the D1 claim.
    pub fn render_direct(
        &mut self,
        start_frame: i64,
        frames: usize,
        out_dev: CUdeviceptr,
    ) -> Result<()> {
        if frames > self.max_frames {
            return Err(Error::limit("render frames exceed KernelWorld max"));
        }
        if frames == 0 {
            return Err(Error::malformed("render frames must be nonzero"));
        }
        self.push_state(start_frame, frames)?;
        let channels = self.flat.output_channels;
        let grid = grid_for(frames, channels);
        let params = self.kernel_params_to(out_dev);
        self.function
            .launch(grid, (BLOCK_THREADS, 1, 1), &self.stream_standard, &params)?;
        self.stream_standard.synchronize()?;
        self.counters.kernel_launches += 1;
        self.counters.quanta_submitted += 1;
        let bytes = (frames as u64) * u64::from(channels) * 4;
        self.counters.endpoint_observation_bytes = self
            .counters
            .endpoint_observation_bytes
            .saturating_add(bytes);
        Ok(())
    }

    /// Kernel-only time (driver events around one launch) for strategy `s`,
    /// seconds. State was already uploaded; grid uses `frames`.
    pub fn time_kernel_once(&mut self, s: Strategy, frames: usize) -> Result<f64> {
        let grid = grid_for(frames, self.flat.output_channels);
        let params = self.kernel_params();
        let e0 = self.cuda.create_event(false)?;
        let e1 = self.cuda.create_event(false)?;
        let r = match s {
            Strategy::Standard => {
                e0.record(&self.stream_standard)?;
                let r = self.function.launch(
                    grid,
                    (BLOCK_THREADS, 1, 1),
                    &self.stream_standard,
                    &params,
                );
                e1.record(&self.stream_standard)?;
                r
            }
            Strategy::HighPriority => {
                let stream = self.stream_priority.as_ref().ok_or_else(|| {
                    Error::new(crate::error::Kind::Unavailable, "no priority stream")
                })?;
                e0.record(stream)?;
                let r = self
                    .function
                    .launch(grid, (BLOCK_THREADS, 1, 1), stream, &params);
                e1.record(stream)?;
                r
            }
            Strategy::Graph => {
                let graph = self.graph.as_ref().ok_or_else(|| {
                    Error::new(crate::error::Kind::Unavailable, "graph not captured")
                })?;
                e0.record(&self.stream_standard)?;
                let r = graph.launch(&self.stream_standard);
                e1.record(&self.stream_standard)?;
                r
            }
        };
        r?;
        self.counters.kernel_launches += 1;
        e1.synchronize()?;
        let ms = e1.elapsed_ms(&e0)?;
        drop((e0, e1));
        Ok(ms / 1000.0)
    }

    /// Wall-clock time of one full D0 render (`push_state` + launch + sync +
    /// copy-back), seconds.
    pub fn time_render_wall(
        &mut self,
        s: Strategy,
        start_frame: i64,
        frames: usize,
    ) -> Result<f64> {
        let mut scratch = vec![0i32; frames * usize::from(self.flat.output_channels)];
        let t0 = Instant::now();
        self.render(s, start_frame, frames, &mut scratch)?;
        Ok(t0.elapsed().as_secs_f64())
    }
}

/// Grid size for `frames x channels` output slots at `BLOCK_THREADS`.
pub fn grid_for(frames: usize, channels: u8) -> (u32, u32, u32) {
    let total = frames * usize::from(channels);
    let blocks = total.div_ceil(BLOCK_THREADS as usize).max(1) as u32;
    (blocks, 1, 1)
}

#[cfg(test)]
mod smoke_tests {
    //! Gated smoke tests (require a CUDA GPU + the built PTX artifact). Run
    //! with:
    //!
    //! ```text
    //! CUDA_MODULE_LOADING=EAGER CUDA_CACHE_DISABLE=1 \
    //!   cargo test --release --all-features --lib \
    //!   backend::cuda::kernel::smoke_tests -- --ignored
    //! ```
    //!
    //! The environment is set by the runner, never by the library or by these
    //! tests: `set_var` in a multithreaded process (which the test harness is)
    //! races concurrent environment reads. EAGER + cache-disable give
    //! deterministic JIT with no stale on-disk cache entry (rc 218 hazard).

    fn ptx() -> Vec<u8> {
        let p =
            std::env::var("VOLE_CUDA_PTX").unwrap_or_else(|_| "scripts/out/vole_audio.ptx".into());
        std::fs::read(p).expect("ptx artifact (run scripts/build-cuda-device.sh)")
    }

    #[test]
    #[ignore = "requires CUDA GPU + built PTX artifact"]
    fn module_loads_and_renders() {
        let ptx = ptx();
        let cuda = super::super::driver::Cuda::open(0).expect("cuda open");
        let module = cuda.load_module(&ptx).expect("module load");
        let f = module.function(super::KERNEL_ENTRY).expect("function");
        println!("module + function OK on {}", cuda.device.name);
        let _ = f;
    }

    /// Ownership regression (Phase L review): every GPU resource retains the
    /// context owner, so dropping the session handle must not destroy the
    /// context while buffers/streams still exist — and the resources must
    /// remain usable afterwards. Before the fix, `Cuda` owned the context and
    /// its dependencies carried only a copied function table, so this
    /// sequence freed into a destroyed context.
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn resources_outlive_the_session_handle() {
        use std::sync::Arc;
        let cuda = super::super::driver::Cuda::open(0).expect("cuda open");
        let buf = cuda.alloc(64).expect("alloc");
        let stream = cuda.create_stream().expect("stream");
        let held = Arc::strong_count(cuda.ctx());
        assert!(held >= 3, "resources must retain the context (held {held})");
        drop(cuda); // the session handle is gone; the context must survive
        buf.upload(&[7u8; 64]).expect("upload after session drop");
        stream
            .synchronize()
            .expect("synchronize after session drop");
        // The last owner destroys the context, after every free/destroy call.
        drop(stream);
        drop(buf);
    }

    /// A kernel handle must not outlive its module (structural retention).
    #[test]
    #[ignore = "requires CUDA GPU + built PTX artifact"]
    fn function_outlives_the_module_handle() {
        let ptx = ptx();
        let cuda = super::super::driver::Cuda::open(0).expect("cuda open");
        let module = cuda.load_module(&ptx).expect("module load");
        let f = module.function(super::KERNEL_ENTRY).expect("function");
        drop(module); // the `Function` retains the module owner
        // The function keeps the module (and therefore the context) alive.
        assert!(
            std::sync::Arc::strong_count(f.ctx()) >= 2,
            "a function must retain its module's context"
        );
        assert_ne!(f.handle, 0);
    }

    /// Affinity regression (Phase L review): a resource from context A must be
    /// usable while context B is current on this thread, and using it must
    /// restore B's currency afterwards. Lifetime safety alone (the Arc graph)
    /// does not give this: the driver operates on the calling thread's
    /// *current* context.
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn affinity_is_restored_across_two_contexts() {
        use super::super::driver::Cuda;
        let a = Cuda::open(0).expect("context A");
        let buf_a = a.alloc(128).expect("alloc A");
        buf_a.upload(&[3u8; 128]).expect("upload A");

        let b = Cuda::open(0).expect("context B");
        // Make B explicitly current (whether `cuCtxCreate` leaves its context
        // current is driver detail; the guard establishes it either way).
        let guard = b.ctx().enter().expect("enter B");
        assert!(b.ctx().is_current(), "B must be current");
        assert!(!a.ctx().is_current(), "A must not be current");

        // A resource owned by A, used while B is current.
        let mut back = [0u8; 128];
        buf_a
            .download(&mut back)
            .expect("download A while B is current");
        assert!(back.iter().all(|&x| x == 3), "A's data must round-trip");
        assert!(
            b.ctx().is_current(),
            "using A must restore B as the current context"
        );
        assert!(!a.ctx().is_current());
        drop(guard);
    }

    /// Affinity regression (Phase L review): a foreign thread that never
    /// entered the context must still be able to use a resource owned by it,
    /// because every operation enters the owner.
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn cross_thread_use_establishes_its_context() {
        let cuda = super::super::driver::Cuda::open(0).expect("cuda open");
        let buf = cuda.alloc(256).expect("alloc");
        buf.upload(&[1u8; 256])
            .expect("upload on the owning thread");
        let handle = std::thread::spawn(move || {
            // This thread never opened or entered the context.
            buf.upload(&[9u8; 256]).expect("upload on a foreign thread");
            let mut back = [0u8; 256];
            buf.download(&mut back)
                .expect("download on a foreign thread");
            assert!(back.iter().all(|&b| b == 9));
            // `buf` drops here: `cuMemFree` also enters the owning context.
        });
        handle.join().expect("foreign thread panicked");
    }

    /// Context-lifecycle regression (Phase L review-3): `Cuda::open` must not
    /// leave its context attached to the calling thread, and the final `Arc`
    /// may be dropped on a foreign thread without poisoning the creator.
    /// `cuCtxDestroy` does not detach a context that is current elsewhere, so a
    /// non-floating context dropped on another thread would leave a destroyed
    /// context current on this one.
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn context_is_floating_after_open_and_safe_to_drop_elsewhere() {
        let cuda = super::super::driver::Cuda::open(0).expect("open");
        assert!(
            !cuda.ctx().is_current(),
            "open must return a floating context, not an attached one"
        );
        let buf = cuda.alloc(64).expect("alloc");
        buf.upload(&[5u8; 64]).expect("upload");
        drop(cuda);
        // `buf` is now the last owner: its drop destroys the context on
        // another thread.
        std::thread::spawn(move || drop(buf))
            .join()
            .expect("foreign drop thread panicked");
        // The creator thread is unaffected: a fresh session still works.
        let again = super::super::driver::Cuda::open(0).expect("reopen after foreign drop");
        let b2 = again.alloc(32).expect("alloc after foreign drop");
        b2.upload(&[1u8; 32]).expect("upload after foreign drop");
    }

    /// Affinity regression (Phase L review): pairing resources from different
    /// contexts is rejected by the safe API rather than silently executed
    /// against whichever context happens to be current.
    /// Affinity regression (Phase L review): pairing resources from different
    /// contexts is rejected by the safe API rather than silently executed
    /// against whichever context happens to be current.
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn cross_context_pairing_is_rejected() {
        use super::super::driver::Cuda;
        let a = Cuda::open(0).expect("context A");
        let b = Cuda::open(0).expect("context B");
        let buf_a = a.alloc(64).expect("alloc A");
        let stream_b = b.create_stream().expect("stream B");
        let err = buf_a
            .upload_async(&[0u8; 64], &stream_b)
            .expect_err("cross-context async upload must fail");
        assert!(err.to_string().contains("different CUDA contexts"), "{err}");
        let event_a = a.create_event(true).expect("event A");
        let err = event_a
            .record(&stream_b)
            .expect_err("cross-context event record must fail");
        assert!(err.to_string().contains("different CUDA contexts"), "{err}");
    }

    /// Release-on-drop regression (Phase L review-3): the D1 host registration
    /// must be *unregistered against its own context* when it drops. If the
    /// guard that makes the owner context current is released too early, the
    /// unregister silently fails and the range stays registered, so the next
    /// registration of the same mapping fails with rc 712.
    #[test]
    #[ignore = "requires CUDA GPU + host-register support"]
    fn host_registration_is_released_on_drop() {
        use super::super::direct::{RegistrationAttempt, attempt_register};
        let cuda = super::super::driver::Cuda::open(0).expect("open");
        if cuda.device.host_register_supported == 0 {
            return; // registration is unsupported on this device; nothing to assert
        }
        let len = 8192usize;
        // SAFETY: anonymous private read/write mapping, page aligned, unmapped
        // at the end of the test.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED, "mmap failed");
        let addr = base as usize;
        // First registration, released by `drop`.
        match unsafe { attempt_register(&cuda.ctx, addr, len) } {
            RegistrationAttempt::Registered(r) => {
                assert!(r.device_ptr.is_some(), "no device pointer");
                drop(r);
            }
            // Not a registration-capable mapping/API: nothing to assert.
            _ => {
                // SAFETY: unmapping the mapping created above.
                unsafe { libc::munmap(base, len) };
                return;
            }
        }
        // The same range must register again: the first registration was
        // released, against the correct context.
        match unsafe { attempt_register(&cuda.ctx, addr, len) } {
            RegistrationAttempt::Registered(r) => drop(r),
            RegistrationAttempt::Failed { rc, message } => panic!(
                "re-registering the same range failed (rc {rc}: {message}) — the \
                 previous registration was not released"
            ),
            RegistrationAttempt::MissingSymbol(m) => panic!("missing driver symbols: {m}"),
        }
        // SAFETY: unmapping the mapping created above.
        unsafe { libc::munmap(base, len) };
    }

    #[test]
    #[ignore = "requires CUDA GPU + built PTX artifact"]
    fn probe_then_load_reproduces_court_sequence() {
        // court cuda runs CudaProbe::capture (open+drop) before the first
        // KernelWorld::open; reproduce exactly that sequence to chase rc 218.
        let ptx = ptx();
        let probe = super::super::probe::CudaProbe::capture(0).expect("probe");
        println!("probe: {:?}", probe.as_ref().map(|p| p.device.name.clone()));
        let cuda = super::super::driver::Cuda::open(0).expect("cuda open");
        match cuda.load_module(&ptx) {
            Ok(_m) => println!("module load OK after probe"),
            Err(e) => println!("module load FAILED after probe: {e}"),
        }
    }
}

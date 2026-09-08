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
        let d_voices = DeviceBuffer::alloc(&cuda.fns, unsafe { pod_bytes(&flat.voices) }.len())?;
        d_voices.upload(unsafe { pod_bytes(&flat.voices) })?;
        let d_samples = DeviceBuffer::alloc(&cuda.fns, unsafe { pod_bytes(&flat.samples) }.len())?;
        d_samples.upload(unsafe { pod_bytes(&flat.samples) })?;
        let d_partials =
            DeviceBuffer::alloc(&cuda.fns, unsafe { pod_bytes(&flat.partials) }.len())?;
        d_partials.upload(unsafe { pod_bytes(&flat.partials) })?;
        let state = flat.state(0, max_frames);
        let state_arr = [state];
        let state_bytes: Vec<u8> = unsafe { pod_bytes(&state_arr) }.to_vec();
        let d_state = DeviceBuffer::alloc(&cuda.fns, state_bytes.len())?;
        d_state.upload(&state_bytes)?;
        let out_bytes = vec![0u8; max_frames * channels * 4];
        let d_out = DeviceBuffer::alloc(&cuda.fns, out_bytes.len())?;

        let stream_standard = cuda.create_stream()?;
        let stream_priority = if cuda.device.stream_priorities_supported != 0 {
            Some(cuda.create_stream_priority(cuda.highest_priority())?)
        } else {
            None
        };

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
        let f = self.function;
        let params = self.kernel_params();
        let graph = GraphExec::capture(&self.cuda.fns, self.stream_standard.handle, || {
            f.launch(
                grid,
                (BLOCK_THREADS, 1, 1),
                self.stream_standard.handle,
                &params,
            )
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
                    self.stream_standard.handle,
                    &params,
                )?;
                self.stream_standard.synchronize()?;
            }
            Strategy::HighPriority => {
                let stream = self.stream_priority.as_ref().ok_or_else(|| {
                    Error::new(crate::error::Kind::Unavailable, "no priority stream")
                })?;
                self.function
                    .launch(grid, (BLOCK_THREADS, 1, 1), stream.handle, &params)?;
                stream.synchronize()?;
            }
            Strategy::Graph => {
                let graph = self.graph.as_ref().ok_or_else(|| {
                    Error::new(crate::error::Kind::Unavailable, "graph not captured")
                })?;
                graph.launch(self.stream_standard.handle)?;
                self.stream_standard.synchronize()?;
            }
        }
        self.download(out)
    }

    fn download(&mut self, out: &mut [i32]) -> Result<()> {
        // The D0 block is sized for the max quantum; copy the whole block
        // back (simplest correct path) and read the leading frames.
        self.d_out.download(&mut self.out_bytes)?;
        let codes: &[i32] =
            unsafe { std::slice::from_raw_parts(self.out_bytes.as_ptr() as *const i32, out.len()) };
        out.copy_from_slice(codes);
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
                e0.record(self.stream_standard.handle)?;
                let r = self.function.launch(
                    grid,
                    (BLOCK_THREADS, 1, 1),
                    self.stream_standard.handle,
                    &params,
                );
                e1.record(self.stream_standard.handle)?;
                r
            }
            Strategy::HighPriority => {
                let stream = self.stream_priority.as_ref().ok_or_else(|| {
                    Error::new(crate::error::Kind::Unavailable, "no priority stream")
                })?;
                e0.record(stream.handle)?;
                let r = self
                    .function
                    .launch(grid, (BLOCK_THREADS, 1, 1), stream.handle, &params);
                e1.record(stream.handle)?;
                r
            }
            Strategy::Graph => {
                let graph = self.graph.as_ref().ok_or_else(|| {
                    Error::new(crate::error::Kind::Unavailable, "graph not captured")
                })?;
                e0.record(self.stream_standard.handle)?;
                let r = graph.launch(self.stream_standard.handle);
                e1.record(self.stream_standard.handle)?;
                r
            }
        };
        r?;
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
    //! with: VOLE_CUDA_PTX=scripts/out/vole_audio.ptx cargo test --release \
    //!     --lib backend::cuda::kernel::smoke_tests -- --ignored

    fn ptx() -> Vec<u8> {
        let p =
            std::env::var("VOLE_CUDA_PTX").unwrap_or_else(|_| "scripts/out/vole_audio.ptx".into());
        std::fs::read(p).expect("ptx artifact (run scripts/build-cuda-device.sh)")
    }

    #[test]
    #[ignore = "requires CUDA GPU + built PTX artifact"]
    fn module_loads_and_renders() {
        // Set eager module loading where supported so JIT errors surface at
        // load (LAZY defers them to first use on some drivers).
        unsafe { std::env::set_var("CUDA_MODULE_LOADING", "EAGER") };
        let ptx = ptx();
        let cuda = super::super::driver::Cuda::open(0).expect("cuda open");
        let module = cuda.load_module(&ptx).expect("module load");
        let f = module.function(super::KERNEL_ENTRY).expect("function");
        println!("module + function OK on {}", cuda.device.name);
        let _ = f;
    }

    #[test]
    #[ignore = "requires CUDA GPU + built PTX artifact"]
    fn probe_then_load_reproduces_court_sequence() {
        // court cuda runs CudaProbe::capture (open+drop) before the first
        // KernelWorld::open; reproduce exactly that sequence to chase rc 218.
        unsafe { std::env::set_var("CUDA_MODULE_LOADING", "EAGER") };
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

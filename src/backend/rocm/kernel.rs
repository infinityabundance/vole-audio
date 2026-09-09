//! ROCm kernel orchestration (Phase J) — the AMD host-side mirror of
//! `backend::cuda::kernel` + `backend::cuda::entropy`, driving the code
//! object's three entries (`vole_render_d0`, `vole_entropy_decode`,
//! `vole_upmix_mono_dup`) from `device::amdgcn_entry`.
//!
//! The flat world upload is byte-identical to the CUDA path (same shared
//! `device::kernel_shared` / `entropy_shared` plain-data records). The AMD
//! divergence is launch geometry: the kernels take `blocks_x`/`threads_x`
//! as explicit parameters AND are launched with those exact dimensions
//! (frozen contract, `device::geom`).
//!
//! Everything here is host `std`; the device-side semantics live in the
//! shared no_std modules, so differential testing
//! (`scalar == SIMD == CUDA == ROCm`) applies unchanged.

use crate::backend::entropy_flat::FlatEntropyJob;
use crate::backend::flatten::FlattenedWorld;
use crate::backend::rocm::ffi::Fns;
use crate::backend::rocm::runtime::{Arg, DeviceBuffer, Function, Module, Rocm};
use crate::device::entropy_shared::EntropyJobDesc;
use crate::device::geom::Grid;
use crate::device::kernel_shared::FlatState;
use crate::error::{Error, Result};

/// Kernel entries exported by the amdgcn device module.
pub const RENDER_ENTRY: &str = "vole_render_d0";
pub const ENTROPY_ENTRY: &str = "vole_entropy_decode";
pub const UPMIX_ENTRY: &str = "vole_upmix_mono_dup";

/// Threads per workgroup for the render kernel.
pub const RENDER_THREADS: u32 = 256;
/// Threads per workgroup for the entropy decode kernel (one thread/page).
pub const ENTROPY_THREADS: u32 = 128;
/// Threads per workgroup for the mono upmix kernel (one thread/frame).
pub const UPMIX_THREADS: u32 = 256;

/// Frozen grid-stride geometry for `total` work items at `threads` per
/// workgroup: `blocks = ceil(total/threads)`, never zero.
pub fn grid_for(total: usize, threads: u32) -> Grid {
    Grid::new(total.div_ceil(threads as usize).max(1) as u32, threads)
}

/// Byte view of a `repr(C)` POD slice.
///
/// # SAFETY
/// `T` must be a `#[repr(C)]` POD whose every byte is defined (no
/// uninitialized padding). The flat records satisfy this by construction
/// (explicit `_pad` fields, no holes; enforced by the layout anchors in
/// `device::kernel_shared` / `device::entropy_shared`).
unsafe fn pod_bytes<T: Sized>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// GPU-resident flat world + D0 render pipeline for one fixed maximum
/// quantum — the AMD mirror of `cuda::kernel::KernelWorld` (single
/// synchronized legacy-default-stream path; no strategy/graph surface).
///
/// Drop order is declaration order, so the session (declared LAST) dies
/// after the module and buffers.
pub struct RocmWorld {
    /// Owned so the module (and its function handles) outlives every launch
    /// and is unloaded before the session dies.
    module: Module,
    function: Function,
    d_state: DeviceBuffer,
    d_voices: DeviceBuffer,
    d_samples: DeviceBuffer,
    d_partials: DeviceBuffer,
    d_out: DeviceBuffer,
    /// Maximum render quantum (frames) the D0 block is sized for.
    pub max_frames: usize,
    pub flat: FlattenedWorld,
    state_bytes: Vec<u8>,
    out_bytes: Vec<u8>,
    /// Honest D0 traffic/launch counters for every render on this world
    /// (see `evidence::counters`); the court folds them into its receipt.
    pub counters: crate::evidence::counters::Counters,
    /// The ROCm session. Declared last: destroyed last.
    pub rocm: Rocm,
}

impl RocmWorld {
    /// Open the HIP session, load the AMDGPU code object, upload the flat
    /// world, and prepare a D0 output block for up to `max_frames`-frame
    /// windows.
    pub fn open(
        ordinal: i32,
        artifact_bytes: &[u8],
        flat: FlattenedWorld,
        max_frames: usize,
    ) -> Result<RocmWorld> {
        let rocm = Rocm::open(ordinal)?;
        let module = rocm.load_module(artifact_bytes)?;
        let function = module.function(RENDER_ENTRY)?;
        let channels = usize::from(flat.output_channels);
        if channels == 0 || channels > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::limit("output channels outside domain"));
        }
        if max_frames == 0 || max_frames > crate::limits::MAX_QUANTUM_FRAMES as usize {
            return Err(Error::limit("max_frames outside quantum domain"));
        }

        // Upload the world once (no per-quantum allocation anywhere in the
        // render path).
        let d_voices = DeviceBuffer::alloc(&rocm.fns, unsafe { pod_bytes(&flat.voices) }.len())?;
        d_voices.upload(unsafe { pod_bytes(&flat.voices) })?;
        let d_samples = DeviceBuffer::alloc(&rocm.fns, unsafe { pod_bytes(&flat.samples) }.len())?;
        d_samples.upload(unsafe { pod_bytes(&flat.samples) })?;
        let d_partials =
            DeviceBuffer::alloc(&rocm.fns, unsafe { pod_bytes(&flat.partials) }.len())?;
        d_partials.upload(unsafe { pod_bytes(&flat.partials) })?;
        let state = flat.state(0, max_frames);
        let state_arr = [state];
        let state_bytes: Vec<u8> = unsafe { pod_bytes(&state_arr) }.to_vec();
        let d_state = DeviceBuffer::alloc(&rocm.fns, state_bytes.len())?;
        d_state.upload(&state_bytes)?;
        let out_bytes = vec![0u8; max_frames * channels * 4];
        let d_out = DeviceBuffer::alloc(&rocm.fns, out_bytes.len())?;

        // D0 accounting anchors: the VRAM observation block and its host
        // staging mirror (persistent, sized for the max quantum).
        let mut counters = crate::evidence::counters::Counters::new();
        counters.host_pcm_resident_peak_bytes = out_bytes.len() as u64;
        counters.device_sample_block_bytes = out_bytes.len() as u64;

        Ok(RocmWorld {
            module,
            function,
            d_state,
            d_voices,
            d_samples,
            d_partials,
            d_out,
            max_frames,
            flat,
            state_bytes,
            out_bytes,
            counters,
            rocm,
        })
    }

    /// The session (borrowed; e.g. a shared entropy world launching on the
    /// same device).
    pub fn session(&self) -> &Rocm {
        &self.rocm
    }

    /// The loaded module (borrowed; its function handles stay valid while
    /// the world lives).
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Render argument list: the five world pointers plus the frozen launch
    /// geometry the AMD kernel takes as parameters.
    fn render_args(&self, out_dev: u64) -> Vec<Arg> {
        let grid = grid_for(self.max_frames, RENDER_THREADS);
        vec![
            Arg::Ptr(self.d_state.device_ptr()),
            Arg::Ptr(self.d_voices.device_ptr()),
            Arg::Ptr(self.d_samples.device_ptr()),
            Arg::Ptr(self.d_partials.device_ptr()),
            Arg::Ptr(out_dev),
            Arg::U32(grid.blocks_x),
            Arg::U32(grid.threads_x),
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

    /// Geometry for one window of `frames` (output slots = frames x
    /// channels; the kernel indexes output sample slots).
    fn window_grid(frames: usize, channels: u8) -> Grid {
        grid_for(frames * usize::from(channels), RENDER_THREADS)
    }

    /// Render `[start, start+frames)` into the D0 block and copy the result
    /// back into `out` (exact scalar-comparable i32 codes).
    pub fn render(&mut self, start_frame: i64, frames: usize, out: &mut [i32]) -> Result<()> {
        if frames > self.max_frames {
            return Err(Error::limit("render frames exceed RocmWorld max"));
        }
        if out.len() != frames * usize::from(self.flat.output_channels) {
            return Err(Error::malformed("output buffer length mismatch"));
        }
        self.push_state(start_frame, frames)?;
        let grid = Self::window_grid(frames, self.flat.output_channels);
        let args = self.render_args(self.d_out.device_ptr());
        // Geometry must equal the kernel's declared parameters exactly.
        debug_assert_eq!(args[5], Arg::U32(grid.blocks_x));
        debug_assert_eq!(args[6], Arg::U32(grid.threads_x));
        self.function.launch(grid, &args)?;
        self.rocm.synchronize()?;
        self.counters.kernel_launches += 1;
        let want = out.len() * 4;
        self.d_out.download(&mut self.out_bytes[..want])?;
        self.counters.gpu_to_host_pcm_bytes += want as u64;
        self.counters.host_pcm_copy_bytes += want as u64;
        // SAFETY: out_bytes holds `want` bytes of i32 codes.
        let codes: &[i32] =
            unsafe { std::slice::from_raw_parts(self.out_bytes.as_ptr() as *const i32, out.len()) };
        out.copy_from_slice(codes);
        self.counters.quanta_submitted += 1;
        Ok(())
    }

    /// D1 fused render: the kernel writes the final interleaved i32 codes
    /// directly into `out_dev` — the device-visible pointer of a registered
    /// *endpoint* region — with no D0 observation block, no device->host
    /// transfer, and no host PCM copy. Returns only after full device
    /// synchronization, so GPU writes to the mapped region are visible to
    /// the host (mapped-host-memory coherency).
    ///
    /// Counters: `kernel_launches`/`quanta_submitted` increment;
    /// `endpoint_observation_bytes` accumulates the bytes written.
    /// `gpu_to_host_pcm_bytes` and `host_pcm_copy_bytes` are NOT incremented
    /// — that is the D1 claim.
    pub fn render_direct(&mut self, start_frame: i64, frames: usize, out_dev: u64) -> Result<()> {
        self.render_to(start_frame, frames, out_dev)?;
        let channels = self.flat.output_channels;
        let bytes = (frames as u64) * u64::from(channels) * 4;
        self.counters.endpoint_observation_bytes = self
            .counters
            .endpoint_observation_bytes
            .saturating_add(bytes);
        Ok(())
    }

    /// Render into a device buffer without endpoint accounting (used by the
    /// mono->upmix D1 path where the arena write is transient window
    /// scratch, exposed separately as the bounded GPU-global intermediate).
    pub fn render_to(&mut self, start_frame: i64, frames: usize, out_dev: u64) -> Result<()> {
        if frames > self.max_frames {
            return Err(Error::limit("render frames exceed RocmWorld max"));
        }
        if frames == 0 {
            return Err(Error::malformed("render frames must be nonzero"));
        }
        self.push_state(start_frame, frames)?;
        let channels = self.flat.output_channels;
        let grid = Self::window_grid(frames, channels);
        let args = self.render_args(out_dev);
        self.function.launch(grid, &args)?;
        self.rocm.synchronize()?;
        self.counters.kernel_launches += 1;
        self.counters.quanta_submitted += 1;
        Ok(())
    }

    /// Expand a mono device sample arena into an interleaved multi-channel
    /// region by duplication (L = R = ... = sample) on the device — the
    /// sampler/mix observation transform at the endpoint boundary. The
    /// destination is the registered endpoint region, so the bytes written
    /// count as endpoint observation. Uses this world's module + session
    /// (borrow-safe for per-chunk session loops).
    pub fn upmix(&mut self, src_dev: u64, dst_dev: u64, frames: u64, channels: u64) -> Result<()> {
        let function = self.module.function(UPMIX_ENTRY)?;
        let grid = grid_for(frames as usize, UPMIX_THREADS);
        let args = vec![
            Arg::Ptr(src_dev),
            Arg::Ptr(dst_dev),
            Arg::U64(frames),
            Arg::U64(channels),
            Arg::U32(grid.blocks_x),
            Arg::U32(grid.threads_x),
        ];
        function.launch(grid, &args)?;
        self.rocm.synchronize()?;
        self.counters.kernel_launches += 1;
        let bytes = frames * channels * 4;
        self.counters.endpoint_observation_bytes = self
            .counters
            .endpoint_observation_bytes
            .saturating_add(bytes);
        Ok(())
    }
}

/// GPU-resident entropy state + decode launch (AMD mirror of
/// `cuda::entropy::EntropyWorld`; geometry parameters appended).
///
/// The world either owns its session + module (`open`) or shares a caller
/// session/module (`open_shared`/`build` — the fused entropy->D1 endpoint
/// path). The session/module must outlive a shared world; drop order:
/// entropy world first.
pub struct EntropyWorldRocm {
    /// Owning session (None when sharing a caller session).
    session: Option<Rocm>,
    /// Module handle (None when the caller keeps the module alive).
    module: Option<Module>,
    fns: Fns,
    function: Function,
    d_pages: DeviceBuffer,
    d_streams: DeviceBuffer,
    d_payload: DeviceBuffer,
    d_mvalues: DeviceBuffer,
    d_mstarts: DeviceBuffer,
    d_mfreqs: DeviceBuffer,
    d_mranges: DeviceBuffer,
    d_cycle: DeviceBuffer,
    d_desc: DeviceBuffer,
    /// Uploaded descriptor (geometry mirror for launches).
    pub desc: EntropyJobDesc,
}

impl EntropyWorldRocm {
    /// Open a fresh HIP session + module and build a decode world over one
    /// flat job.
    pub fn open(
        ordinal: i32,
        artifact_bytes: &[u8],
        job: &FlatEntropyJob,
    ) -> Result<EntropyWorldRocm> {
        let session = Rocm::open(ordinal)?;
        let module = session.load_module(artifact_bytes)?;
        let function = module.function(ENTROPY_ENTRY)?;
        let mut world = EntropyWorldRocm::build(&session.fns, function, job)?;
        world.module = Some(module);
        world.session = Some(session);
        Ok(world)
    }

    /// Build a decode world over an existing session, loading the module
    /// from `artifact_bytes` (session must outlive this world).
    pub fn open_shared(
        session: &Rocm,
        artifact_bytes: &[u8],
        job: &FlatEntropyJob,
    ) -> Result<EntropyWorldRocm> {
        let module = session.load_module(artifact_bytes)?;
        let function = module.function(ENTROPY_ENTRY)?;
        let mut world = EntropyWorldRocm::build(&session.fns, function, job)?;
        world.module = Some(module);
        Ok(world)
    }

    /// Build a decode world reusing an already-loaded module + function.
    /// The module must outlive this world.
    pub fn build(fns: &Fns, function: Function, job: &FlatEntropyJob) -> Result<EntropyWorldRocm> {
        let mut world = EntropyWorldRocm {
            session: None,
            module: None,
            fns: *fns,
            function,
            d_pages: DeviceBuffer::alloc(fns, 1)?,
            d_streams: DeviceBuffer::alloc(fns, 1)?,
            d_payload: DeviceBuffer::alloc(fns, 1)?,
            d_mvalues: DeviceBuffer::alloc(fns, 2)?,
            d_mstarts: DeviceBuffer::alloc(fns, 1)?,
            d_mfreqs: DeviceBuffer::alloc(fns, 1)?,
            d_mranges: DeviceBuffer::alloc(fns, 1)?,
            d_cycle: DeviceBuffer::alloc(fns, 1)?,
            d_desc: DeviceBuffer::alloc(fns, 52)?,
            desc: EntropyJobDesc::zeroed(),
        };
        world.upload(job)?;
        Ok(world)
    }

    /// Upload a (possibly new) flat job: pages/streams/payload/models/cycle
    /// are re-uploaded and the descriptor updated.
    pub fn upload(&mut self, job: &FlatEntropyJob) -> Result<()> {
        let fns = self.fns;
        let upload = |buf: &DeviceBuffer, bytes: &[u8]| -> Result<()> {
            if bytes.is_empty() {
                return Ok(());
            }
            buf.upload(bytes)
        };
        let pages = DeviceBuffer::alloc(&fns, job.pages.len().max(1) * 56)?;
        upload(&pages, unsafe { pod_bytes(&job.pages) })?;
        let streams = DeviceBuffer::alloc(&fns, job.streams.len().max(1) * 24)?;
        upload(&streams, unsafe { pod_bytes(&job.streams) })?;
        let payload = DeviceBuffer::alloc(&fns, job.payload.len().max(1))?;
        upload(&payload, &job.payload)?;
        let mut mvalue_bytes = Vec::with_capacity(job.model_values.len() * 2);
        for v in &job.model_values {
            mvalue_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mvalues = DeviceBuffer::alloc(&fns, mvalue_bytes.len().max(2))?;
        upload(&mvalues, &mvalue_bytes)?;
        let mstarts = DeviceBuffer::alloc(&fns, job.model_starts.len().max(1) * 4)?;
        upload(&mstarts, unsafe { pod_bytes(&job.model_starts) })?;
        let mfreqs = DeviceBuffer::alloc(&fns, job.model_freqs.len().max(1) * 4)?;
        upload(&mfreqs, unsafe { pod_bytes(&job.model_freqs) })?;
        let mranges = DeviceBuffer::alloc(&fns, job.model_ranges.len().max(1) * 4)?;
        upload(&mranges, unsafe { pod_bytes(&job.model_ranges) })?;
        let cycle = DeviceBuffer::alloc(&fns, job.cycle.len().max(1) * 4)?;
        upload(&cycle, unsafe { pod_bytes(&job.cycle) })?;
        let mut desc = EntropyJobDesc::zeroed();
        desc.page_count = job.pages.len() as u32;
        desc.arena_samples = job.arena_samples as u32;
        desc.scratch_stride = job.scratch_stride() as u32;
        desc.payload_len = job.payload.len() as u32;
        desc.stream_count = job.streams.len() as u32;
        desc.mvalue_count = job.model_values.len() as u32;
        desc.mstart_count = job.model_starts.len() as u32;
        desc.mfreq_count = job.model_freqs.len() as u32;
        desc.mrange_count = job.model_ranges.len() as u32;
        desc.cycle_count = job.cycle.len() as u32;
        desc.statuses_len = job.pages.len() as u32;
        let desc_arr = [desc];
        let d_desc = DeviceBuffer::alloc(&fns, 52)?;
        upload(&d_desc, unsafe { pod_bytes(&desc_arr) })?;
        self.d_pages = pages;
        self.d_streams = streams;
        self.d_payload = payload;
        self.d_mvalues = mvalues;
        self.d_mstarts = mstarts;
        self.d_mfreqs = mfreqs;
        self.d_mranges = mranges;
        self.d_cycle = cycle;
        self.d_desc = d_desc;
        self.desc = desc;
        Ok(())
    }

    /// Launch the decode kernel for the uploaded job, writing into
    /// `out_dev` (a device arena or the registered endpoint region). Status
    /// bytes per page are downloaded and must all be 1 (exact decode); the
    /// scratch arena is per-launch (the documented H.2 limitation, mirrored
    /// from the CUDA path).
    fn launch(&self, out_dev: u64) -> Result<()> {
        let fns = &self.fns;
        let desc = &self.desc;
        let scratch_len = desc.page_count as usize * desc.scratch_stride as usize;
        let d_scratch = DeviceBuffer::alloc(fns, scratch_len.max(1))?;
        let d_status = DeviceBuffer::alloc(fns, desc.page_count as usize)?;
        let grid = grid_for(desc.page_count as usize, ENTROPY_THREADS);
        let args = vec![
            Arg::Ptr(self.d_desc.device_ptr()),
            Arg::Ptr(self.d_pages.device_ptr()),
            Arg::Ptr(self.d_streams.device_ptr()),
            Arg::Ptr(self.d_payload.device_ptr()),
            Arg::Ptr(self.d_mvalues.device_ptr()),
            Arg::Ptr(self.d_mstarts.device_ptr()),
            Arg::Ptr(self.d_mfreqs.device_ptr()),
            Arg::Ptr(self.d_mranges.device_ptr()),
            Arg::Ptr(self.d_cycle.device_ptr()),
            Arg::Ptr(out_dev),
            Arg::Ptr(d_scratch.device_ptr()),
            Arg::Ptr(d_status.device_ptr()),
            Arg::U32(grid.blocks_x),
            Arg::U32(grid.threads_x),
        ];
        self.function.launch(grid, &args)?;
        // SAFETY: hipDeviceSynchronize makes device writes visible.
        ffi::check(&self.fns, "hipDeviceSynchronize", unsafe {
            (self.fns.hipDeviceSynchronize.expect("bound"))()
        })?;
        let mut status = vec![0u8; desc.page_count as usize];
        d_status.download(&mut status)?;
        if let Some(bad) = status.iter().position(|&s| s == 0) {
            return Err(Error::new(
                crate::error::Kind::Internal,
                format!("entropy decode kernel failed on page {bad}"),
            ));
        }
        Ok(())
    }

    /// Decode the uploaded job into a fresh device arena and download it.
    pub fn decode(&self) -> Result<Vec<i32>> {
        let out = DeviceBuffer::alloc(&self.fns, self.desc.arena_samples as usize * 4)?;
        self.launch(out.device_ptr())?;
        let mut bytes = vec![0u8; self.desc.arena_samples as usize * 4];
        out.download(&mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| i32::from_le_bytes(*w))
            .collect())
    }

    /// Launch the decode kernel writing directly into `out_dev`
    /// (`out_samples >= desc.arena_samples`). Used by the fused
    /// entropy->endpoint D1 path where the arena is the registered region.
    pub fn decode_into(&self, out_dev: u64) -> Result<()> {
        self.launch(out_dev)
    }
}

use crate::backend::rocm::ffi;

/// Expand a mono device sample arena into an interleaved multi-channel
/// region by duplication (L = R = ... = sample) — the sampler/mix transform
/// at the observation boundary, executed on the device.
pub fn upmix_mono_dup(
    session: &Rocm,
    module: &Module,
    src_dev: u64,
    dst_dev: u64,
    frames: u64,
    channels: u64,
) -> Result<()> {
    let function = module.function(UPMIX_ENTRY)?;
    let grid = grid_for(frames as usize, UPMIX_THREADS);
    let args = vec![
        Arg::Ptr(src_dev),
        Arg::Ptr(dst_dev),
        Arg::U64(frames),
        Arg::U64(channels),
        Arg::U32(grid.blocks_x),
        Arg::U32(grid.threads_x),
    ];
    function.launch(grid, &args)?;
    session.synchronize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_the_frozen_contract() {
        // grid_for never returns a zero dimension for any work count.
        for total in [0usize, 1, 2, 511, 512, 513, 4096] {
            let g = grid_for(total, RENDER_THREADS);
            assert!(g.valid());
            assert_eq!(g.threads_x, RENDER_THREADS);
            assert!(g.stride() >= total as u64);
        }
        // Windows of the D1 court size (512 frames x 2 channels).
        let g = grid_for(512 * 2, RENDER_THREADS);
        assert_eq!(g.blocks_x, 4);
        // 16 entropy pages, one thread per page at 128 threads/block.
        let g = grid_for(16, ENTROPY_THREADS);
        assert_eq!(g.blocks_x, 1);
        assert_eq!(g.threads_x, 128);
    }

    #[test]
    fn entropy_desc_layout_matches_the_device_contract() {
        // The host allocates the descriptor buffer at the device-record
        // size; guard against silent layout drift.
        assert_eq!(std::mem::size_of::<EntropyJobDesc>(), 52);
        let d = EntropyJobDesc::zeroed();
        assert_eq!(d.page_count, 0);
    }
}

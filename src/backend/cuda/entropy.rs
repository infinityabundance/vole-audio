//! CUDA entropy decode host runtime (Phase H.2) — driver API.
//!
//! Loads the same PTX module as the render kernel and drives
//! `vole_entropy_decode` (one thread per entropy page). The entropy object
//! (payloads/models/page records) is uploaded and stays GPU-resident; a
//! decode job covers a bounded page range (an observation window), and the
//! decoded samples land in a GPU output arena — never a full-object decoded
//! waveform (H.2.17/H.2.18).
//!
//! Exactness: the kernel runs the same `device::entropy_shared` functions as
//! the host flat decoder on the identical job; the courts compare the
//! downloaded arena byte-for-byte with the host decode.

use super::driver::{Cuda, CudaContext, DeviceBuffer, Function, Module, Stream};
use crate::backend::entropy_flat::FlatEntropyJob;
use crate::device::entropy_shared::EntropyJobDesc;
use crate::error::{Error, Result};
use std::sync::Arc;

/// PTX entry of the entropy decode kernel.
pub const ENTROPY_KERNEL_ENTRY: &str = "vole_entropy_decode";

/// One-thread-per-page decode launch shape.
const BLOCK_THREADS: u32 = 128;

/// Raw pod bytes of a slice of plain-data records.
fn pod_bytes<T: Sized>(v: &[T]) -> &[u8] {
    // SAFETY: plain-data records (no padding secrets); slice lifetime bound.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, core::mem::size_of_val(v)) }
}

/// GPU-resident entropy state + decode launch.
///
/// The world either owns its CUDA session (`EntropyWorld::open`) or borrows
/// one (`open_shared` — used by the fused entropy->D1 endpoint path). The
/// session must outlive a shared world; the world retains the context, so
/// either order is safe.
pub struct EntropyWorld {
    /// Shared context owner (retained by every buffer and by `function`).
    ctx: Arc<CudaContext>,
    /// Owning session (None when sharing a caller session).
    session: Option<Cuda>,
    /// Module handle (None when the caller keeps the module alive).
    module: Option<Module>,
    function: Function,
    stream: Stream,
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

impl EntropyWorld {
    /// Open a fresh CUDA session and build a decode world over one flat job.
    pub fn open(ordinal: i32, ptx_bytes: &[u8], job: &FlatEntropyJob) -> Result<EntropyWorld> {
        let cuda = Cuda::open(ordinal)?;
        let mut world = EntropyWorld::open_shared(&cuda, ptx_bytes, job)?;
        world.session = Some(cuda);
        Ok(world)
    }

    /// Build a decode world over an existing CUDA session (session must
    /// outlive this world); loads the module from `ptx_bytes`.
    pub fn open_shared(
        cuda: &Cuda,
        ptx_bytes: &[u8],
        job: &FlatEntropyJob,
    ) -> Result<EntropyWorld> {
        let module = cuda.load_module(ptx_bytes)?;
        let function = module.function(ENTROPY_KERNEL_ENTRY)?;
        let mut world = EntropyWorld::build(&cuda.ctx, function, cuda.create_stream()?, job)?;
        world.module = Some(module);
        Ok(world)
    }

    /// Build a decode world reusing an already-loaded module + function and
    /// the caller's stream (per-window D1 jobs). The module must outlive
    /// this world.
    pub fn open_preloaded(
        ctx: &Arc<CudaContext>,
        function: Function,
        stream: Stream,
        job: &FlatEntropyJob,
    ) -> Result<EntropyWorld> {
        EntropyWorld::build(ctx, function, stream, job)
    }

    fn build(
        ctx: &Arc<CudaContext>,
        function: Function,
        stream: Stream,
        job: &FlatEntropyJob,
    ) -> Result<EntropyWorld> {
        let mut world = EntropyWorld {
            ctx: ctx.clone(),
            session: None,
            module: None,
            function,
            stream,
            d_pages: DeviceBuffer::alloc(ctx, 1)?,
            d_streams: DeviceBuffer::alloc(ctx, 1)?,
            d_payload: DeviceBuffer::alloc(ctx, 1)?,
            d_mvalues: DeviceBuffer::alloc(ctx, 2)?,
            d_mstarts: DeviceBuffer::alloc(ctx, 1)?,
            d_mfreqs: DeviceBuffer::alloc(ctx, 1)?,
            d_mranges: DeviceBuffer::alloc(ctx, 1)?,
            d_cycle: DeviceBuffer::alloc(ctx, 1)?,
            d_desc: DeviceBuffer::alloc(ctx, 52)?,
            desc: EntropyJobDesc::zeroed(),
        };
        world.upload(job)?;
        Ok(world)
    }

    /// Upload a (possibly new) flat job: pages/streams/payload/models/cycle
    /// are re-uploaded and the descriptor updated. Bounded observation jobs
    /// reuse one world across windows.
    pub fn upload(&mut self, job: &FlatEntropyJob) -> Result<()> {
        let ctx = self.ctx.clone();
        let upload = |buf: &DeviceBuffer, bytes: &[u8]| -> Result<()> {
            if bytes.is_empty() {
                return Ok(());
            }
            buf.upload(bytes)
        };
        let pages = DeviceBuffer::alloc(&ctx, job.pages.len().max(1) * 56)?;
        upload(&pages, pod_bytes(&job.pages))?;
        let streams = DeviceBuffer::alloc(&ctx, job.streams.len().max(1) * 24)?;
        upload(&streams, pod_bytes(&job.streams))?;
        let payload = DeviceBuffer::alloc(&ctx, job.payload.len().max(1))?;
        upload(&payload, &job.payload)?;
        let mut mvalue_bytes = Vec::with_capacity(job.model_values.len() * 2);
        for v in &job.model_values {
            mvalue_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mvalues = DeviceBuffer::alloc(&ctx, mvalue_bytes.len().max(2))?;
        upload(&mvalues, &mvalue_bytes)?;
        let mstarts = DeviceBuffer::alloc(&ctx, job.model_starts.len().max(1) * 4)?;
        upload(&mstarts, pod_bytes(&job.model_starts))?;
        let mfreqs = DeviceBuffer::alloc(&ctx, job.model_freqs.len().max(1) * 4)?;
        upload(&mfreqs, pod_bytes(&job.model_freqs))?;
        let mranges = DeviceBuffer::alloc(&ctx, job.model_ranges.len().max(1) * 4)?;
        upload(&mranges, pod_bytes(&job.model_ranges))?;
        let cycle = DeviceBuffer::alloc(&ctx, job.cycle.len().max(1) * 4)?;
        upload(&cycle, pod_bytes(&job.cycle))?;
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
        let d_desc = DeviceBuffer::alloc(&ctx, 52)?;
        upload(&d_desc, pod_bytes(&[desc]))?;
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

    /// Device pointer of the decode output arena start (for `decode_into`
    /// callers targeting the registered endpoint region).
    pub fn desc(&self) -> &EntropyJobDesc {
        &self.desc
    }

    fn launch(&self, out_dev: u64) -> Result<()> {
        let desc = &self.desc;
        let scratch_len = desc.page_count as usize * desc.scratch_stride as usize;
        let d_scratch = DeviceBuffer::alloc(&self.ctx, scratch_len.max(1))?;
        let d_status = DeviceBuffer::alloc(&self.ctx, desc.page_count as usize)?;
        let params = vec![
            self.d_desc.device_ptr(),
            self.d_pages.device_ptr(),
            self.d_streams.device_ptr(),
            self.d_payload.device_ptr(),
            self.d_mvalues.device_ptr(),
            self.d_mstarts.device_ptr(),
            self.d_mfreqs.device_ptr(),
            self.d_mranges.device_ptr(),
            self.d_cycle.device_ptr(),
            out_dev,
            d_scratch.device_ptr(),
            d_status.device_ptr(),
        ];
        let grid_blocks = desc.page_count.div_ceil(BLOCK_THREADS).max(1);
        self.function.launch(
            (grid_blocks, 1, 1),
            (BLOCK_THREADS, 1, 1),
            self.stream.handle,
            &params,
        )?;
        self.stream.synchronize()?;
        let mut status = vec![0u8; desc.page_count as usize];
        d_status.download_prefix(&mut status)?;
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
        let out = DeviceBuffer::alloc(&self.ctx, self.desc.arena_samples as usize * 4)?;
        self.launch(out.device_ptr())?;
        let mut bytes = vec![0u8; self.desc.arena_samples as usize * 4];
        out.download_prefix(&mut bytes)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::entropy_shared::FlatPage;

    #[test]
    fn pod_sizes_match_device_records() {
        let page = FlatPage {
            frames: 512,
            channels: 1,
            sym: 1,
            kind: 0,
            mode: 0,
            _pad: [0; 3],
            stream_off: 0,
            stream_count: 0,
            out_off: 0,
            payload_off: 0,
            payload_len: 8,
            hyp_kind: 0,
            _pad2: [0; 3],
            hyp_level: 0,
            cycle_off: 0,
            cycle_len: 0,
            hyp_phase: 0,
            scratch_off: 0,
        };
        let job = FlatEntropyJob {
            pages: vec![page],
            arena_samples: 512,
            max_page_scratch: 8,
            channels: 1,
            ..Default::default()
        };
        assert_eq!(pod_bytes(&job.pages).len(), 56);
        assert_eq!(pod_bytes(&job.streams).len(), 0);
        assert_eq!(core::mem::size_of::<EntropyJobDesc>(), 52);
    }
}

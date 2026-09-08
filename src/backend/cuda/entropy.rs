//! CUDA entropy decode host runtime (Phase H.2) — driver API.
//!
//! Loads the same PTX module as the render kernel and drives
//! `vole_entropy_decode` (one thread per entropy page). The entropy object
//! (payloads/models/page records) is uploaded once and stays GPU-resident;
//! a decode job covers a bounded page range (an observation window), and the
//! decoded samples land in a GPU output arena — never a full-object decoded
//! waveform (H.2.17/H.2.18).
//!
//! Exactness: the kernel runs the same `device::entropy_shared` functions as
//! the host flat decoder on the identical job; the court compares the
//! downloaded arena byte-for-byte with the host decode.

use super::driver::{Cuda, DeviceBuffer, Function, Module, Stream};
use super::ffi::Fns;
use crate::backend::entropy_flat::FlatEntropyJob;
use crate::device::entropy_shared::EntropyJobDesc;
use crate::error::{Error, Result};

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
/// Does not own the CUDA session: it borrows an existing context (the
/// entropy-d1 fused path shares the render world's context). The caller must
/// keep the session alive longer than this world (drop order: entropy world
/// first).
pub struct EntropyWorld {
    /// Driver function table (copy; the owning session stays alive via the
    /// caller).
    fns: Fns,
    /// Owning session when this world opened its own context (None when
    /// sharing a caller session).
    session: Option<Cuda>,
    _module: Module,
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

    /// Build a decode world over an existing CUDA session (the session must
    /// outlive this world).
    pub fn open_shared(
        cuda: &Cuda,
        ptx_bytes: &[u8],
        job: &FlatEntropyJob,
    ) -> Result<EntropyWorld> {
        let fns = cuda.fns;
        let module = cuda.load_module(ptx_bytes)?;
        let function = module.function(ENTROPY_KERNEL_ENTRY)?;
        let stream = cuda.create_stream()?;
        let upload = |buf: &DeviceBuffer, bytes: &[u8]| -> Result<()> {
            if bytes.is_empty() {
                return Ok(());
            }
            buf.upload(bytes)
        };

        let d_pages = DeviceBuffer::alloc(&fns, job.pages.len().max(1) * 56)?;
        upload(&d_pages, pod_bytes(&job.pages))?;
        let d_streams = DeviceBuffer::alloc(&fns, job.streams.len().max(1) * 24)?;
        upload(&d_streams, pod_bytes(&job.streams))?;
        let d_payload = DeviceBuffer::alloc(&fns, job.payload.len().max(1))?;
        upload(&d_payload, &job.payload)?;

        let mut mvalue_bytes = Vec::with_capacity(job.model_values.len() * 2);
        for v in &job.model_values {
            mvalue_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let d_mvalues = DeviceBuffer::alloc(&fns, mvalue_bytes.len().max(2))?;
        upload(&d_mvalues, &mvalue_bytes)?;
        let d_mstarts = DeviceBuffer::alloc(&fns, job.model_starts.len().max(1) * 4)?;
        upload(&d_mstarts, pod_bytes(&job.model_starts))?;
        let d_mfreqs = DeviceBuffer::alloc(&fns, job.model_freqs.len().max(1) * 4)?;
        upload(&d_mfreqs, pod_bytes(&job.model_freqs))?;
        let d_mranges = DeviceBuffer::alloc(&fns, job.model_ranges.len().max(1) * 4)?;
        upload(&d_mranges, pod_bytes(&job.model_ranges))?;
        let d_cycle = DeviceBuffer::alloc(&fns, job.cycle.len().max(1) * 4)?;
        upload(&d_cycle, pod_bytes(&job.cycle))?;

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
        let d_desc = DeviceBuffer::alloc(&fns, 52)?;
        upload(&d_desc, pod_bytes(&[desc]))?;

        Ok(EntropyWorld {
            fns,
            session: None,
            _module: module,
            function,
            stream,
            d_pages,
            d_streams,
            d_payload,
            d_mvalues,
            d_mstarts,
            d_mfreqs,
            d_mranges,
            d_cycle,
            d_desc,
            desc,
        })
    }

    /// Decode the uploaded job (one thread per page). Returns the decoded
    /// i32 sample arena after synchronization. A kernel status 0 on any page
    /// is an internal inconsistency (host validation precedes upload).
    pub fn decode(&self) -> Result<Vec<i32>> {
        let desc = &self.desc;
        let d_out = DeviceBuffer::alloc(&self.fns, desc.arena_samples as usize * 4)?;
        let scratch_len = desc.page_count as usize * desc.scratch_stride as usize;
        let d_scratch = DeviceBuffer::alloc(&self.fns, scratch_len.max(1))?;
        let d_status = DeviceBuffer::alloc(&self.fns, desc.page_count as usize)?;
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
            d_out.device_ptr(),
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
        let mut bytes = vec![0u8; desc.arena_samples as usize * 4];
        d_out.download_prefix(&mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect())
    }

    /// Launch the decode kernel writing directly into a caller-provided
    /// device pointer (used by the fused entropy->D1 endpoint path where the
    /// arena is the registered endpoint region or a bounded device arena).
    pub fn decode_into(&self, out_dev: u64, out_samples: u32) -> Result<()> {
        let desc = &self.desc;
        if out_samples < desc.arena_samples {
            return Err(Error::limit("decode_into arena smaller than job arena"));
        }
        let scratch_len = desc.page_count as usize * desc.scratch_stride as usize;
        let d_scratch = DeviceBuffer::alloc(&self.fns, scratch_len.max(1))?;
        let d_status = DeviceBuffer::alloc(&self.fns, desc.page_count as usize)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::entropy_shared::FlatPage;

    #[test]
    fn descriptor_bytes_are_stable() {
        // A tiny job descriptor round-trips through the pod encoding at the
        // exact 52-byte size the device expects.
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
        let mut job = FlatEntropyJob {
            pages: vec![page],
            arena_samples: 512,
            max_page_scratch: 8,
            channels: 1,
            ..Default::default()
        };
        let _ = job.scratch_stride();
        let mut desc = EntropyJobDesc::zeroed();
        desc.page_count = job.pages.len() as u32;
        desc.arena_samples = job.arena_samples as u32;
        desc.scratch_stride = job.scratch_stride() as u32;
        desc.statuses_len = job.pages.len() as u32;
        let desc_arr = [desc];
        let bytes = pod_bytes(&desc_arr);
        assert_eq!(bytes.len(), 52);
        // Page pod size matches the device record.
        assert_eq!(pod_bytes(&job.pages).len(), 56);
        job.pages.clear();
        let _ = job;
    }
}

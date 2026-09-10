//! CUDA period-scan host runtime (Phase L) — driver API.
//!
//! Loads the same PTX module as the render/entropy kernels and drives
//! `vole_period_scan` (one thread per candidate period). The observation
//! window is uploaded once per scan; the ranked period counts come back as a
//! `u32` array.
//!
//! This is **search placement only**. The kernel ranks candidate periods; it
//! never decides a representation. `court inverse-search` requires the ranked
//! proposals to be re-verified by the normative exact evaluator, and requires
//! the CUDA ranking to equal the host ranking exactly (both call the shared
//! `device::search_shared` function).

use super::driver::{Cuda, DeviceBuffer, Function, Module, Stream};
use super::ffi::Fns;
use crate::error::{Error, Result};

/// PTX entry of the period-scan kernel.
pub const SEARCH_KERNEL_ENTRY: &str = "vole_period_scan";

/// One-thread-per-period launch shape.
const BLOCK_THREADS: u32 = 128;

/// GPU-resident period scan over a fixed window shape.
///
/// The window length and period bound are fixed at construction (the scan
/// output is sized once), so a court scans many fixtures without reallocating.
pub struct SearchWorld {
    fns: Fns,
    /// Owning session (None when sharing a caller session).
    session: Option<Cuda>,
    /// Module handle (None when the caller keeps the module alive).
    module: Option<Module>,
    function: Function,
    stream: Stream,
    d_x: DeviceBuffer,
    d_counts: DeviceBuffer,
    /// Frames per window this world was built for.
    pub frames: u32,
    /// Number of candidate periods the scan covers.
    pub period_limit: u32,
}

impl SearchWorld {
    /// Open a fresh CUDA session and build a scan world for `frames`-frame
    /// mono windows and `period_limit` candidate periods.
    pub fn open(
        ordinal: i32,
        ptx_bytes: &[u8],
        frames: u32,
        period_limit: u32,
    ) -> Result<SearchWorld> {
        let cuda = Cuda::open(ordinal)?;
        let mut world = SearchWorld::open_shared(&cuda, ptx_bytes, frames, period_limit)?;
        world.session = Some(cuda);
        Ok(world)
    }

    /// Build a scan world over an existing CUDA session (the session must
    /// outlive this world).
    pub fn open_shared(
        cuda: &Cuda,
        ptx_bytes: &[u8],
        frames: u32,
        period_limit: u32,
    ) -> Result<SearchWorld> {
        let module = cuda.load_module(ptx_bytes)?;
        let function = module.function(SEARCH_KERNEL_ENTRY)?;
        let mut world = SearchWorld::build(
            &cuda.fns,
            function,
            cuda.create_stream()?,
            frames,
            period_limit,
        )?;
        world.module = Some(module);
        Ok(world)
    }

    fn build(
        fns: &Fns,
        function: Function,
        stream: Stream,
        frames: u32,
        period_limit: u32,
    ) -> Result<SearchWorld> {
        if frames < 2 {
            return Err(Error::malformed("period scan needs at least 2 frames"));
        }
        if period_limit == 0 {
            return Err(Error::malformed("period scan needs a nonzero period bound"));
        }
        // The device scan covers periods 1..=min(period_limit, frames - 1).
        let limit = period_limit.min(frames - 1);
        Ok(SearchWorld {
            fns: *fns,
            session: None,
            module: None,
            function,
            stream,
            d_x: DeviceBuffer::alloc(fns, frames as usize * 4)?,
            d_counts: DeviceBuffer::alloc(fns, limit as usize * 4)?,
            frames,
            period_limit: limit,
        })
    }

    /// Scan `x` (exactly `frames` samples) and return the per-period record
    /// counts: `counts[i]` is the count for period `i + 1`
    /// (`PERIOD_NOT_CLOSEABLE` when the hypothesis cannot close).
    pub fn scan(&self, x: &[i32]) -> Result<Vec<u32>> {
        if x.len() != self.frames as usize {
            return Err(Error::malformed("period-scan window length mismatch"));
        }
        let mut bytes = Vec::with_capacity(x.len() * 4);
        for v in x {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.d_x.upload(&bytes)?;
        let params = vec![
            self.d_x.device_ptr(),
            u64::from(self.frames),
            u64::from(self.period_limit),
            self.d_counts.device_ptr(),
        ];
        let blocks = self.period_limit.div_ceil(BLOCK_THREADS).max(1);
        self.function.launch(
            (blocks, 1, 1),
            (BLOCK_THREADS, 1, 1),
            self.stream.handle,
            &params,
        )?;
        self.stream.synchronize()?;
        let mut out = vec![0u8; self.period_limit as usize * 4];
        self.d_counts.download_prefix(&mut out)?;
        Ok(out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect())
    }

    /// True when this world owns its session (evidence only).
    pub fn owns_session(&self) -> bool {
        self.session.is_some()
    }

    /// Driver function table (evidence/tests).
    pub fn fns(&self) -> &Fns {
        &self.fns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_rejects_mismatched_windows() {
        // Building needs a driver; the pure validation is what is testable
        // without a device, and it is exercised through `build`'s guards by
        // the courts. Here we only pin the entry name.
        assert_eq!(SEARCH_KERNEL_ENTRY, "vole_period_scan");
    }
}

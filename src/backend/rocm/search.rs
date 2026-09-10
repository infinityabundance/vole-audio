//! ROCm period-scan host runtime (Phase L) — the AMD mirror of
//! `backend::cuda::search`.
//!
//! Drives `vole_period_scan` from the same AMDGPU code object as the render/
//! entropy/upmix entries. Launch geometry follows the frozen `device::geom`
//! contract: one `Grid` is computed once and used BOTH for the launch and for
//! the `blocks_x`/`threads_x` kernel parameters, so the host cannot violate
//! the AMD geometry contract.
//!
//! The kernel ranks candidate periods; it never decides a representation.
//! `court inverse-search` re-verifies every ranked proposal with the normative
//! exact evaluator and requires the ROCm ranking to equal the host ranking
//! (both call the shared `device::search_shared` function). On hosts without
//! an AMD device the court records a typed negative and no launch is attempted.

use crate::backend::rocm::ffi;
use crate::backend::rocm::runtime::{Arg, DeviceBuffer, Function, HipDevice, Module, Rocm};
use crate::device::geom::Grid;
use crate::error::{Error, Result};
use std::sync::Arc;

/// Entry name of the period-scan kernel in the AMDGPU code object.
pub const SEARCH_ENTRY: &str = "vole_period_scan";

/// Threads per workgroup for the period scan (one workitem per period).
pub const SEARCH_THREADS: u32 = 128;

/// Grid-stride geometry for `total` work items at `threads` per workgroup.
pub fn grid_for(total: usize, threads: u32) -> Grid {
    Grid::new(total.div_ceil(threads as usize).max(1) as u32, threads)
}

/// GPU-resident period scan over one fixed window shape (AMD surface).
pub struct SearchWorldRocm {
    /// Owned so the module (and its function handle) outlives every launch.
    module: Module,
    function: Function,
    d_x: DeviceBuffer,
    d_counts: DeviceBuffer,
    /// Frames per window this world was built for.
    pub frames: u32,
    /// Number of candidate periods the scan covers.
    pub period_limit: u32,
    /// The ROCm session (owns the device affinity).
    pub rocm: Rocm,
}

impl SearchWorldRocm {
    /// Open the HIP session, load the code object, and prepare the scan
    /// buffers for `frames`-frame windows over `period_limit` periods.
    pub fn open(
        ordinal: i32,
        artifact_bytes: &[u8],
        frames: u32,
        period_limit: u32,
    ) -> Result<SearchWorldRocm> {
        if frames < 2 {
            return Err(Error::malformed("period scan needs at least 2 frames"));
        }
        if period_limit == 0 {
            return Err(Error::malformed("period scan needs a nonzero period bound"));
        }
        let rocm = Rocm::open(ordinal)?;
        let module = rocm.load_module(artifact_bytes)?;
        let function = module.function(SEARCH_ENTRY)?;
        let device: &Arc<HipDevice> = &rocm.device;
        let limit = period_limit.min(frames - 1);
        let d_x = DeviceBuffer::alloc(device, frames as usize * 4)?;
        let d_counts = DeviceBuffer::alloc(device, limit as usize * 4)?;
        Ok(SearchWorldRocm {
            module,
            function,
            d_x,
            d_counts,
            frames,
            period_limit: limit,
            rocm,
        })
    }

    /// Scan `x` (exactly `frames` samples) and return the per-period record
    /// counts (`counts[i]` is the count for period `i + 1`).
    pub fn scan(&self, x: &[i32]) -> Result<Vec<u32>> {
        if x.len() != self.frames as usize {
            return Err(Error::malformed("period-scan window length mismatch"));
        }
        let mut bytes = Vec::with_capacity(x.len() * 4);
        for v in x {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.d_x.upload(&bytes)?;
        // One Grid feeds BOTH the launch and the kernel parameters.
        let grid = grid_for(self.period_limit as usize, SEARCH_THREADS);
        let args = [
            Arg::Ptr(self.d_x.device_ptr()),
            Arg::U32(self.frames),
            Arg::U32(self.period_limit),
            Arg::Ptr(self.d_counts.device_ptr()),
            Arg::U32(grid.blocks_x),
            Arg::U32(grid.threads_x),
        ];
        self.function.launch(grid, &args)?;
        self.rocm.synchronize()?;
        let mut out = vec![0u8; self.period_limit as usize * 4];
        self.d_counts.download(&mut out)?;
        Ok(out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect())
    }

    /// The module handle (evidence).
    pub fn module_handle(&self) -> ffi::hipModule_t {
        self.module.handle()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_covers_the_period_range() {
        assert_eq!(grid_for(512, 128), Grid::new(4, 128));
        assert_eq!(grid_for(1, 128), Grid::new(1, 128));
        assert_eq!(grid_for(129, 128), Grid::new(2, 128));
        assert!(grid_for(0, 128).valid());
    }

    #[test]
    fn entry_name_matches_the_device_export() {
        assert_eq!(SEARCH_ENTRY, "vole_period_scan");
    }
}

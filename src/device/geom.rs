//! Launch geometry for grid-stride device loops (no_std, shared).
//!
//! CUDA exposes block/grid dims as kernel builtins; AMDGCN does not expose
//! the workgroup count or size, so the AMD kernels take `blocks_x` /
//! `threads_x` as explicit kernel parameters (see `device::amdgcn_entry`).
//! This module is the single, host-tested definition of that geometry:
//!
//! * **frozen launch contract** — the Phase-J host must launch with
//!   `blocks_x > 0 && threads_x > 0`, a declared `threads_x` equal to the
//!   actual workgroup size, and a declared `blocks_x` equal to the actual
//!   number of workgroups;
//! * **defensive device behavior** — the kernels return immediately for
//!   zero/invalid geometry (a zero stride would otherwise make every
//!   grid-stride loop non-progressing: `g += 0` forever);
//! * **grid-stride coverage is a mathematical property** — for any valid
//!   geometry and any work count, every index in `0..total` is visited
//!   exactly once across the grid. The unit battery below tests the
//!   pathological cases the reviewer-specified contract names (1x1, 1x64,
//!   non-divisible counts, more blocks than work, page-count ±1, maximum
//!   geometry, and zero rejection) without needing a GPU.

/// Frozen grid-stride launch geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub blocks_x: u32,
    pub threads_x: u32,
}

impl Grid {
    pub const fn new(blocks_x: u32, threads_x: u32) -> Self {
        Self {
            blocks_x,
            threads_x,
        }
    }

    /// The launch contract: both dimensions must be nonzero. The device
    /// kernels return (do nothing) when this is false.
    #[inline]
    pub const fn valid(self) -> bool {
        self.blocks_x > 0 && self.threads_x > 0
    }

    /// Total stride per workitem: `blocks_x * threads_x`. The product of two
    /// `u32` fits `u64` (max `(2^32-1)^2 < 2^64`), so this never wraps.
    #[inline]
    pub const fn stride(self) -> u64 {
        (self.blocks_x as u64) * (self.threads_x as u64)
    }

    /// Global index of workitem `tid` in workgroup `bid`. `None` when the
    /// geometry is invalid or the ids are out of range (defensive).
    #[inline]
    pub fn first(self, bid: u32, tid: u32) -> Option<u64> {
        if self.valid() && bid < self.blocks_x && tid < self.threads_x {
            Some((bid as u64) * (self.threads_x as u64) + (tid as u64))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every index in `0..total` is visited exactly once across the full
    /// grid, in ascending order per workitem — the grid-stride invariant.
    fn assert_exact_coverage(g: Grid, total: u64) {
        assert!(g.valid());
        let mut seen = vec![false; total as usize];
        let mut count = 0u64;
        for bid in 0..g.blocks_x {
            for tid in 0..g.threads_x {
                let Some(mut idx) = g.first(bid, tid) else {
                    panic!("ids in range must map");
                };
                while idx < total {
                    assert!(!seen[idx as usize], "index {idx} visited twice");
                    seen[idx as usize] = true;
                    count += 1;
                    idx += g.stride();
                }
            }
        }
        assert_eq!(count, total, "coverage count mismatch");
        assert!(seen.iter().all(|s| *s), "some index never visited");
    }

    #[test]
    fn zero_geometry_is_rejected() {
        for g in [
            Grid::new(0, 0),
            Grid::new(1, 0),
            Grid::new(0, 64),
            Grid::new(0, u32::MAX),
            Grid::new(u32::MAX, 0),
        ] {
            assert!(!g.valid());
            assert_eq!(g.first(0, 0), None, "{g:?} must not map a workitem");
        }
    }

    #[test]
    fn out_of_range_ids_are_rejected() {
        let g = Grid::new(8, 64);
        assert_eq!(g.first(8, 0), None); // bid == blocks_x
        assert_eq!(g.first(0, 64), None); // tid == threads_x
        assert_eq!(g.first(u32::MAX, u32::MAX), None);
        assert_eq!(g.first(7, 63), Some(511)); // last workitem
    }

    #[test]
    fn one_by_one_geometry() {
        let g = Grid::new(1, 1);
        assert_eq!(g.first(0, 0), Some(0));
        assert_eq!(g.stride(), 1);
        assert_exact_coverage(g, 1);
        assert_exact_coverage(g, 4096);
    }

    #[test]
    fn single_workgroup_geometry() {
        for threads in [1u32, 2, 64, 256, 1024] {
            let g = Grid::new(1, threads);
            assert_exact_coverage(g, 0);
            assert_exact_coverage(g, 1);
            assert_exact_coverage(g, 511);
            assert_exact_coverage(g, 512);
            assert_exact_coverage(g, 513);
        }
    }

    #[test]
    fn non_divisible_counts_are_exact() {
        // sample counts that do not divide blocks*threads must still get
        // full, non-overlapping coverage.
        let g = Grid::new(4, 128); // stride 512
        for total in [1u64, 2, 100, 511, 512, 513, 1000, 4096, 8193] {
            assert_exact_coverage(g, total);
        }
    }

    #[test]
    fn more_blocks_than_work_is_exact() {
        assert_exact_coverage(Grid::new(1024, 64), 1); // one workitem's worth
        assert_exact_coverage(Grid::new(1024, 64), 63);
        assert_exact_coverage(Grid::new(64, 64), 4096);
        assert_exact_coverage(Grid::new(64, 64), 4097);
    }

    #[test]
    fn page_count_neighborhood_is_exact() {
        // The entropy kernel grid-stride loop bounds by page_count; test
        // pages-1 / pages / pages+1 at the court's page counts.
        for pages in [1u64, 15, 16, 17, 255, 256, 257] {
            let blocks = pages.div_ceil(64).max(1) as u32;
            assert_exact_coverage(Grid::new(blocks, 64), pages);
        }
    }

    #[test]
    fn maximum_permitted_geometry_does_not_overflow() {
        // (2^32-1)^2 fits u64; stride must not wrap and first() must stay in
        // range for valid ids.
        let g = Grid::new(u32::MAX, u32::MAX);
        assert!(g.valid());
        assert_eq!(g.stride(), (u64::from(u32::MAX)) * (u64::from(u32::MAX)));
        assert_eq!(g.first(u32::MAX - 1, u32::MAX - 1), Some(g.stride() - 1));
        // And a practical huge geometry is exact for a bounded work count.
        assert_exact_coverage(Grid::new(65_536, 1024), 67_108_864);
    }
}

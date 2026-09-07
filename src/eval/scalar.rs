//! Scalar reference evaluator — semantic authority.
//!
//! Every other execution surface (SIMD, CUDA, ROCm) must reproduce the scalar
//! oracle's observations *exactly* (hash equality). The scalar oracle is a
//! thin, readable wrapper over the frozen sampler semantics (`sampler::world`)
//! — intentionally no cleverness.

use crate::error::Result;
use crate::object::ObjectStore;
use crate::sampler::world::World;

/// Scalar reference evaluator.
#[derive(Debug, Clone)]
pub struct ScalarOracle {
    pub world: World,
}

impl ScalarOracle {
    pub fn new(world: World) -> ScalarOracle {
        ScalarOracle { world }
    }

    /// Observe `[start, start+frames)` in canonical interleaved i32 codes.
    pub fn observe(
        &self,
        store: &ObjectStore,
        start_frame: i64,
        frames: usize,
    ) -> Result<Vec<i32>> {
        self.world.observe(store, start_frame, frames)
    }

    /// Observe and return the canonical SHA-256 of the window.
    pub fn observe_hash(
        &self,
        store: &ObjectStore,
        start_frame: i64,
        frames: usize,
    ) -> Result<[u8; 32]> {
        let out = self.observe(store, start_frame, frames)?;
        Ok(crate::universe::observation::observation_sha256(&out))
    }
}

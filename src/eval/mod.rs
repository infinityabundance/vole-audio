//! Evaluator surfaces.
//!
//! `eval::common` defines the shared vocabulary (execution surfaces and
//! transform classification) and is `no_std`-clean. `eval::scalar` is the
//! scalar reference evaluator (host): it owns semantic authority and every
//! other backend is differentially tested against it. SIMD (`eval::simd`)
//! arrives in Phase F.

pub mod common;

#[cfg(feature = "std")]
pub mod scalar;

#[cfg(feature = "std")]
pub use scalar::ScalarOracle;

//! Evaluator surfaces.
//!
//! `eval::common` defines the shared vocabulary (execution surfaces and
//! transform classification) and is `no_std`-clean. `eval::scalar` is the
//! scalar reference evaluator (host): it owns semantic authority and every
//! other backend is differentially tested against it. `eval::backend`
//! (Phase F) provides backend selection + runtime ISA dispatch; `eval::simd`
//! (Phase F) is the planned host engine with vector kernels (`eval::x86`,
//! x86-64 only) that reproduce the scalar oracle's observations bit-for-bit.

pub mod common;

#[cfg(feature = "std")]
pub mod backend;
#[cfg(feature = "std")]
pub mod battery;
#[cfg(feature = "std")]
pub mod learned_simd;
#[cfg(feature = "std")]
pub mod scalar;
#[cfg(feature = "std")]
pub mod simd;

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(crate) mod x86;
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(crate) mod x86_ops;

#[cfg(feature = "std")]
pub use backend::{Backend, ConcreteBackend, Isa, resolve};
#[cfg(feature = "std")]
pub use scalar::ScalarOracle;
#[cfg(feature = "std")]
pub use simd::SimdOracle;

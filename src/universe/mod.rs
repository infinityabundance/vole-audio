//! `vole.audio.u1` — the first frozen semantic universe.
//!
//! Everything in this module is `no_std`-clean, deterministic, and compiled
//! unchanged for scalar/SIMD/GPU surfaces. The normative freeze lives in
//! `docs/U1_SPEC.md`; this module is its executable form. Changing semantics
//! requires a profile change plus fresh reference vectors — never a silent
//! edit.

pub mod arithmetic;
pub mod clock;
pub mod event;
pub mod layout;
pub mod observation;
pub mod phase;
pub mod prng;
pub mod sample;
pub mod time;
pub mod u1;

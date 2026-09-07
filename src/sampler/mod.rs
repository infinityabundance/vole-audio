//! Sampler semantics: voices, envelopes, gain/pan, rate/loop mapping,
//! interpolation, resampling, mixing, and the narrow exact filter set.
//!
//! The pure semantic modules (`envelope`, `gain`, `pan`, `rate`,
//! `interpolation`, `resampler`, `mix`, `filter`) are `no_std`-clean and are
//! reused unchanged by GPU device code. The host orchestration modules
//! (`voice`, `world`, `scheduler`) are `std`-gated.

pub mod envelope;
pub mod filter;
pub mod gain;
pub mod interpolation;
pub mod pan;
pub mod procedural;
pub mod rate;
pub mod resampler;

#[cfg(feature = "std")]
pub mod mix;
#[cfg(feature = "std")]
pub mod scheduler;
#[cfg(feature = "std")]
pub mod voice;
#[cfg(feature = "std")]
pub mod world;

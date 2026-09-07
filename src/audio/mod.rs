//! Audio endpoint courts: directness classes, topology, ALSA endpoint work.
//!
//! Phase A provides the path-class and topology vocabularies (pure data).
//! The ALSA endpoint implementation, mmap discipline, and D1/D2 courts land
//! in later phases inside this module.
//!
//! This module is host-only.

pub mod directness;
pub mod topology;

//! Audio endpoint courts: directness classes, topology, ALSA endpoint work.
//!
//! Phase A provides the path-class and topology vocabularies (pure data).
//! Phase H adds the ALSA `hw:` direct-mmap endpoint support and the D1 court
//! (`alsa`/`alsa_ffi`, Linux only).
//!
//! This module is host-only.

pub mod directness;
pub mod topology;

/// ALSA endpoint support (Linux only): audited libasound FFI + `hw:`
/// direct-mmap configuration/discipline. Compiled only on Linux; other hosts
/// simply lack the module (courts report it as unavailable).
#[cfg(target_os = "linux")]
pub mod alsa;
#[cfg(target_os = "linux")]
pub mod alsa_ffi;

//! Executable courts.
//!
//! Courts are the empirical program of the paper (paper §"Courts"): each
//! `court <name>` run produces a human summary, a machine receipt, and raw
//! traces where relevant. Courts never invent results; unsupported paths
//! report `UNSUPPORTED_BY_*` / `INCONCLUSIVE` and remain visible.
//!
//! Court registry grows by phase: semantic (Phase C), authored (Phase D),
//! simd parity (Phase F), facts (semantic independent-oracle coverage, Phase G
//! prerequisite), inverse/flattening (Phase K), cuda/rocm/d1/d2 (GPU
//! phases), depth/conventional/random-access/negative/interference (Phase M),
//! all.

#[cfg(feature = "std")]
pub mod authored;
#[cfg(feature = "std")]
pub mod facts;
#[cfg(feature = "std")]
pub mod semantic;
#[cfg(feature = "std")]
pub mod simd;

use crate::status::Verdict;
use std::path::Path;

/// Court registry: name -> human description.
pub const COURT_NAMES: &[(&str, &str)] = &[
    (
        "semantic",
        "scalar oracle determinism battery: repeated-hash, chunked==contiguous, seek==sequential",
    ),
    (
        "authored",
        "procedural SampleObject battery: deterministic observation without resident full-object PCM",
    ),
    (
        "simd",
        "Phase F SIMD parity: scalar == SIMD on every ISA floor + fixture-level timing",
    ),
    (
        "facts",
        "independent semantic facts: first-principles oracles for every representation/\
transform on every host surface",
    ),
];

/// Run `court <name>`; unknown courts are usage errors.
pub fn run(name: &str, receipts_root: &Path) -> crate::error::Result<Verdict> {
    match name {
        "semantic" => semantic::run(receipts_root),
        "authored" => authored::run(receipts_root),
        "simd" => simd::run(receipts_root),
        "facts" => facts::run(receipts_root),
        other => Err(crate::error::Error::malformed(format!(
            "unknown court '{other}' (available: {})",
            COURT_NAMES
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

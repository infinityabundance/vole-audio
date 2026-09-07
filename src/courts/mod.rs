//! Executable courts.
//!
//! Courts are the empirical program of the paper (paper §"Courts"): each
//! `court <name>` run produces a human summary, a machine receipt, and raw
//! traces where relevant. Courts never invent results; unsupported paths
//! report `UNSUPPORTED_BY_*` / `INCONCLUSIVE` and remain visible.
//!
//! Court registry grows by phase: semantic (Phase C), authored/inverse/
//! flattening (later), cuda/rocm/d1/d2 (GPU phases), depth/conventional/
//! random-access/negative/interference (Phase M), all.

#[cfg(feature = "std")]
pub mod semantic;

use crate::status::Verdict;
use std::path::Path;

/// Court registry: name -> human description.
pub const COURT_NAMES: &[(&str, &str)] = &[(
    "semantic",
    "scalar oracle determinism battery: repeated-hash, chunked==contiguous, seek==sequential",
)];

/// Run `court <name>`; unknown courts are usage errors.
pub fn run(name: &str, receipts_root: &Path) -> crate::error::Result<Verdict> {
    match name {
        "semantic" => semantic::run(receipts_root),
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

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
pub mod cuda;
#[cfg(feature = "std")]
pub mod d1;
#[cfg(feature = "std")]
pub mod entropy_common;
#[cfg(feature = "std")]
pub mod entropy_literal;
#[cfg(feature = "std")]
pub mod entropy_pages;
#[cfg(feature = "std")]
pub mod entropy_partial;
#[cfg(feature = "std")]
pub mod entropy_rans;
#[cfg(feature = "std")]
pub mod entropy_residual;
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
    (
        "cuda",
        "Phase G CUDA D0: scalar == CUDA parity on frozen fixtures + facts on the device\
surface + strategy comparison + fixture-level throughput",
    ),
    (
        "d1",
        "Phase H CUDA D1 falsification: register the actual ALSA mmap endpoint region with \
cuMemHostRegister(DEVICEMAP), render final codes directly into it, and measure the bytes D1 \
removes vs the D0-mmap baseline (default silence-safe; --emit-audio opt-in)",
    ),
    (
        "entropy-rans",
        "Phase H.2 rANS battery: canonical determinism, model sweep, hostile corpus (typed or clean)",
    ),
    (
        "entropy-literal",
        "Phase H.2 literal entropy floor: RAW vs native rANS vs canonical U1 literal vs FLAC baseline",
    ),
    (
        "entropy-residual",
        "Phase H.2 exact-residual entropy coding: byte-identical reconstruction, complete costs",
    ),
    (
        "entropy-pages",
        "Phase H.2 page-size Pareto: sizes 64..4096, seek latency, corruption locality",
    ),
    (
        "entropy-partial",
        "Phase H.2 partial materialization: partial == full slice, pages touched, decode halo",
    ),
];

/// Run `court <name>`; unknown courts are usage errors.
pub fn run(name: &str, receipts_root: &Path) -> crate::error::Result<Verdict> {
    match name {
        "semantic" => semantic::run(receipts_root),
        "authored" => authored::run(receipts_root),
        "simd" => simd::run(receipts_root),
        "facts" => facts::run(receipts_root),
        "cuda" => cuda::run(receipts_root),
        "d1" => d1::run(receipts_root),
        "entropy-rans" => entropy_rans::run(receipts_root),
        "entropy-literal" => entropy_literal::run(receipts_root),
        "entropy-residual" => entropy_residual::run(receipts_root),
        "entropy-pages" => entropy_pages::run(receipts_root),
        "entropy-partial" => entropy_partial::run(receipts_root),
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

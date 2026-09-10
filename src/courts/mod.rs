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
pub mod conventional;
#[cfg(feature = "std")]
pub mod corpus;
#[cfg(feature = "std")]
pub mod cuda;
#[cfg(feature = "std")]
pub mod d1;
#[cfg(feature = "std")]
pub mod dsfb_entropy;
#[cfg(feature = "std")]
pub mod entropy_common;
pub mod entropy_cuda;
#[cfg(feature = "std")]
pub mod entropy_d1;
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
pub mod entropy_simd;
#[cfg(feature = "std")]
pub mod entropyfs;
#[cfg(feature = "std")]
pub mod facts;
#[cfg(feature = "std")]
pub mod flagship;
#[cfg(feature = "std")]
pub mod flattening;
#[cfg(feature = "std")]
pub mod fullobj;
#[cfg(feature = "std")]
pub mod h2;
#[cfg(feature = "std")]
pub mod inverse;
#[cfg(feature = "std")]
pub mod inverse_search;
#[cfg(feature = "std")]
pub mod rocm;
#[cfg(feature = "std")]
pub mod rocm_d0;
#[cfg(feature = "std")]
pub mod rocm_d1;
#[cfg(feature = "std")]
pub mod runtime;
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
        "inverse",
        "Phase K bounded inverse compiler: deterministic proposals (literal, silence, \
constant, exact-repeat, residual zero/constant/periodic, shared reference) accepted only \
when they reproduce the window exactly through both the intrinsic closure and the scalar \
evaluator; complete dependency accounting and a deterministic Pareto frontier",
    ),
    (
        "flattening",
        "Phase K host flat-evaluator parity: flat == scalar bit-for-bit over the frozen \
fixtures and an adversarial battery, with honest residual-closure materialization and \
upload accounting",
    ),
    (
        "inverse-search",
        "Phase L search placement: the bounded period scan on scalar / host-parallel / \
CUDA / ROCm surfaces must produce identical rankings, and the device-ranked proposals \
must produce exactly the sequential candidate set re-verified by the exact evaluator",
    ),
    (
        "conventional",
        "Phase M conventional baselines (contract §47): B0 raw PCM vs B1 in-process pure-Rust \
FLAC at 32 bits/sample (level 5 primary, 0/8 controls) over the exact canonical i32 domain, \
with the full B0-B9 ladder visible and every FLAC row required to round-trip exactly",
    ),
    (
        "corpus",
        "Phase M flagship-corpus gate: the frozen manifest's identity (schema, corpus hash, \
class assignments, rates, sizes, canonical i32 hashes) must regenerate exactly and agree \
with the frozen membership; missing/extra/mutated/mis-sized/wrong-rate/wrong-hash all fail",
    ),
    (
        "fullobj",
        "Phase M full-object archival container mechanism: frozen 65,536-frame segmentation, \
exact U1 segments, real serialized bytes, semantics-preserving reconstruction, boundary \
observation and hostile-container rejection over non-flagship fixtures",
    ),
    (
        "flagship",
        "Phase M flagship result: B1 FLAC versus current bounded VOLE inverse selection over \
the frozen 115-object corpus (110 B1-comparable), with per-object container bytes, \
selected representations and comparison buckets",
    ),
    (
        "runtime",
        "Phase M runtime substrate and measurement: B2 resident PCM, B3 raw PCM disk \
(cold/warm verified each repeat), B4 the exact B1 FLAC-5 artifact preloaded, B5 bounded VOLE \
(first-play and prepared control), over the frozen sequential 512-frame trace repeated with \
rotated source order; harness-owned latency, split storage/residency accounting, two explicit \
populations, and a stratified crossover surface",
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
    (
        "entropy-cuda",
        "Phase H.2 CUDA entropy decode: scalar == CUDA on literal/RAW-fallback/residual closure \
jobs (one thread per page; artifact scripts/out/vole_audio.ptx)",
    ),
    (
        "entropy-simd",
        "Phase H.2 CPU parallel decode surface: scalar == page-parallel decode (exact); \
instruction-SIMD decode honestly NOT_IMPLEMENTED (H.2.16: no fabricated vectorization)",
    ),
    (
        "entropy-d1",
        "Phase H.2 flagship fused entropy -> CUDA -> D1 endpoint: per-window bounded page \
decode writes exact final S32 codes directly into the registered ALSA ring for literal, \
procedural mono+residual (device mono->stereo) and a high-entropy control, beside an \
equal-work D0 entropy baseline (default silence-safe; VOLE_ENTROPY_D1_EMIT_AUDIO=1 opt-in)",
    ),
    (
        "entropyfs",
        "Phase H.2 optional persistence: embedded == EntropyFS adapter roundtrip, integrity \
reverify, declared/unique/physical accounting, exact shared model once physically \
(feature entropyfs-store; INCONCLUSIVE + limitation without it)",
    ),
    (
        "dsfb-entropy",
        "Phase H.2 zero-authority search governance: exhaustive vs fixed-heuristic vs \
DSFB-guided over ONE frozen candidate universe; N and J per strategy (feature dsfb; \
INCONCLUSIVE + limitation without it)",
    ),
    (
        "h2",
        "aggregate Phase H.2 seal: runs every H.2 court in sequence; SUPPORTED only when \
all sub-courts are SUPPORTED (--all-features on supported hardware)",
    ),
    (
        "rocm",
        "Phase I ROCm evidence: AMD/ROCm presence probe + amdgcn code-object artifact state; \
        no kernel is executed (hardware-unavailable evidence; the differential device battery is \
        Phase J on ROCm hardware)",
    ),
    (
        "rocm-d0",
        "Phase J differential battery: scalar == ROCm on frozen fixture windows, entropy \
        decode jobs, and the mono->stereo upmix transform (D0; executes only on a D0-ready \
        ROCm runtime + AMD device)",
    ),
    (
        "rocm-d1",
        "Phase J D1 endpoint experiment: register the actual ALSA mmap endpoint region with \
        hipHostRegister(hipHostRegisterMapped), render final codes directly into it (incl. \
        the device mono->stereo expansion), and measure the bytes D1 removes vs the D0 \
        baseline (default silence-safe)",
    ),
];

/// Run `court <name>`; unknown courts are usage errors.
pub fn run(name: &str, receipts_root: &Path) -> crate::error::Result<Verdict> {
    match name {
        "semantic" => semantic::run(receipts_root),
        "authored" => authored::run(receipts_root),
        "simd" => simd::run(receipts_root),
        "facts" => facts::run(receipts_root),
        "inverse" => inverse::run(receipts_root),
        "inverse-search" => inverse_search::run(receipts_root),
        "conventional" => conventional::run(receipts_root),
        "corpus" => corpus::run(receipts_root),
        "fullobj" => fullobj::run(receipts_root),
        "flagship" => flagship::run(receipts_root),
        "runtime" => runtime::run(receipts_root),
        "flattening" => flattening::run(receipts_root),
        "cuda" => cuda::run(receipts_root),
        "d1" => d1::run(receipts_root),
        "entropy-rans" => entropy_rans::run(receipts_root),
        "entropy-literal" => entropy_literal::run(receipts_root),
        "entropy-residual" => entropy_residual::run(receipts_root),
        "entropy-pages" => entropy_pages::run(receipts_root),
        "entropy-partial" => entropy_partial::run(receipts_root),
        "entropy-cuda" => entropy_cuda::run(receipts_root),
        "entropy-simd" => entropy_simd::run(receipts_root),
        "entropy-d1" => entropy_d1::run(receipts_root),
        "entropyfs" => entropyfs::run(receipts_root),
        "dsfb-entropy" => dsfb_entropy::run(receipts_root),
        "h2" => h2::run(receipts_root),
        "rocm" => rocm::run(receipts_root),
        "rocm-d0" => rocm_d0::run(receipts_root),
        "rocm-d1" => rocm_d1::run(receipts_root),
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

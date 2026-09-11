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
pub mod all;
#[cfg(feature = "std")]
pub mod archive;
#[cfg(feature = "std")]
pub mod authored;
#[cfg(feature = "std")]
pub mod compound;
pub mod conventional;
#[cfg(feature = "std")]
pub mod corpus;
#[cfg(feature = "std")]
pub mod cuda;
#[cfg(feature = "std")]
pub mod d1;
#[cfg(feature = "std")]
pub mod depth;
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
pub mod interference;
#[cfg(feature = "std")]
pub mod inverse;
#[cfg(feature = "std")]
pub mod inverse_search;
#[cfg(feature = "std")]
pub mod learned;
#[cfg(feature = "std")]
pub mod learned_capacity;
#[cfg(feature = "std")]
pub(crate) mod learned_common;
#[cfg(feature = "std")]
pub mod learned_determinism;
#[cfg(feature = "std")]
pub mod learned_exp2_baseline;
#[cfg(feature = "std")]
pub mod learned_exp2_mechanisms;
#[cfg(feature = "std")]
pub mod learned_exp2_real_corpus;
#[cfg(feature = "std")]
pub mod learned_exp2_transfer;
#[cfg(feature = "std")]
pub mod learned_gpu;
#[cfg(feature = "std")]
pub mod learned_intrinsic;
#[cfg(feature = "std")]
pub mod learned_inverse;
#[cfg(feature = "std")]
pub mod learned_linear;
#[cfg(feature = "std")]
pub mod learned_ngsa;
#[cfg(feature = "std")]
pub mod learned_quantization;
#[cfg(feature = "std")]
pub mod learned_random_access;
#[cfg(feature = "std")]
pub mod learned_real_corpus_u1;
#[cfg(feature = "std")]
pub mod learned_residual;
#[cfg(feature = "std")]
pub mod learned_residual_anatomy;
#[cfg(feature = "std")]
pub mod learned_residual_codec;
#[cfg(feature = "std")]
pub mod learned_residual_codec2;
#[cfg(feature = "std")]
pub mod learned_residual_entropy;
#[cfg(feature = "std")]
pub mod learned_residual_fusion;
#[cfg(feature = "std")]
pub mod learned_shared;
#[cfg(feature = "std")]
pub mod learned_speech;
#[cfg(feature = "std")]
pub mod learned_speech_trace;
#[cfg(feature = "std")]
pub mod learned_training_cost;
#[cfg(feature = "std")]
pub mod learned_transfer;
#[cfg(feature = "std")]
pub mod learned_u1_wasted;
#[cfg(feature = "std")]
pub(crate) mod measure;
#[cfg(feature = "std")]
pub mod negative;
#[cfg(feature = "std")]
pub mod phase_n;
#[cfg(feature = "std")]
pub mod random_access;
#[cfg(feature = "std")]
pub mod rocm;
#[cfg(feature = "std")]
pub mod rocm_d0;
#[cfg(feature = "std")]
pub mod rocm_d1;
#[cfg(feature = "std")]
pub mod runtime;
#[cfg(feature = "std")]
pub mod runtime_advanced;
#[cfg(feature = "std")]
pub mod semantic;
#[cfg(feature = "std")]
pub mod simd;
#[cfg(feature = "std")]
pub mod transport;

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
        "runtime-advanced",
        "Report 3 whole-repository runtime surfaces: hybrid sleep-then-spin deadline entry, huge-page execution arenas, and SCHED_FIFO/SCHED_DEADLINE/mlockall attempts (optional and additive)",
    ),
    (
        "random-access",
        "Phase M bounded random access: deterministic per-object random windows (seeded from the \
canonical hash, with boundary/first/last edges) over B2 resident PCM, B3-warm disk PCM, B4 the \
decoded-resident FLAC artifact, B4-seek the same artifact via stateless decode_seek (no \
SEEKTABLE) and B5 bounded VOLE, measured against a sequential pass for the random-access penalty",
    ),
    (
        "negative",
        "Phase M negative controls: the incompressible corpus objects (full-width random, \
scrambled) measured against B0/B1/VOLE with no fake wins, plus a hostile-archive battery \
(truncation, bit flip, resealed structural mutation, allocation bomb, garbage) where every \
candidate must be rejected with a typed error and none may panic",
    ),
    (
        "depth",
        "Phase M observation-depth sweep: the minimum buffered lookahead (in quanta) at which \
B2/B3-warm/B4/B5 never underrun, computed from measured per-window latencies at quanta \
64/128/256/512/1024 frames over the frozen corpus",
    ),
    (
        "interference",
        "Phase M adversarial real-time load (contract §49): the frozen workload under idle, \
CPU-burn, memory-bandwidth and storage-IO pressure on all logical CPUs, with a bounded soak \
under CPU contention; uncontrolled conditions and unavailable energy are reported honestly",
    ),
    (
        "all",
        "Phase-M aggregate: runs every Phase-M court (conventional, corpus, fullobj, flagship, \
runtime, random-access, negative, depth, interference) in sequence and is SUPPORTED only when \
all of them are",
    ),
    (
        "compound",
        "Native procedural composition (optimization Track A, experimental profile \
vole.audio.compound.exp1): known constructions materialized exactly and priced against the literal \
floor, FLAC and the bounded VOLE inverse compiler",
    ),
    (
        "archive",
        "Phase N canonical `.volea` archive container: the frozen corpus packaged as \
integrity-bound full-object payloads under a reproducible manifest, with canonical \
decode/re-encode, deterministic encoding and a resealed hostile battery",
    ),
    (
        "transport",
        "Phase N deterministic transport: ordered bounded framing for \
object/state/event/checkpoint/dependency/clock/integrity frames, stream attestation, \
receiver classification of duplicates/stale epochs/gaps/late events/resync, and \
deterministic xrun recovery outcomes (contract §36/§37)",
    ),
    (
        "phase-n",
        "Phase-N aggregate: runs every Phase-N court (archive, transport) in sequence and is \
SUPPORTED only when all of them are",
    ),
    (
        "learned-determinism",
        "Phase O learned canonical determinism: exact closure, repeated-evaluation and \
serialization identity, chunked == contiguous, seek == sequential for every implemented learned \
family",
    ),
    (
        "learned-residual-codec",
        "Phase O exact residual codec family: SparseDelta / DenseI32 / ZigZagVarint / BlockRice / \
PredictiveRice / LiteralResidual over representative and corpus residuals",
    ),
    (
        "learned-exp2-baseline",
        "Exp2 Seal A baseline import: the frozen Exp1 identity is verified and every Exp1 candidate \
remains available inside Exp2 with exp2 bytes <= exp1 bytes",
    ),
    (
        "learned-exp2-mechanisms",
        "Exp2 mechanism portfolio: residual codec v2 + adaptive segmentation + sparse high-order \
linear prediction + long-term prediction + optimizer v2 over the frozen intrinsic corpus, with \
the structural portfolio no-regression gates",
    ),
    (
        "learned-exp2-transfer",
        "Exp2 analytic-first transfer: identity / polarity / delay / affine analytic candidates plus \
an optional bounded learned correction over the analytic residual, with a paired no-regression gate",
    ),
    (
        "learned-exp2-real-corpus",
        "Exp2 real + held-out Mode-C corpus (LibriSpeech, CC BY 4.0): the portfolio against FLAC and \
the Exp1 baseline with exact Wilcoxon signed-rank and a deterministic bootstrap median CI, \
reporting effectiveness and held-out Mode C separately",
    ),
    (
        "learned-real-corpus-u1",
        "Seal S0 U1-domain real-speech replay: the wired three-family Exp2 portfolio over the frozen U1 s16 ingest mapping (i32 = i16 << 16) with new identities, paired against FLAC-5",
    ),
    (
        "learned-speech",
        "Report 3 real-speech portfolio: the growing exact candidate set (baseline dense/sparse/\
hierarchy + fixed finite differences, then LPC …) over the real + Mode-C corpus, paired against \
FLAC-5 with per-family attribution",
    ),
    (
        "learned-u1-wasted",
        "U1-domain common-factor fix: the Wasted model wrapper (kind 16) plus the FactorShift residual codec (id 14) against the S0 U1 baseline portfolio and FLAC-5",
    ),
    (
        "learned-residual-anatomy",
        "Fourth-pass Seal E0 residual entropy anatomy (diagnostic, no format change): the exact
S8 winner residual over the effectiveness clips plus representative Phase-M objects, binarized
canonically, with empirical conditional entropy H(bit | context) under bit position, current
prefix, previous/previous-two residual magnitudes, previous sign, residual FSM, matched lag,
local energy and predictor disagreement; held-out Mode C untouched",
    ),
    (
        "learned-residual-entropy",
        "Fourth-pass Seal E1 attributable residual-codec ladder: re-encode the fixed S8 winner
residual with the pre-E1 Exp3 family versus the Seal E1 signed/FSM adaptive binary range coder
(id 16), reporting per-object and per-codec bytes with exact round-trip; held-out Mode C
untouched",
    ),
    (
        "learned-ngsa",
        "Fourth-pass Seal A0 natural-gradient experiment: a clean-room fixed-point NNGSA
backward-adaptive predictor (model kind 18, preconditioned by an O(p) AR(1) inverse) against
the existing sign-sign adaptive family, by actual complete bytes and residual magnitude, over
the effectiveness clips and a synthetic drifting AR(2)",
    ),
    (
        "learned-residual-fusion",
        "Fourth-pass Seal F0 multi-hypothesis fusion diagnostic (no format change): the plug-in
conditional entropy H(bit | disagreement feature) of the exact S8 winner residual over a small
exact hypothesis ensemble, measuring the upper bound before paying for a side stream or a
closed-loop entropy coder; held-out Mode C untouched",
    ),
    (
        "learned-speech-trace",
        "Seal S0 diagnostic: bit-for-bit trace of the frozen B1 FLAC-5 artifact (subframe kinds, orders, precision, shift, partition order, residual payload) beside the wired three-family Exp2 VOLE byte waterfall, on the real + Mode-C corpus",
    ),
    (
        "learned-residual-codec2",
        "Exp2 Seal B residual codec family: PartitionRice / CoreTailRice / RunLengthRice / \
ZeroMaskRice / BytePlane / ContextRans beside the frozen Exp1 codecs, with the structural gate \
best_v2 <= best_v1 on every residual",
    ),
    (
        "learned-linear",
        "Phase O minimum linear proof: a learned linear finite-field predictor vs the literal \
floor, the existing VOLE inverse compiler, FLAC-5 and simple exact predictors",
    ),
    (
        "learned-intrinsic",
        "Phase O learned intrinsic families (linear, block-local, stateful, nonlinear) vs the \
existing VOLE hypotheses and conventional baselines over the frozen intrinsic corpus",
    ),
    (
        "learned-transfer",
        "Phase O learned transfer operators vs analytic baselines (identity, gain, affine, delay, \
FIR, IIR, convolution, polynomial, piecewise, moving average) with standalone and marginal \
accounting",
    ),
    (
        "learned-residual",
        "Phase O residual shape and residual-cost-aware training: zero fraction, run lengths, \
magnitude statistics, selected codec, and MSE-optimal vs residual-aware fits",
    ),
    (
        "learned-quantization",
        "Phase O canonical precision: post-training quantization vs quantization-aware training at \
the implemented i16 Q12 precision, with exact SIMD parity and explicit i8/mixed unavailability",
    ),
    (
        "learned-capacity",
        "Phase O multi-capacity Pareto surface: several bounded tap capacities per target with \
model/residual/complete bytes, decode work and seek cost; no capacity is privileged",
    ),
    (
        "learned-shared",
        "Phase O shared-model amortization: whole-corpus bytes and the amortization crossover N* \
(or an explicit no-crossover) for identical and distinct object regimes",
    ),
    (
        "learned-random-access",
        "Phase O bounded access: contiguous, chunked, single-frame, small/large range, randomized \
and reverse order, cold/warm repetition, and a stateful checkpoint-spacing sweep",
    ),
    (
        "learned-gpu",
        "Phase O execution surfaces: scalar == SIMD with exact parity and measured throughput; \
CUDA/ROCm learned kernels reported as explicitly unavailable (the frozen device artifact must \
stay byte-identical)",
    ),
    (
        "learned-training-cost",
        "Phase O training cost: wall time, candidates, iterations, quantization attempts, peak \
host RSS and the declared search budget, reported separately from playback",
    ),
    (
        "learned-inverse",
        "Phase O inverse-compiler integration: exact learned candidates admitted only by measured \
Pareto improvement over the existing VOLE best, with typed negative-result reasons",
    ),
    (
        "learned",
        "Phase O aggregate: runs every learned court in sequence and is SUPPORTED only when all of \
them are",
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
        "runtime-advanced" => runtime_advanced::run(receipts_root),
        "random-access" => random_access::run(receipts_root),
        "negative" => negative::run(receipts_root),
        "depth" => depth::run(receipts_root),
        "interference" => interference::run(receipts_root),
        "learned-determinism" => learned_determinism::run(receipts_root),
        "learned-exp2-baseline" => learned_exp2_baseline::run(receipts_root),
        "learned-exp2-mechanisms" => learned_exp2_mechanisms::run(receipts_root),
        "learned-exp2-real-corpus" => learned_exp2_real_corpus::run(receipts_root),
        "learned-speech-trace" => learned_speech_trace::run(receipts_root),
        "learned-speech" => learned_speech::run(receipts_root),
        "learned-real-corpus-u1" => learned_real_corpus_u1::run(receipts_root),
        "learned-u1-wasted" => learned_u1_wasted::run(receipts_root),
        "learned-residual-anatomy" => learned_residual_anatomy::run(receipts_root),
        "learned-residual-entropy" => learned_residual_entropy::run(receipts_root),
        "learned-ngsa" => learned_ngsa::run(receipts_root),
        "learned-residual-fusion" => learned_residual_fusion::run(receipts_root),
        "learned-exp2-transfer" => learned_exp2_transfer::run(receipts_root),
        "learned-residual-codec" => learned_residual_codec::run(receipts_root),
        "learned-residual-codec2" => learned_residual_codec2::run(receipts_root),
        "learned-linear" => learned_linear::run(receipts_root),
        "learned-intrinsic" => learned_intrinsic::run(receipts_root),
        "learned-transfer" => learned_transfer::run(receipts_root),
        "learned-residual" => learned_residual::run(receipts_root),
        "learned-quantization" => learned_quantization::run(receipts_root),
        "learned-capacity" => learned_capacity::run(receipts_root),
        "learned-shared" => learned_shared::run(receipts_root),
        "learned-random-access" => learned_random_access::run(receipts_root),
        "learned-gpu" => learned_gpu::run(receipts_root),
        "learned-training-cost" => learned_training_cost::run(receipts_root),
        "learned-inverse" => learned_inverse::run(receipts_root),
        "learned" => learned::run(receipts_root),
        "all" => all::run(receipts_root),
        "compound" => compound::run(receipts_root),
        "archive" => archive::run(receipts_root),
        "transport" => transport::run(receipts_root),
        "phase-n" => phase_n::run(receipts_root),
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

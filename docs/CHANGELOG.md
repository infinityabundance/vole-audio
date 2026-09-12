# CHANGELOG — VOLE-Audio

The chronological **seal ledger** for this repository: each released version
names its seal, what changed, and the frozen evidence position. It is the
detailed history that used to live in the README, kept here so the README can
stay a concise orientation.

Normative phase charters live in [`PHASE_*.md`](.) and the exact status ledger
is [`PROJECT_STATE.md`](PROJECT_STATE.md). Every claim below is backed by an
immutable receipt under `receipts/`; `vole-audio seal verify` checks the
expected-verdict matrix over the newest receipt per court.

Two conventions are load-bearing and never relaxed:

* **A seal is not a benchmark claim.** The portfolio minimum is the measured
  minimum complete physical bytes over all candidates; negative and
  near-negative results stay visible.
* **The seal subject** is every tracked source file except the excluded
  governance/evidence trees (`receipts/`, `target/`, `scripts/out/`, `docs/`,
  `.git/`). Editing `docs/` never invalidates a seal; editing anything else
  (including this repository's README) does, and requires a reseal.

Tag window note: the repository's tags begin at `v0.22.1`. Earlier seal
releases (Phase M Seals 2–12) are recorded in `PROJECT_STATE.md` but their tags
are not present in the current tag window.

---

## Phase A — Evidence constitution

Receipts, counters, timing, environment capture: every claim is bound to an
immutable, self-verifying `vole.audio.evidence.v1` receipt, and the tooling
refuses to verify a matrix it cannot bind to a clean source tree.

## Phase B — `vole.audio.u1`

The exact universe semantics: sample domain, mapping, profile tag, and the
frozen identity strings.

## Phase C — Scalar oracle

The semantic authority. `SampleObject`, the sampler world/scheduler, and the
scalar evaluator are the reference against which every other surface is
compared bit-for-bit.

## Phase D — Procedural objects

Procedural and hybrid `SampleObject`s; PCM is an observation view, never the
authoritative durable state.

## Phase E — Exact residual / literal + exact WAV ingest

Literal representation as the mandatory universal fallback, residual-governed
objects, and canonical exact WAV ingest.

## Phase F — SIMD

The honest SIMD baseline: scalar == AVX2 == AVX-512 over frozen worlds, with
fixture-level timing. SIMD makes the GPU comparison meaningful rather than
flattering.

## Semantic facts court

F01–F14 are independently device-verified in `court cuda`; F15 is
authority-level and surface-independent. See
[`SEMANTIC_FACTS.md`](SEMANTIC_FACTS.md).

## Phase G — CUDA D0

The buffered-diagnostic CUDA backend: `scalar == SIMD == CUDA` bit-for-bit,
semantic facts F01–F14 re-verified on the device, a random differential subset,
and fixture-level CPU-vs-CUDA throughput cells including a voices × quantum
crossover sweep. Needs the PTX artifact (`scripts/build-cuda-device.sh`); absent
hardware or artifact yields an honest `UNSUPPORTED_BY_HARDWARE` / `INCONCLUSIVE`
receipt.

## Phase H — CUDA D1

The first direct-endpoint evidence: register the **actual** ALSA `hw:` mmap
region (`cuMemHostRegister` DEVICEMAP + device pointer), render each contiguous
mmap chunk's final codes directly into that region (kernel write → stream sync →
in-place shadow verify vs the scalar oracle → `snd_pcm_mmap_commit` with an
exact transferred-frame check), and compare against a D0-mmap baseline running
the same 48 000-frame window. D1 removes 384 KB DtoH + 384 KB host copy (0 B /
0 B in D1): a directness/traffic result, not a latency claim in this court. The
sealed run registered the on-board HDA ring (`snd_hda_intel`) and played
byte-exact with zero xruns (`D1_ENDPOINT_MAPPED`, `HOST_MAPPED`); endpoints that
refuse open/mmap/format/registration stay visible as their own negative rows.

## Phase H.2 — Entropy-native audio core

A deterministic native rANS codec with canonical models, block-addressable pages
and a mandatory RAW fallback; literal and exact-residual entropy
representations; optional EntropyFS persistence and DSFB search governance; CUDA
entropy decode; and the flagship **fused entropy → CUDA → D1 endpoint** court.

Courts: `entropy-rans`, `entropy-literal`, `entropy-residual`,
`entropy-pages`, `entropy-partial`, `entropy-simd` (instruction-SIMD decode
honestly `NOT_IMPLEMENTED`), `entropy-cuda`, `entropyfs` / `dsfb-entropy`
(feature-gated, `INCONCLUSIVE` without), and `entropy-d1` — which decodes
entropy-coded literal, procedural mono+residual, and a high-entropy control on
the GPU per bounded 512-frame window and writes the ring directly, removing
32 768 B GPU→host + 32 768 B host copies (literal) and 16 384 B + 32 768 B
(residual) with zero xruns and byte-exact ring codes. `court h2` is the
aggregate. Normative docs: [`ENTROPY_NATIVE.md`](ENTROPY_NATIVE.md),
[`RANS.md`](RANS.md), [`ENTROPY_ACCOUNTING.md`](ENTROPY_ACCOUNTING.md),
[`ENTROPYFS.md`](ENTROPYFS.md), [`DSFB_SEARCH.md`](DSFB_SEARCH.md), ADRs
0001–0005, and the charter [`PHASE_H2.md`](PHASE_H2.md).

## Phase I — ROCm

A clean `amdgcn-amd-amdhsa` code-object build (`scripts/build-rocm-device.sh`,
thin kernels over the same shared `no_std` semantics, byte-deterministic across
isolated builds), the `backend::rocm` loader probe (GPU → amdgpu → KFD → HIP/HSA)
and two-dimensional, fail-closed `court rocm` / `probe rocm` evidence
(hardware-unavailable on this host with a typed cause).

## Phase J — ROCm D0/D1

`backend::rocm` HIP host runtime with structural resource lifetimes and device
affinity; `court rocm-d0` (differential `scalar == ROCm`) and `court rocm-d1`
(endpoint experiment), both typed to this host's missing device. Their positive
paths execute on a D0/D1-ready ROCm stack.

## Phase K — Inverse compiler

Bounded deterministic proposal search for the cheapest *exact* deterministic
`SampleObject` explanation of an observed window, priced with the H.2
complete-cost oracle and reported as a deterministic Pareto frontier. On the
frozen corpus 6/14 fixtures have a cheaper exact non-literal explanation
(silence 46 B vs literal 183 B, DC 50 B vs 327 B, single-sine 574 B at an exact
period of 64, quasi-periodic 15 937 B, AM signal 1 420 B at residual period
128), while impulse-train, transient-heavy, harmonic-tone, FM, stereo-correlated
and the three negative controls are honestly cheapest as entropy-coded literals.
Courts `inverse` + `flattening`; docs [`INVERSE.md`](INVERSE.md),
[`PHASE_K.md`](PHASE_K.md).

## Phase L — GPU inverse search

The bounded period scan placed on scalar / host-parallel / CUDA / ROCm surfaces:
14 fixtures × 512 candidate periods, identical per-period counts on every
surface, and the device-ranked proposals re-verified to reproduce exactly the
sequential accepted set through the exact evaluator. Measured ratios are
reported as-is with no claim that the GPU wins. Court `inverse-search`; doc
[`PHASE_L.md`](PHASE_L.md).

## Phase M — production depth and conventional baselines

Evidence closed at Seal 13 / v0.22.1. The runtime substrate is measured under a
frozen protocol; the negative, random-access, depth, interference and aggregate
courts exist; the hostile-control definition is shared with the corpus.

| seal | version | deliverable |
| ---- | ------- | ----------- |
| 2 | v0.12.0 | flagship corpus freeze |
| 3 | v0.13.0 | freeze-integrity closure |
| 4 | v0.14.0 | flagship conventional baseline (B0/B1 FLAC) |
| 5 | v0.15.0 | full-object archival container mechanism |
| 6 | v0.16.0 | B1 FLAC vs bounded VOLE inverse selection |
| 7 | v0.17.0 | flagship aggregate + container-integrity review |
| 8 | v0.18.0 | entropy complete-cost physical framing |
| 9 | v0.19.0 | runtime substrate mechanism |
| 10 | v0.20.0 | runtime measurement protocol |
| 11 | v0.21.0 | remaining Phase-M courts |
| 12 | v0.22.0 | evidence-contract closure |
| 13 | v0.22.1 | hostile-control definition |

What remains in Phase M is not another court: the license-clean real-recording
stratum (vacant), energy on a host with a readable cumulative counter, an
unbounded soak, and load conditions this host cannot control — future evidence
extensions, not blockers. Doc [`PHASE_M.md`](PHASE_M.md).

## Phase N — transport and archive

Complete at Seal 1 / **v0.23.0**. Canonical `.volea` archive (explicit
little-endian section table, manifest-frozen session counts, every read
length-checked, unknown kinds rejected); deterministic
`OBJECT/EVENT/STATE/CHECKPOINT/DEPENDENCY/CLOCK/INTEGRITY` transport carrying
procedural/state information rather than mandatory PCM; a bounded receiver
classifying duplicates, gaps, stale epochs, late events and resync; and
reproducible manifests. `court archive` packs the frozen corpus as 115 objects /
115 events / 1 checkpoint / 1 dependency, 31 089 591 payload bytes in a
31 113 564 B archive, decode → re-encode byte-identical, with a resealed hostile
battery. `court transport` + `court phase-n` aggregate. 26-row seal. Doc
[`PHASE_N.md`](PHASE_N.md).

## Phase O — learned deterministic prediction

Complete at Seal 1 / **v0.24.0**. An addendum phase: learned hypotheses are one
more bounded, falsifiable candidate family inside the exact residual-closure
architecture, under the experimental profile `vole.audio.learned.exp1`
(container magic `vole.learned`). `u1/v1` and the `SampleObject` taxonomy are
untouched.

- Six canonical exact residual codecs with deterministic minimum-bytes
  selection.
- i16 Q12 weights, i64 exact accumulators, frozen rounding, proven overflow
  safety and reduction-order independence.
- Families: closed-loop causal FIR, block-local, closed-loop recurrent with
  canonical checkpoints, bounded integer nonlinear graph, and transfer operators.
- One explicit container whose decomposition sums **exactly** to the canonical
  byte length; source dependencies are never free.
- No generic autodiff: ridge/least squares, coordinate descent,
  quantization-aware training minimizing the actual encoded size.
- Courts `learned-determinism` … `learned` (aggregate). 40-row seal. Doc
  [`PHASE_O.md`](PHASE_O.md).

## Exp2 addendum — `vole.audio.learned.exp2` (v0.25.0 – v0.28.0)

The optimization addendum to the sealed Exp1. Exp1 is frozen permanently; Exp2
imports every Exp1 candidate and adds mechanisms beside it, so that
`exp1_candidates ⊂ exp2_candidates ⇒ min(exp2) ≤ min(exp1)` structurally.

Seals A–L: baseline import; residual codec v2 (PartitionRice, CoreTailRice,
RunLengthRice, ZeroMaskRice, BytePlane, ContextRans); complete-byte adaptive
segmentation; sparse/high-order linear prediction; long-term (pitch) prediction;
multichannel lifting; optimizer v2; context mixture + backward adaptation;
hierarchical residual prediction; analytic-first transfer; the real +
held-out Mode-C corpus (LibriSpeech-derived, CC BY 4.0); and stronger external
baselines (WavPack). Courts `learned-exp2-baseline`, `learned-residual-codec2`,
`learned-exp2-mechanisms`, `learned-exp2-transfer`, `learned-exp2-real-corpus`.
Doc [`PHASE_O_EXP2.md`](PHASE_O_EXP2.md).

## Speech campaign — Seals S0–S8 (v0.29.0 – v0.37.0)

The real-speech campaign that took the exact learned portfolio past FLAC-8 on a
frozen LibriSpeech-derived corpus.

| version | seal | deliverable |
| ------- | ---- | ----------- |
| v0.29.0 | S0 | FLAC bitstream trace + U1-domain replay |
| v0.30.0 | S1 | fixed finite-difference predictors (model kind 13) |
| v0.31.0 | S2 | dense per-block LPC |
| v0.32.0 | S3 | variable precision/shift + error-feedback quantisation (+ Levinson–Durbin fix) |
| v0.33.0 | S4 | general Golomb + centered Golomb + Exp3 profile |
| v0.34.0 | S5 | Burg + covariance/least-squares estimators |
| v0.35.0 | S6 | LPC orders to 16 + FLAC-8 control |
| v0.36.0 | S7 | lattice/PARCOR (model kind 14) |
| v0.37.0 | S8 | pole-zero / ARMA (model kind 15) |

## Representation and runtime increments (v0.38.0 – v0.41.0)

| version | deliverable |
| ------- | ----------- |
| v0.38.0 | `Wasted` wrapper (kind 16) + `FactorShift` codec (id 14) |
| v0.39.0 | general (non-power-of-two) GCD factor extraction |
| v0.40.0 | direction wrapper (kind 17) + lattice precision search + Elias–Fano (id 15) |
| v0.41.0 | `runtime-advanced` court |

## Fourth-pass entropy and adaptation ladder — E0–F0 (v0.42.0 – v0.53.0)

| version | seal | deliverable |
| ------- | ---- | ----------- |
| v0.42.0 | E0 | `learned-residual-anatomy` diagnostic |
| v0.43.0 | E1 | `signed_fsm` (id 16) |
| v0.44.0 | E2 | `signed_fsm_sse` (id 17) |
| v0.45.0 | E3 | `signed_fsm_lag` (id 18) |
| v0.46.0 | E4 | `signed_fsm_mix` (id 19) |
| v0.47.0 | E5 | `signed_fsm_rcm` (id 20) |
| v0.48.0 | A0 | `Ngsa` natural-gradient predictor (kind 18) |
| v0.49.0 | C0 | `bgmc` (id 21) |
| v0.50.0 | C1 | `ctw` (id 22) — honest negative |
| v0.51.0 | A1 | banked Ngsa into the speech portfolio |
| v0.52.0 | C2 | `bgmc_sse` (id 23) |
| v0.53.0 | F0 | `learned-residual-fusion` diagnostic (55-row seal) |

Findings that shape later work: the natural-gradient predictor's gain dominates
the entropy-calibration gains (`signed_fsm_rcm` 126 828 → `bgmc` 126 147 →
`bgmc_sse` 126 138 on the fixed S8 residual), CTW is far behind on both
populations, and multi-hypothesis disagreement carries too little information to
justify an architecture change.

## Phase 6 — parsing, representation and entropy (v0.54.0 – )

The ordered Phase-6 campaign. One mechanism per release, each with its own
court, its own attributable delta, and a full battery seal.

### v0.54.0 — `StatefulSyntaxParse` (mechanism 1; 56-row seal)

A decoder-synchronized move-to-front carousel over model tuples (new model kind
19 `stateful_syntax`): symbols `00/01/10` reuse carousel slots, `11` introduces a
fresh tuple. The parser is a `(position, carousel)` shortest path — choosing a
tuple now changes its price later — held under a deterministic bounded beam with
exact dominance deduplication and hard memory ceilings.

Measured: move-to-front saves 57/71/100 B on the synthetic alternating /
palindrome / drifting fixtures against the identical parse under plain segmented
syntax; on real effectiveness clips it wins the portfolio on 2/8 (1673 −31 B,
1988 −217 B), a decisive win on repeated-regime material and an honest
near-negative on most speech.

### v0.55.0 — `IterativeReprice` (mechanism 2; 57-row seal)

Coordinate descent between the stateful parse and the order-0 empirical
magnitude entropy prices induced by that parse (`P0 → Parse0 → P1 → Parse1 …`),
stopping when the exact assembled bytes stop shrinking, the parse repeats, or a
frozen ceiling is reached. Iteration 0 is always retained, so the result cannot
exceed the single-pass parse; the format is unchanged.

Measured: improves the stateful candidate on 5/8 real clips (20–136 B), sometimes
restructuring the parse (174: 6 → 1 segments), and strengthens the portfolio win
on 1988 to 13 532 B vs the 13 776 B portfolio.

### v0.56.0 — `EntropyReblock` (mechanism 3; 58-row seal)

New residual codec **id 24**. The predictor and its segmentation are untouched;
the residual is re-partitioned for entropy coding alone by a shortest path over
256-aligned boundaries, and **each partition selects its base coder** among
Exp-Golomb(0), Rice, general Golomb and BGMC. Strictly more general than
`PartitionRice` (id 6), which is Rice-only over a fixed ladder.

Measured: on the real effectiveness residuals Reblock strictly beats the best
pre-existing codec on 7/8 clips (13–75 B) and beats PartitionRice on all 8
(124–653 B). Landing in `encode_best_v3` moves the speech portfolio
126 899 → **126 620** B on effectiveness (−279) and 137 362 → **136 852** B on
the held-out Mode C (−510), with the 7/8 and 8/8 FLAC-5 win records preserved.
Six courts were re-frozen because Reblock now wins in their `encode_best_v3`.

### v0.57.0 — `SampleExpertMux` (mechanism 4; 59-row seal)

New model kind 20 `expert_mux`. The extent is partitioned into microgroups and
each group selects **exactly one** decoder-synchronized expert (`x̂ = P_{j*}`,
never a mixture); the selector stream is a packed bit field (one bit per group
for two experts). Every expert is stepped from the reconstructed value, so the
trajectories are independent of the selector and the per-group choice is exactly
optimal for the fixed expert set.

Measured: beats the best single expert and the existing adaptive family on 6/8
real effectiveness clips (up to +327 B) while losing marginally on 2; on the
synthetic regime fixtures the best expert dominates and the selector overhead is
a near-negative. It does not beat the portfolio's natural-gradient predictor, so
it is not added to the production portfolio — the court records the family-level
result rather than a portfolio claim.

## Current measured position

Frozen real-speech portfolio (effectiveness + held-out Mode C, LibriSpeech
CC BY 4.0), as of v0.57.0:

| split | VOLE portfolio | FLAC-5 | FLAC-8 | record |
| ----- | -------------- | ------ | ------ | ------ |
| effectiveness (dev-clean, 8 clips) | **126 620 B** | 130 331 B | 129 713 B | 7/8 wins vs both |
| held-out Mode C (test-clean, 8 clips) | **136 852 B** | 144 145 B | 142 714 B | 8/8 wins |

Scale, not universality: this is a speech-corpus result on this host. No
general-audio flagship claim is made, and no real-time/deadline claim follows
from it.

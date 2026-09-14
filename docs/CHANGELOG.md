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

## Phase 7 — the three-objective program (v0.74.0 – )

Phase 7 has exactly three objectives and no fourth track:

```text
7A — entropy-seed proceduralization for sampled audio   (v0.74.0, complete)
7B — vole.audio.lossy.exp1 vs Opus and Lyra             (v0.75.0, this release)
7C — vole.audio.stream.voice.exp1                       (open)
```

### v0.74.0 — Phase 7A: entropy-seed proceduralization

`src/inverse/compound_propose.rs` makes the inverse path discover a
`Compound` explanation **blind** from samples: autocorrelation fundamental
estimation, orthogonal matching pursuit over the *integer tone atoms the graph
can actually emit* with a joint least-squares re-solve and split refinement,
harmonic projection, and envelope/onset fitting — with no fixture dispatch.
`src/inverse/seed.rs` persists it as entropy-coded H state plus an
entropy-coded exact residual, and `CompoundGraph::materialize_range` bounds
materialization to the requested window. Court `learned-entropy-seed`: a known
3-oscillator construction lands in **2 372 bytes** with a residual of ~3 code
units, and the blind path recovers 180/181/270 Hz from samples alone.

### v0.75.0 — Phase 7B: `vole.audio.lossy.exp1`

The first real VOLE lossy profile, built on the Phase-7A principle: a
deterministic explanation `H` is chosen by search to minimise the *actual
emitted representation*, and whatever `H` cannot explain is quantised and coded
with the VOLE-native entropy family. Two engines propose `H`:

* **predictive** — short-term LPC with transmitted, stability-preserving
  quantised reflection coefficients, plus a long-term pitch predictor, driven by
  a decoder-synchronised DPCM loop;
* **transform** — MDCT with a Bark/Schroeder masking model, a separate band
  shape and global gain (with 2-D shape prediction), and a dead-zone scalar
  quantiser.

Rate control is an **absolute** step (reverse water-filling), which is what
makes the allocation MSE-optimal, and the candidate is chosen as the
lowest-distortion representation that meets the byte target. The transform's
per-frame distortion is expressed in time-domain units (`2/N · Σe²`, verified
against a true overlap-add) so the two engines are compared in the same units.

Court `learned-lossy` measures against **external** Opus and Lyra at *matched
actual bitrate* — each competitor's measured `(bitrate, quality)` curve is
interpolated at VOLE's achieved bitrate, because their VBR encoders do not land
on the requested rate — with ViSQOL as the external perceptual metric where the
pinned binary and model run. No competitor code is imported, linked, wrapped or
used as a fallback.

Measured position, the attributed losing cells and the declared remainder are
in [`PHASE_7B.md`](PHASE_7B.md).

## Phase 7C — `vole.audio.stream.voice.exp1` (voice-call profile)

A **streaming** profile whose constitution is latency, packet loss and jitter
rather than archival bytes or a static listening score
([`PHASE_7C.md`](PHASE_7C.md), frozen before implementation).

```text
frame      16 kHz mono, 160 (10 ms) or 320 (20 ms) samples; no whole-clip mode
packet     1 or 2 coded frames, self-contained
parameters ABSOLUTE within a packet: a lost packet can never desynchronise a later one
state      two bounded rings of reconstructed output and excitation
```

* **Predictor** — bounded competition over autocorrelation+Levinson, Burg and
  covariance least-squares, orders 8–16, uniform Q quantisation in the
  **reflection** domain, plus a long-term predictor whose reference is the
  reconstructed *excitation*. Predicting from the reconstructed output instead
  puts the long-term gain inside the short-term filter's feedback path and the
  cascade diverges for legal parameters; the excitation reference is bounded by
  the quantiser step, so it cannot.
* **Residual** — a compact voice residual coder (Elias-gamma zero runs +
  Golomb-Rice, plain Rice, varint) over the same machinery, because every member
  of the general learned residual family writes an 8-byte length prefix that by
  itself exceeds a 10 ms frame's allowance at 8 kbps. The general family stays
  reachable (`id ≥ 128`) and is chosen when it wins on complete bytes.
* **Concealment** — excitation-ring repetition at the last good pitch period
  (voiced) or shaped noise through a four-tap envelope (unvoiced), with monotone
  decay and a fade to the comfort-noise class. Concealed audio is never called
  reconstruction.
* **Capsules / redundancy / DTX** — checksummed state capsules applied to the
  concealment model only; redundancy priced on/off and defaulted **off** because
  the measurement is a net negative; VAD/DTX with procedural comfort noise and
  active/inactive/whole-call rates reported separately.
* **Impairment engine** — seeded and fully deterministic, with no networking
  stack: i.i.d. and Gilbert–Elliott loss, duplication, adjacent reordering,
  uniform and two-mode jitter, late delivery and clock drift. A frame decodes
  exactly when its packet arrived and beat its playout deadline, which makes the
  required jitter-buffer depth a measured quantity.

Court `learned-voice-stream` measures (1) clean quality at matched **actual**
bitrate against external Opus, EVS and Lyra, (2) quality under the loss, burst,
jitter, drift and mixed ladders, and (3) one-way latency with all eight
contributions kept separate. EVS is additionally driven through its own G.192
bad-frame erasure sync word, which is the standard's documented loss simulation.
No competitor code is imported, linked, wrapped or used as a fallback, and no
competitor payload enters a VOLE object.

Measured: encode p99 1.46 ms against a 5 ms budget with zero deadline misses;
one-way p50 31.1 ms on 20 ms frames against the 40 ms envelope; zero concealment
at every jitter rung once the measured depth is provided; recovery to the
no-loss trajectory within 10 ms at 1 % loss. Clean quality **loses** to Opus
(−8.36 dB mean), EVS (−5.75 dB mean) and has no cell inside Lyra's range —
attributed to scalar coefficient transmission setting a ~17.7 kbps floor before
any residual symbol exists. The remainder, with coefficient vector quantisation
first, is declared in [`PHASE_7C.md`](PHASE_7C.md) §13.4.

## Phase 7C.1 — `vole.audio.stream.voice.exp1` increment (v0.76.0)

LSF split multi-stage VQ (`src/voice/vq.rs`, frozen asset
`assets/voice/lsf_msvq_v1.bin`), analysis-by-synthesis CELP excitation
(`src/voice/celp.rs`) with combinatorial pulse-position coding, and two
mechanisms retained but **defaulted off after measurement** (per-subframe
residual gain; per-frame encoder state advance). Measured at matched actual
bitrate: **−7.15 dB** vs Opus, **−4.32 dB** vs EVS (0 of 3 cells won), no
operating point inside Lyra's range. Attribution in
[`VOICE_RD.md`](VOICE_RD.md); the sourced mechanism reference is
[`VOICE_MECHANISMS.md`](VOICE_MECHANISMS.md).

## Phase 7C.2-A/B — `vole.audio.stream.voice.exp2` charter and serializer (v0.77.0)

Charter [`PHASE_7C2.md`](PHASE_7C2.md) frozen **before** code, in the same
discipline as 7C.

* **The `exp1` representation floor, removed.** The generic frame envelope
  duplicates CELP-specific pitch/gain and pads to bytes. For a 320-sample
  pitched VQ-spectrum CELP frame the committed serializer spends 188 information
  bits at zero pulses (padded 192); the new bit-accurate, mode-specific
  serializer (`src/voice/exp2.rs`) spends **128**, and at four pulses 316 →
  **228**. Both figures are asserted by test, not estimated. At 3.2 kbps a 20 ms
  frame is 64 bits, which `exp1` cannot express at all.
* **Challenger court.** `learned-voice-stream-exp2` is frozen before the codec
  work, on a **speaker-disjoint held-out** corpus (test-clean, identity
  `7de2453d2b249130b2edff2b457fcc29a37eb200215387aac82debec4200c6fa`), and
  reports an integrated bitrate-at-equal-quality delta with a case-level
  bootstrap CI plus its robust dual, quality-at-equal-rate.
* **Control baseline (a loss, recorded before tuning).** On the `voice.exp1`
  control: quality-at-equal-rate vs Opus **−4.54 dB SNR** (95 % CI −5.35 …
  −3.72) and −2.61 ViSQOL MOS, vs EVS −4.39 dB / −2.59 MOS, vs Lyra `n/a` (no
  overlapping rate range); bitrate-at-equal-quality vs Opus **+2.72 dB**
  (CI +2.53 … +2.91), vs EVS +3.39 dB. `voice.exp2` is **not yet a live
  profile**; only the serializer's bit accounting and decode fidelity are
  established.
* **Incidental fix.** `predict::analyse` could propose no candidate on a
  degenerate frame (e.g. digital silence), erroring the encoder; it now falls
  back to the stable model `A(z) = 1`. The regression court's result identity is
  unchanged (`cc2addfa…`).

Seal status: the full 76-row `seal verify` battery was **not** re-run for
v0.77.0. The regression court `learned-voice-stream` remains SUPPORTED at
`cc2addfa019a96b3c722ddaabe9fd0dff7eee4ed0cc47039eb435acecc2dba50` and the new
challenger court is SUPPORTED at
`23eaf3610af66d789bb188aedd6e1359ba9ac06ba706d691a2f853845588d43c`; the
repository-wide seal still requires a source-bound battery on the tagged tree.

## Phase 7C.2-C/D — hard-rate core, then multirate side information (v0.78.0)

* **7C.2-C, hard rate conformance.** `Exp2Codec::encode_frame` (`src/voice/exp2.rs`)
  is the hard-rate authority: it enumerates the spectral tiers and excitation
  families, keeps only frames whose **exact** serialised bit count fits the
  declared allowance, and returns the lowest-distortion survivor. There is no
  `OVERSHOOT_LIMIT` in `exp2`; a rate with no richer legal frame falls back to
  `Frame2::minimal` (frozen VQ spectrum + lowest noise core), which fits 64 bits
  and is proven legal at every declared rate by test.
* **7C.2-D, multirate side information.** A CELP frame no longer repeats
  slowly-varying state at the subframe rate. It now sends one frame lag anchor
  (9 b) plus a 4-bit subframe contour, one pitch gain (5 b) plus 3-bit deltas,
  one innovation gain (6 b) plus 4-bit deltas, and **one frame pulse count**
  (3 b) — per-subframe overhead falls from 23 b to ~6 b. A 320-sample CELP frame
  with no pulses drops from 128 to **92 b**, a 60 b/frame saving before any
  pulse is sent. The delta fields are clamped, so the wire is a canonicalising
  map: the property asserted is idempotence plus decode fidelity to the
  read-back shot, not equality to the pre-wire search parameters.
* **Measured ladder (synthetic speech-like signal, 40 frames/rate).** Hard-rate
  SNR at the declared allowances: 3.2 kbps 0.00 dB, 6 kbps 0.09, 8 kbps 0.09,
  9.2 kbps 3.60, 12 kbps 4.61, 16 kbps 4.36. This is a conformance instrument,
  not a quality claim: 6–8 kbps still collapse to the noise core (the spectrum
  is 32 b and the remaining ~88–128 b do not yet buy pulses), and 16 kbps
  *regressed* from 5.45 to 4.36 dB because one shared frame pulse count cannot
  satisfy subframes that want different counts. That tension is the measured
  motivation for 7C.2-E.
* `voice.exp2` is still **not a live profile**. No court result, quality claim
  or bitrate claim attaches to it. The regression court `learned-voice-stream`
  is unaffected (its result identity is unchanged); the whole library test suite
  passes (793 passed, 12 ignored).

## Phase 7C.2-E — fractional-track ACELP (v0.79.0)

A second, exp2-only excitation core (`src/voice/fcelp.rs`, wire **family 3**),
offered *alongside* the 7C.2-B/D core and selected by exact bits against measured
distortion. `exp1` and its frozen synthesis loop are untouched, and the
regression court still reproduces `cc2addfa…` bit-exactly.

* **Fractional long-term prediction.** The lag is resolved at quarter-sample
  resolution by interpolating the reconstructed excitation with a fixed 4-tap
  Lagrange filter. The control core is limited to one 16 kHz sample (62.5 µs);
  `voice_bench pg` measures that its integer predictor adds only 0.66 dB over the
  LPC residual on real speech.
* **Interleaved-track innovation.** Four interleaved tracks of `k` signed pulses,
  with the pulse count carried by the codebook **class** rather than a field. A
  free pulse codebook with one shared gain has a structural flaw — adding a pulse
  always adds energy, so it coarsens rather than refines. `voice_bench cmp`
  measures it: the control core reaches 10.66 dB at one pulse per subframe and
  *falls* to 3.83 dB at two.
* **Measured A/B on the frozen effectiveness corpus** (600 frames,
  `voice_bench exp2`), versus the 7C.2-B/D core: **+1.77 dB at 12 kbps** and
  **+2.23 dB at 16 kbps**; versus scalar+noise, +3.27 dB and +3.26 dB. Synthetic
  hard-rate ladder at those rates moved from 4.61/4.36 dB to 6.63/8.75 dB.
* **Honest limitation.** Family 3 needs 189/253 bits per 20 ms frame, so it is
  selected in **zero** frames at 3.2, 6, 8 and 9.2 kbps. The 12–16 kbps win is
  real; the low-rate target is still open and belongs to the cheaper-side-
  information seals.
* **Repair en route.** `exp2`'s family-1 pulse budget was still derived from the
  pre-7C.2-D per-subframe overhead model, so the encoder believed CELP did not
  fit at 6–8 kbps. Deriving it from the exact serialised size lifted the
  synthetic 8 kbps cell from 0.09 dB to 3.45 dB.
* **Cost, disclosed.** The first working version breached the constitution
  (encode p99 11 477 µs at 16 kbps). Three measured fixes — offer the core once
  per frame instead of six times, hoist the perceptual filter into a precomputed
  `vp::Weighting`, and a weights-passing `synthesize_w` — brought it to
  **4805 µs**, inside the 5 ms budget, with no quality loss.
* **Retained finding.** The long-term gain is *recursive* (it multiplies the
  interpolated reconstructed excitation, so `out(g)` is a polynomial in `g`, not
  linear). This invalidated a planned one-probe search shortcut and is pinned by
  test.

`voice.exp2` is still **not a live profile**; the numbers above are development-
instrument measurements on the frozen effectiveness corpus, not court results.
Library suite: 799 passed, 12 ignored, 0 failed. Both courts reproduce their
frozen identities (`cc2addfa…`, `23eaf361…`).

## Phase 7C.2-F — nested spectral tier and selector proxy (v0.80.0)

Seal 7C.2-F was scoped as rate-scaled spectral precision plus voicing-dependent
allocation. Both halves were implemented; **neither is promoted**, and both
negative results are recorded.

* **Nested spectral operating point.** `vq::stage0` / `vq::pack_stage0` expose the
  MSVQ's stage 0, and `Spectral::Vq0` (wire tier 2) transmits it alone — 18 bits
  instead of 34, a 16-bit saving that is a legal *nested* spectrum because the
  codebook is embedded. Measured A/B: identical to the bit at every declared
  rate, with the tier selected in **zero** frames. At 9.2 kbps the full spectrum
  plus a coarse ACELP frame already fit, so the freed bits buy fewer pulses than
  the precision they cost. Retained in-tree (free when unused, and the
  precondition for 7C.2-H).
* **A structural finding: MSE selects silence.** At 3.2 kbps the measured SNR is
  **0.00 dB** — the output is uncorrelated with the input. Cause: for a
  stochastic excitation the MSE-optimal gain is zero, so silence outscores
  correctly-levelled shaped noise and the lowest-rate fallback degenerates to
  near-silence. This is a concrete instance of the charter's §10 warning that SNR
  cannot be the search objective.
* **Envelope proxy implemented, measured, disabled.** `envelope_penalty` (a
  short-time temporal-envelope term) plus `mse + W·penalty`, as §10 requires. At
  `W = 2.0` waveform SNR *fell* at 6/8/9.2/16 kbps (+0.24 → −0.69, +1.04 →
  +0.73, +1.77 → +1.22, +5.04 → +4.80) and rose only at 12 kbps (+4.23 →
  +4.77). Promotion on dev-only evidence is forbidden and `voice.exp2` is not a
  live profile, so `ENV_WEIGHT = 0.0` and the mechanism is retained in-tree,
  following the `SUBFRAME_GAIN` precedent from 7C.1. A test pins the property.
* **Reading.** The low-rate deficit is not a frame-local parameter-allocation
  problem; the 64-bit frame cannot carry a coarse spectrum, a pitch trajectory,
  gains and a useful innovation simultaneously. That confirms the charter's own
  §18 conclusion that 3.2 kbps must be a synthesis/refinement mode (7C.2-H).

No quality or bitrate claim is made. Regression court `learned-voice-stream`
still reproduces `cc2addfa…`. Library suite: 88 voice tests pass.

## Phase 7C.2-G — transform/PVQ escape mode: a new voice track (v0.81.0)

A new voice-track primitive and an escape excitation core, **implemented,
measured, and shipped disabled**. The most important output is the diagnosis in
[`PHASE_7C2.md`](PHASE_7C2.md) §13.

* **A new track, deliberately.** The charter suggested reusing the existing
transform machinery, but that lives in `src/lossy/` — the general-audio lossy
profile — and cannot be used here. `lossy::mdct::Mdct` is a *framing* transform
reconstructed by windowed overlap-add; inverting its sine window inside a single
frame amplifies edge quantisation noise by ≈200× at `n = 160`, and a per-frame
escape has no second half to cancel it against. `src/voice/pvq.rs` therefore
adds a self-contained orthonormal **DCT-IV** plus **PVQ**
(`count`/`rank`/`unrank`/greedy shape) with **zero dependency on the lossy
track**.
* **Wire placement.** The escape is a **sub-mode of the fallback family** (one
discriminator bit), not a fifth family, so ACELP frames pay nothing for it. A
fifth family would have cost one bit on every frame, including 3.2 kbps where
the escape cannot fit at all.
* **Measured negative.** On the frozen development corpus (600 frames) it is
selected in **zero frames at every declared rate** — the A/B is identical in
bits and SNR. Encode p99 reached **11 258 µs** at 16 kbps; sharing the transform
across pulse densities (it depends on the tier, not on `k`) brought that to
**8864 µs**, still outside the 5 ms constitution. Shipped disabled via
`Options::default().tcx = false`.
* **Verified faithful before being called a negative.**
`the_escape_core_reconstructs_its_own_quantisation` drives the real encode path
and checks the decoded shape correlates > 0.5 with what it was given, so the
result is about the selection, not the wiring.
* **The generalising diagnosis.** With MSE as the objective, a reconstruction
with correlation `ρ` has error `2E(1−ρ)` while silence has error `E`, so
**silence outscores every reconstruction with `ρ < 0.5`**. A sparse PVQ
transform shape at these rates sits below that threshold. MSE does not merely
prefer weak reconstructions; at the bottom of the rate range it prefers
*nothing*. Every non-CELP core at low and mid rate is currently judged by an
objective structurally biased against it — and the natural fix (7C.2-F's
envelope proxy) lowered waveform SNR and so cannot be promoted without an
external perceptual court.
* **Structural constraints found.** `V(40,8) > 2^32`, so the block size or the
index width had to change — the block was set to the codec's own 5 ms subframe.
And the greedy PVQ score `(x_i)²` ties across signs, so a search that does not
break the tie toward the target picks the wrong sign; both are pinned by tests.

Regression court `learned-voice-stream` still reproduces `cc2addfa…`. Voice
tests: 95 passed.

## Phase 7C.2-H (first step) — perceptual arbiter (v0.82.0)

`voice_bench visqol` runs the court's **frozen ViSQOL protocol** (same binary,
same speech-mode lattice model, same `moslqo` column) on the **development**
corpus, so nothing fitted here contaminates the held-out challenger court. The
protocol is duplicated into the instrument rather than shared, so a dev probe
cannot silently change what the court measures.

Calibration: on three development cases ViSQOL scores a self-comparison at
**4.402 MOS-LQO** — its ceiling here, not 5.0 — and one call takes ≈0.42 s.

Measured MOS-LQO mean:

| rate | noise only | +celp | +acelp | +tcx | bits/frame noise → acelp |
| ---- | ---------- | ----- | ------ | ---- | ------------------------ |
| 3.2 kbps | 1.000 | 1.000 | 1.000 | 1.000 | 46.0 → 46.0 |
| 6 kbps | 1.205 | 1.205 | 1.205 | 1.205 | 87.8 → 87.8 |
| 8 kbps | 1.151 | 1.180 | 1.180 | 1.180 | 99.1 → 122.1 |
| 9.2 kbps | 1.151 | 1.156 | 1.156 | 1.156 | 108.4 → 143.3 |
| 12 kbps | 1.190 | 1.221 | **1.342** | 1.342 | 123.7 → 209.1 |
| 16 kbps | 1.110 | 1.144 | **1.369** | 1.369 | 149.7 → 246.1 |

* **7C.2-E's win survives a perceptual judge.** The fractional-track core beats
the control core at 12 kbps (1.342 vs 1.221) and 16 kbps (1.369 vs 1.144) — the
same two rates, the same direction as waveform SNR. That seal's claim does not
depend on SNR being a good proxy. This is its strongest evidence.
* **7C.2-G's negative is confirmed perceptually**: `+tcx` is identical to
`+acelp` at every rate.
* **Absolute position**: 1.0–1.37 MOS-LQO against a 4.40 ceiling, corroborating
the court's −2.61 / −2.59 ViSQOL MOS against Opus / EVS.
* **New defect — and it is a rate defect.** With only fallback cores available
the codec spends **99.1 of 160 bits** at 8 kbps and **149.7 of 320** at 16 kbps:
up to **53 % of the frame allowance is left idle**, because MSE prefers a 12-bit
stochastic frame to a several-hundred-bit scalar frame that is weakly but
positively correlated. The MSE objective costs the codec bandwidth as well as
quality, and no additional excitation machinery can help while the selector is
free to choose near-silence and leave the frame half-empty.
* At 3.2–9.2 kbps the spread between cores is within ±0.03 MOS: the low-rate
range is uniformly non-competitive regardless of core, reaching 7C.2-F's
conclusion independently by a perceptual measure.

Regression court `learned-voice-stream` unaffected. Voice tests: 95 passed.

## Phase 7C.2-H (second step) — fitted selector weight (v0.83.0)

The most important result of the phase. §12–§14 established that the MSE
selector objective was the binding blocker; this seal fits a perceptual proxy on
development material against ViSQOL, as the charter prescribes, and measures what
changes.

**The fit** (`voice_bench weights`), mean MOS-LQO over five rates:

| weight | 6 kbps | 8 kbps | 9.2 kbps | 12 kbps | 16 kbps | mean |
| ------ | ------ | ------ | -------- | ------- | ------- | ---- |
| 0.0 | 1.205 | 1.180 | 1.156 | 1.342 | 1.369 | 1.250 |
| 0.5 | 1.390 | 1.309 | 1.183 | 1.304 | 1.298 | 1.297 |
| **2.0** | 1.445 | **1.383** | **1.408** | 1.414 | 1.428 | **1.416** |
| 4.0 | **1.496** | 1.248 | 1.305 | **1.443** | **1.459** | 1.390 |
| 8.0 | 1.316 | 1.318 | 1.413 | 1.303 | 1.377 | 1.345 |

Unimodal and shallow; `2.0` peaks the mean and is best or near-best at every
rate, while `4.0` collapses at 8 kbps. Frozen as `DEFAULT_ENV_WEIGHT`, recorded
as a **dev fit** (three cases) with **no claim attached** until the held-out court
measures it.

**SNR and perceptual quality now disagree in sign.**

| rate | SNR (w=0 → w=2) | ViSQOL MOS (w=0 → w=2) |
| ---- | --------------- | ---------------------- |
| 6 kbps | +0.24 → **−0.69** | 1.205 → **1.445** |
| 8 kbps | +1.04 → +0.73 | 1.180 → **1.383** |
| 9.2 kbps | +1.77 → **1.22** | 1.156 → **1.408** |
| 12 kbps | +4.23 → +4.77 | 1.342 → 1.414 |
| 16 kbps | +5.04 → +4.80 | 1.369 → 1.428 |

At 6 and 9.2 kbps the measures move in **opposite directions** and the perceptual
judge prefers what the waveform measure rejects. Mean MOS rises 1.250 → 1.416
(+13 %), largest where the codec was weakest. This is the measured vindication of
the charter's refusal to make SNR the search objective: for three seals it was not
merely imprecise at the bottom of the rate range, it pointed the wrong way.

**7C.2-G's verdict is reversed.** Under MSE the escape core was selected in zero
frames. Under the fitted proxy it is selected at 8 and 9.2 kbps and is *better*
(1.406 vs 1.383; 1.481 vs 1.408) at the same or fewer bits. It still ships
disabled, because the cost stands (4464 µs at 8 kbps, 5379 at 9.2 kbps, 8.8 ms at
16 kbps); the switch remains so the reversal is reproducible. A mechanism can be
rejected by a bad objective and be fine.

**Cost, disclosed as pre-existing.** Encode p99 at 16 kbps is 5527 µs against the
5 ms constitution — but the *control* `voice.exp1` baseline is already 5811 µs in
the regression court, so the voice profile's encode deadline is an inherited open
problem, not a regression introduced here.

Regression court `learned-voice-stream` unaffected. Voice tests: 95 passed.

## Phase 7C.2-H (third step) — procedural excitation, and the diagnosis it forced (v0.84.0)

The charter's §13 proposes the decoder *materialise* the excitation. `src/voice/proc.rs`
implements it: a pitch-synchronous glottal excitation blended with shaped noise,
every sample a pure function of the transmitted fields and the sample index, so
packet independence holds by construction. It costs
`lag_q(11) · gain(6) · voicing(4) · phase(6) · seed(3)` = 34 bits as a sub-mode of
the fallback family, so ACELP frames pay nothing for it. Tests pin the three
properties that matter: exact transmitted RMS at every voicing, pulses on the
transmitted fractional period, and determinism.

**Measured: selected in zero frames.** Forced to be the only core on offer, its
output is bit-identical to the stochastic fallback at every rate (MOS-LQO 1.000 /
1.445 / 1.453 / 1.423 / 1.158 / 1.211 at 3.2/6/8/9.2/12/16 kbps), and identical in
waveform SNR and bits/frame. Offering it costs 0.9–1.5 ms encode, so it ships
disabled.

* **Two hypotheses tested and rejected.** (a) *Phase resolution*: a comb carries
  no memory, so its phase must be transmitted, and the first version spent 3 bits
  on four positions inside `period/8`. The diagnostic
  `phase_resolution_is_what_limits_a_memoryless_pulse_train` shows the coarse grid
  is a limiter in isolation (a 64-step grid reaches ≈0.70 against a pure comb at
  period 81). The phase became a 6-bit field **solved analytically**
  (`proc::best_phase`). Effect on the codec: none. (b) *Lag resolution*: the lag
  field is already quarter-sample but only integer lags were searched; searching
  its full transmitted resolution costs no wire bits. Effect: none.
* **The real cause, measured.** `voice_bench pg`: order-16 LPC already extracts
  **18.31 dB**, and a pitch predictor on that residual adds only **0.66 dB**
  (18.97 dB total). Order-16 short-term prediction over 20 ms has absorbed most of
  the pitch periodicity, so the residual the excitation must code is very nearly
  pitch-free.

> **The binding constraint is the short-term predictor's order, not the excitation
> model.** A high-order LPC that has already eaten the pitch leaves every long-term
> mechanism under 1 dB of headroom and leaves the quantiser an almost-white
> residual — which the MSE objective then declines to code.

This one number explains all three excitation negatives of the phase (7C.1's weak
adaptive codebook, 7C.2-G's escape core, this procedural core) and why 7C.2-E's
fractional LTP only began paying at 12–16 kbps. The standards do not make this
trade: SILK pairs a *low-order* short-term filter with a fifth-order long-term
predictor, deliberately splitting the prediction. Re-balancing short-term against
long-term prediction order is the next work on this path.

Regression court `learned-voice-stream` reproduces `cc2addfa…`. Voice tests: 100
passed. New code clippy-clean and rustfmt-clean.

## Phase 7C.2-H (fourth step) — the prediction split is refuted (v0.85.0)

A diagnostic-only increment, and the discipline is the point: the previous seal
ended by *proposing* an architecture change (re-balance short-term against
long-term prediction, as SILK does), so this seal measures the hypothesis before
anything is built. `voice_bench split` reports the **total** prediction gain per
short-term order, with the long-term gain at both frame-wide and 5 ms-subframe
resolution.

```text
 order |  frames | short-term | LTP frame | gain  | LTP 5 ms | gain
     8 |     600 |   17.15 dB |  17.95 dB | +0.80 |  18.71 dB | +1.56
    10 |     600 |   17.59 dB |  18.34 dB | +0.75 |  19.05 dB | +1.47
    12 |     600 |   17.86 dB |  18.62 dB | +0.76 |  19.33 dB | +1.47
    14 |     600 |   18.11 dB |  18.82 dB | +0.71 |  19.49 dB | +1.38
    16 |     600 |   18.27 dB |  18.96 dB | +0.69 |  19.62 dB | +1.35

 order-16 width sweep
 width |  frames | short-term | LTP frame | gain  | LTP 5 ms | gain
     6 |     600 |   18.27 dB |  18.96 dB | +0.69 |  19.62 dB | +1.35
     8 |     600 |   18.43 dB |  18.98 dB | +0.55 |  19.62 dB | +1.19
```

* **No beneficial re-balance exists.** Lowering the short-term order raises the
  long-term gain (+0.69 → +1.56 dB) but by *less* than the short-term gain falls
  (18.27 → 17.15 dB). Total prediction gain is maximized at the **highest** order,
  monotonically. SILK's low-order/high-order split does not transfer here.
* **Quantisation is not hiding pitch either.** A finer spectrum at the top order
  (width 8) raises the short-term gain and *lowers* the long-term gain, leaving the
  total unchanged at 19.62 dB. A better short-term filter simply absorbs more of
  the pitch — the same trade from the other direction.
* **A correction to the previous seal.** The 0.66–0.69 dB recorded there is the
  frame-wide figure; the codec carries one lag and gain per 5 ms subframe, and the
  per-subframe value is roughly twice as large (1.35–1.56 dB). The instrument now
  reports both so the figure cannot be misread.

**What this closes off.** Total prediction gain is 19.62 dB — the residual carries
1/92 of the signal energy, so the predictor is good. The codec's loss is not
prediction, which confirms `VOICE_RD.md`'s attribution to the residual quantiser
and closes the prediction-order direction entirely. The remaining levers are
quantisation efficiency — the spectral envelope (charter §16) and packet-reset
entropy coding (7C.2-I) — not more excitation structure and not a re-balanced
predictor.

Diagnostic only: no codec behaviour changed, and the regression court is
unaffected. Voice tests: 100 passed.

## Current measured position

Frozen real-speech **lossless** portfolio (effectiveness + held-out Mode C,
LibriSpeech CC BY 4.0), as of v0.73.0:

| split | VOLE portfolio | FLAC-5 | FLAC-8 | record |
| ----- | -------------- | ------ | ------ | ------ |
| effectiveness (dev-clean, 8 clips) | **126 264 B** | 130 331 B | 129 713 B | 7/8 wins vs both |
| held-out Mode C (test-clean, 8 clips) | **136 783 B** | 144 145 B | 142 714 B | 8/8 wins |

Lossy position (`vole.audio.lossy.exp1`, `learned-lossy` receipt) is reported in
[`PHASE_7B.md`](PHASE_7B.md) and deliberately kept separate: it is a
rate/quality comparison against external Opus and Lyra, not a size claim.

Scale, not universality: this is a speech-corpus result on this host. No
general-audio flagship claim is made, and no real-time/deadline claim follows
from it.

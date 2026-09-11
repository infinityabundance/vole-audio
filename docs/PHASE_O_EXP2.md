# PHASE O — Exp2 addendum (`vole.audio.learned.exp2`)

## Purpose

Exp2 is the optimization addendum to the sealed Phase O (Exp1). It keeps the
strongest parts of the original — learned transfer operators, learned intrinsic
prediction, finite-field / stateful / block-local families, residual-aware
training, deterministic integer evaluation, SIMD parity, shared-model
amortization — and hardens the research further.

Exp2 does **not** mutate Exp1. Exp1's profile tag, four model kinds, six
residual codecs and canonical container bytes are frozen permanently. Exp2 is a
new profile that imports every Exp1 candidate and adds mechanisms beside it, so
that under the same complete-physical-byte objective:

```text
exp1_candidates ⊂ exp2_candidates   ⇒   min(exp2) ≤ min(exp1)
```

This is a structural no-regression guarantee, not a benchmark claim.

Two win definitions are kept strictly separate:

* **Family-level win** — a learned family beats alternatives on structurally
  appropriate material.
* **Portfolio-level win** — keep *all* old candidates, add new ones, select the
  minimum complete physical bytes.

Learned families are **not** forced to win incompressible or trivially-analytic
cells; doing so would be benchmark gaming.

## Namespace

| item | value |
| ---- | ----- |
| universe | `vole.audio.u1` (unchanged) |
| Exp1 profile (frozen) | `vole.audio.learned.exp1` |
| Exp2 profile | `vole.audio.learned.exp2` |
| container magic | `vole.learned` (shared) |
| format version | `1` (shared) |
| Exp2 evidence schema | `vole.audio.learned.exp2.evidence.v1` |

## Seal sequence

| seal | deliverable | modules | court | status |
| ---- | ----------- | ------- | ----- | ------ |
| A | Exp2 baseline import (Exp1 frozen) | `learned::profile` | `learned-exp2-baseline` | **sealed** |
| B | Residual codec v2 | `learned::residual_codec2` | `learned-residual-codec2` | **sealed** |
| C | Complete-byte adaptive segmentation | `learned::segmented`, `learned::segmentation` | `learned-exp2-mechanisms` | **implemented** |
| D | Sparse / high-order linear prediction | `learned::sparse`, `learned::train::sparse` | `learned-exp2-mechanisms` | **implemented** |
| E | Long-term / pitch prediction | `learned::ltp`, `learned::train::ltp` | `learned-exp2-mechanisms` | **implemented** |
| F | Multichannel prediction / lifting | `learned::multichannel`, `learned::train::multichannel` | `learned-exp2-mechanisms` | **implemented** |
| G | Optimizer v2 (multiscale + beam + memo) | `learned::train::optimizer2` | `learned-exp2-mechanisms` | **implemented** |
| H | Context mixture + backward adaptation | `learned::context_mixture`, `learned::train::context_mixture`, `learned::adaptive`, `learned::train::adaptive` | `learned-exp2-mechanisms` | **implemented** |
| I | Hierarchical residual prediction | `learned::hierarchy`, `learned::train::hierarchy` | `learned-exp2-mechanisms` | **implemented** |
| J | Transfer v2 (analytic-first) | `learned::analytical`, `courts::learned_exp2_transfer` | `learned-exp2-transfer` | **implemented** |
| K | Real + held-out Mode-C corpus | — | — | pending |
| L | Stronger external baselines | `baseline::wavpack` | — | **implemented (WavPack)** |

## Seal B — residual codec v2

Exp1's six codecs stay selectable; six new exact codecs are added:

| id | codec | mechanism |
| -- | ----- | --------- |
| 6 | `PartitionRice` | partitioned Rice over a frozen length ladder (16…1024), deterministic shortest path with per-partition Rice parameters |
| 7 | `CoreTailRice` | two-regime Rice with an Exp-Golomb escape and a global frozen parameter search |
| 8 | `RunLengthRice` | VOLE-native adaptive run-length/Rice in the RLGR family (decoder-visible state only; *not* bit-compatible with Malvar's RLGR) |
| 9 | `ZeroMaskRice` | zero mask (bitmap / RLE / sparse delta) + magnitude stream (Rice / varint) |
| 10 | `BytePlane` | byte-length plane + significance byte planes + sign stream |
| 11 | `ContextRans` | context-conditioned rANS over magnitude bit-length buckets, reusing the frozen audio rANS core |

Hard gate (structural): `best_v2(residual) ≤ best_v1(residual)` for every
residual, because every Exp1 codec remains selectable and selection is a strict
minimum with ascending-id ties. Verified by `court learned-residual-codec2` and
unit tests.

## Frozen result hashes (Exp2 seals so far)

| court | result sha256 |
| ----- | ------------- |
| `learned-exp2-baseline` | `3bd61665b214a0bb85e3eee311bf598cbcb7f76e94cc646c057406897682f111` |
| `learned-residual-codec2` | `e134f0c3457a9593e8ab56d071e142c2d3c03a60280c9434e62eca0c433cbcf2` |
| `learned-exp2-mechanisms` | `9ef1b8b5ae79b6ed0a2f42532e8ce4d19107675d4de010b788cf7c5704a53ba9` |
| `learned-exp2-transfer` | `a43ec91fbb31efd9e585e6cfaaa3fefe3bc6e8381aec7049f508cc3d212d265a` |

The frozen Exp1 court `learned-residual-codec` still reproduces its sealed hash
`49f8d5c6…` after the Exp2 changes, so Exp1 is provably unperturbed.

## Measured Exp2 mechanism results (first run)

Over the first six intrinsic cases, the Exp2 portfolio minimum is never larger
than the Exp1 dense-linear candidate, and each mechanism pays on the material
the literature predicts:

| case | class | Exp1 dense linear | Exp2 sparse | Exp2 hierarchy | Exp2 best | best non-learned |
| ---- | ----- | ----------------- | ----------- | -------------- | --------- | ---------------- |
| `sine-440` | pure tone | 4 429 B | 4 064 B | **1 818 B** | 1 818 B | 9 094 B |
| `triangle-220` | periodic | 7 348 B | **839 B** | 846 B | 839 B | 3 232 B |
| `square-mix` | periodic-rich | 5 397 B | 3 908 B | **1 727 B** | 1 727 B | 8 118 B |
| `stereo-unison` | identical stereo | 9 327 B | — | — | **1 443 B** | 15 053 B |
| `vibrato-tone` | quasi-periodic | 5 108 B | 3 883 B | **1 765 B** | 1 765 B | 8 704 B |
| `detuned-partials` | quasi-periodic | 5 052 B | 4 342 B | **1 952 B** | 1 952 B | 8 999 B |

The clearest single-channel wins: sparse selected lags explain the periodic
`triangle-220` at 839 B rather than 7 348 B of dense coefficients; the
hierarchical cascade roughly halves the sparse cost on the tonal and
quasi-periodic cases (e.g. `square-mix` 3 908 B → 1 727 B). The clearest
multichannel win: reversible mid/side lifting collapses identical stereo to
1 443 B versus 9 327 B for the dense stereo FIR and 15 053 B for the best
non-learned baseline.

## Measured analytic-first transfer results (first run)

The analytic-first decomposition does the right thing on both ends of the
spectrum (court `learned-exp2-transfer`):

| pair | transform | analytic-only | analytic+correction | winner |
| ---- | --------- | ------------- | ------------------- | ------ |
| `identity` | identity | **197 B** | 226 B | analytic |
| `delay-13` | delay | **197 B** | 226 B | analytic |
| `gain-0.5` | fixed gain | **1 082 B** | 1 111 B | analytic |
| `affine` | affine | **1 082 B** | 1 111 B | analytic |
| `fir3` | FIR | 9 223 B | **3 590 B** | correction |
| `iir1` | IIR | 9 810 B | **9 183 B** | correction |
| `room` | convolution | 9 228 B | **9 176 B** | correction |
| `bandlimit-b0.25` | bandwidth limit | 9 391 B | **8 007 B** | correction |
| `procedural-to-sampled` | procedural manifestation | 9 448 B | **6 016 B** | correction |

Trivial relations deliberately stay analytic (the learned correction is larger
correctly rejected); complex deterministic relations earn a learned correction.
This is the intended separation between analytic syntax and learning.

## Backward adaptation is an honest negative on the short corpus

The backward-adaptive family (frozen integer sign-sign LMS, decoder-visible
state only) is implemented and exact, but it **loses** on the 4096-frame
intrinsic corpus: 7 816–8 475 B against 839–4 342 B for the sparse / hierarchical
families. This is the expected shape of the mechanism — backward adaptation
amortizes model-description cost over *long* recordings, and a 4096-frame
window is too short for its slow sign-sign convergence to pay off. The negative
is recorded rather than hidden, and the portfolio minimum is unaffected.

The context-mixture family (model kind 11) is likewise an honest near-negative
here: on the stationary synthetic cases it costs essentially the same as sparse
(4 071 B vs 4 064 B on `sine-440`, 846 B vs 839 B on `triangle-220`) because the
gate adds framing without changing the local regime. It is expected to pay only
where the residual regime genuinely changes within a window.

## Honest limitations (recorded, not hidden)

* Seals C, D, E, F, G, H, I and J are fully implemented and covered by the
  `learned-exp2-mechanisms` / `learned-exp2-transfer` courts.
* No new `seal verify` receipt matrix has been written yet: the Phase M/N
  batteries were not re-run in this change set.
* Seal K (real + held-out Mode-C corpus) is **not implemented**; it remains the
  declared remainder of the Exp2 sequence, and the real-recording stratum stays
  `VACANT_DECLARED` as in Phase M (no rights-clean source audio is present on
  this host).
* `RunLengthRice` is a VOLE-native adaptive run-length/Rice design; it is in the
  RLGR family but is not bit-compatible with Malvar's RLGR1.
* Sparse and long-term prediction are mono-first in this build; multichannel
  structure is handled by the dedicated reversible-lifting family (Seal F).
* The WavPack baseline is an external row (verified to round-trip exactly) and
  has no VOLE semantic authority; SRLA and MPEG-4 ALS are recorded as
  unavailable (no reproducible implementation on this host).

# PHASE O — learned deterministic prediction

## Mission

Phase O is an **addendum** research phase, executed only after Phases A–N are
sealed. It investigates whether compact *learned deterministic state* can serve
as a useful **hypothesis class** inside VOLE-Audio's existing exact residual
closure architecture.

It does **not** redefine VOLE-Audio, turn it into an "AI codec", replace
procedural representations, weaken exactness, weaken the literal fallback, or
make learned inference semantic authority. It does not modify the frozen
`vole.audio.u1` / `u1/v1` semantics.

The invariant is unchanged:

> `SampleObject` is authoritative. Sample-domain audio is an observation view.
> Literal fallback remains mandatory. Intrinsic exact closure remains mandatory.
> Scalar semantics remain authoritative. Learned prediction is a hypothesis
> class. Training/search has **zero** semantic authority.

The central question:

> Can a compact executable learned hypothesis explain enough deterministic
> structure that `learned state + exact residual + dependencies + checkpoints +
> metadata + runtime cost` is a better measured representation for some audio
> classes than literal, existing VOLE procedural models, conventional
> predictors or conventional lossless codecs — while preserving exact
> deterministic observation semantics?

Phase O never assumes the answer is yes.

## Namespace

| item | value |
| ---- | ----- |
| universe | `vole.audio.u1` (unchanged) |
| experimental profile | `vole.audio.learned.exp1` |
| container magic | `vole.learned` |
| format version | `1` |
| evidence schema | `vole.audio.learned.evidence.v1` |

Learned objects are **not** `u1/v1` `SampleObject`s and do not extend the frozen
`Representation` taxonomy. A future profile admitting learned representations is
a separate, versioned decision.

## Implementation order (O.62)

| step | deliverable | module | court |
| ---- | ----------- | ------ | ----- |
| O0 | inherited-state seal + namespace | `learned::profile` | — |
| O1 | residual codec foundation | `learned::residual_codec` | `learned-residual-codec` |
| O2 | learned arithmetic | `learned::arithmetic` | (determinism) |
| O3 | linear finite-field | `learned::finite_field` | `learned-linear` |
| O4 | canonical serialization | `learned::serialization`, `learned::object` | `learned-determinism` |
| O5 | minimum linear proof | `learned::train::linear` | `learned-linear` |
| O6 | SIMD | `eval::learned_simd` | `learned-gpu` |
| O7 | transfer operator | `learned::transfer`, `learned::train` | `learned-transfer` |
| O8 | nonlinear vocabulary | `learned::graph`, `learned::train::finite_field` | `learned-intrinsic` |
| O9 | residual-cost-aware training | `learned::train::objective` | `learned-residual` |
| O10 | quantization-aware training | `learned::quantize`, `learned::train::quant_aware` | `learned-quantization` |
| O11 | CUDA/ROCm evaluators | — (see below) | `learned-gpu` |
| O12 | block-local predictor | `learned::finite_field` (`block_frames`) | `learned-random-access` |
| O13 | stateful predictor | `learned::stateful` | `learned-random-access` |
| O14 | shared learned models | `learned::accounting` | `learned-shared` |
| O15 | inverse-compiler integration | `courts::learned_inverse` | `learned-inverse` |
| O16 | production corpora | `learned::corpus` | all |
| O17 | full Pareto seal | `court learned` | `learned` |

## Seal ledger

| seal | release | subject | result hashes |
| ---- | ------- | ------- | ------------- |
| 1 | v0.24.0 | `a347bb22…` | determinism `a0c9f027…`, residual-codec `49f8d5c6…`, linear `db4aed44…`, intrinsic `6080d106…`, transfer `10fc0eb6…`, residual `80c3fbbf…`, quantization `0d8b1250…`, capacity `7879ecc8…`, shared `eab4d545…`, random-access `62b5d41d…`, gpu `542afe4f…`, training-cost `5527eadd…`, inverse `6ce95cb0…`, aggregate `3e3073f7…` |

Seal 1 detail: **40-row** `seal verify` matrix (the Phase M and N matrices plus
the 13 learned courts and the `learned` aggregate); 513 passed / 12 ignored
all-features, 503 passed / 12 ignored default-features. Device artifacts rebuilt
at the seal commit and byte-identical to the frozen values (`d13d22c3…` PTX,
`5c30a4bc…` AMDGPU), so Phase G/J evidence is unchanged. Release **v0.24.0**.

## Honest limitations recorded by the seal

* **Device execution (O.33–O.35, O.56).** The learned device kernel is *not*
  part of the frozen device artifact: adding it would change the Phase G/J PTX
  and AMDGPU bytes and invalidate sealed device evidence. CUDA is therefore
  `NOT_IMPLEMENTED` for learned evaluation, ROCm is `UNSUPPORTED_BY_HARDWARE`,
  and the host SIMD surface is measured with exact scalar parity.
* **Precision (O.22, O.52).** The canonical container stores i16 Q12 weights and
  i32 Q12 biases. i8 and mixed precision are reported as `NOT_IMPLEMENTED`
  rather than faked; they need a weight-bits/scale format extension.
* **Families (O.8, O.13).** Nonlinear and stateful families are mono-only in
  this build.
* **Corpus (O.37, O.38).** The corpora are deterministic synthetic material; the
  real-recording stratum remains `VACANT_DECLARED` as in Phase M.

## Non-claims

See [LEARNED_NON_CLAIMS.md](LEARNED_NON_CLAIMS.md). In short: no claim that
learned representation is smaller, faster, more general, perceptually better,
or closer to a "true source process"; no claim that tensor cores are normative;
no change to VOLE's exactness model.

# Semantic facts — independent-oracle coverage

> Phase G prerequisite, committed with the A–F state. This document is the
> companion to `src/facts.rs` and `court facts` (receipts under
> `receipts/facts/`).

## Why independent facts exist

Differential parity proves **backend** equality:

```text
scalar == SIMD == (later) CUDA == (later) ROCm
```

But a bug shared by every backend survives differential testing: a world
observed identically by two engines can still be *identically wrong* (Phase F
found two such scalar-semantic bugs — the endless-voice natural-end bug and
`compose_transpose` wraparound — only because the SIMD port forced the shared
path to be re-examined).

Each semantic fact is therefore an **independent statement about
`vole.audio.u1`**. Its expected samples are derived from first principles —
closed-form integer math, hand-enumerated boundary sequences, or vectors
produced by an independent implementation (e.g. the Python oracle for
`VOLE-SPLITMIX64-STREAM`) — *never* by calling the sampler/evaluator code
under test, and are then observed through the full `World` observation path on
every available host surface.

A fact that fails means the oracle semantics are wrong — not merely that two
backends disagree.

## Rule

> **Every representation and every transform that ships from this phase on
> carries at least one fact row.** Adding a representation without a fact is a
> review failure. Facts have stable ids (`F01`…) — they are never renumbered;
> new facts append. Expected vectors are frozen; changing u1 semantics means
> re-deriving the affected expectations from first principles and recording
> the change in the ledger.

## Representation/transform coverage

| Representation / transform      | Facts                                                        |
| ------------------------------- | ------------------------------------------------------------ |
| `Silence`                       | F01 (liveness baseline: exact zeros)                         |
| `Constant`                      | F02 (identity), F03 (half-gain rounding)                     |
| `Oscillator`                    | F04 (exact phase points at 12 kHz / 48 kHz)                  |
| `Noise`                         | F05 (frozen stream vectors, independent Python oracle)       |
| `Wavetable` / `SingleCycle` / `ExactRepeat` | F06 (fractional interp), F07 (periodicity)          |
| `Literal`                       | F08 (reverse one-shot), F09 (loop boundary), F10 (content)   |
| `PredictorResidual`             | F13 (closure H+R == X_O)                                     |
| `Referenced`                    | F14 (unity transpose), F15 (wide-integer transpose oracle)   |
| Transform: gain                 | F03                                                          |
| Transform: rate / position      | F06 (1.5×), F08 (reverse), F09 (loop wrap)                   |
| Transform: interpolation        | F06 (linear, fractional)                                     |
| Transform: envelope             | F11 (exact ADSR knots + release)                             |
| Transform: pan                  | F10 (endpoints + center, equal-gain)                         |
| Transform: mix                  | F12 (i64 sum, single final saturation, voice-bus ceiling)    |

## Surface matrix

Rows stay visible as backends arrive: `rocm` joins with its phase, and every
row must pass there too. `cuda` is live since Phase G: `court cuda` re-runs
every windowed fact (F01–F14) on the device surface before reporting
`SUPPORTED` (F15 is authority-level and surface-independent).

```text
                    scalar   simd/scalar   simd/avx2   simd/avx512   cuda   rocm*
semantic fact         ✓          ✓            ✓            ✓          ✓      —
reference hash        ✓          ✓            ✓            ✓          ✓      —
random differential   —          ✓            ✓            ✓          ✓      —
                      (*: ROCm arrives with its phase; rows stay visible)
```

`court facts` verdict is `SUPPORTED` only when every fact passes on every
surface available at run time; otherwise it writes an honest
`FAILED_CORRECTNESS` receipt. `cargo test` enforces the same check as a unit
test (`facts::tests::every_fact_passes_on_every_host_surface`), so a semantic
regression fails CI even without a court run.

## Fact registry (stable ids)

| Id   | Name                              | Independent expectation (summary)                            |
| ---- | --------------------------------- | ------------------------------------------------------------ |
| F01  | silence                           | Silence observes as exact zeros over any window.             |
| F02  | constant-unity-identity           | `Constant(1234567)` at unity gain/env → 1234567 (identity chain). |
| F03  | constant-half-gain                | Half gain: `mul_q16` round-half-away → 1234567 → 617284.     |
| F04  | oscillator-exact-phase-points     | 12000 Hz @ 48000 advances exactly 2^62/frame: samples `0, +(2^31−2), 0, −(2^31−2)`, period 4; windows at 0 and 100. |
| F05  | noise-frozen-vectors              | `VOLE-SPLITMIX64-STREAM` frames 0..16 for seed `0x0BAD5EED20260D1A`, produced by an independent Python implementation and frozen. |
| F06  | wavetable-fractional-interpolation | Cycle `[0,8000,16000,0]` at 1.5 frames/frame → hand-computed period-8 sequence (all even spans ⇒ exact halves). |
| F07  | exact-repeat-periodicity          | Cycle content at unity rate: `X[t] == P[t mod 4]` (independent modulo oracle, 40 frames). |
| F08  | reverse-one-shot-sequence         | Rate −1 from the last frame reproduces the content backwards and silences exactly at the natural end frame. |
| F09  | loop-boundary-enumeration         | Region `[10,20)` entered at 19: hand-enumerated `19,10,11,…,19,10`. |
| F10  | pan-endpoints-and-center          | Hard left `(level,0)`, center equal-gain halves (exact on even level), hard right `(0,level)`. |
| F11  | envelope-exact-knots              | `Constant(2^16)` through A4/D4/S=U/2/R4, note-off at 12 → hand-computed knots `0,16384,…,65536,57344,…,32768×5,24576,16384,8192`. |
| F12  | mix-sum-and-saturation-bound      | 4×2^30 → `i32::MAX`; `+2^30,+2^30,−2^30,−2^30` → 0; single 2×-gain voice saturates at the voice bus. |
| F13  | residual-closure-equals-intrinsic | Zero model + closing records over a formulaic intrinsic → closure H+R == X_O at every frame. |
| F14  | reference-unity-transpose         | Referenced object at unity transpose observes exactly its target's content. |
| F15  | transpose-wide-integer-oracle     | `compose_transpose` == independent i128 formula on 100 000 random pairs; ceiling saturation at ±2^47; out-of-domain composed rate rejected at voice resolution (never wrapped). |

Expected oscillator/pan/envelope values use the exact u1 arithmetic
(`rnd_shift` round-half-away, `mul_q16`, `lerp_i32`); they are closed-form
integer derivations, not float approximations.

## Verification discipline

- Expected values are frozen with the facts; changing them is a profile
  change, never a silent edit (mirroring the court reference hashes).
- The F05 vectors are reproducible from `scripts/`-independent Python:
  `splitmix64_finalize` per the freeze record, verified byte-for-byte against
  `NOISE_VECTORS_16` before commit.
- The CUDA column (Phase G) is enforced by `court cuda`, which flattens each
  fact world, renders every window on the device, and compares to the same
  first-principles expectations — a fact must pass on the device surface
  before that backend reports `SUPPORTED`.
- When ROCm (Phase I) lands, extend the matrix the same way: every fact row
  must pass on the device surface before that backend reports `SUPPORTED`.

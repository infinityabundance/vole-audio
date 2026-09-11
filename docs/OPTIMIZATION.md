# Whole-repository optimization program

A campaign over the parts of VOLE-Audio **outside** the Phase O Exp2 learned
work. Every item keeps the evidence constitution: exact outputs, real byte
accounting, no corpus-name special cases, and the previous implementation kept
as a control.

## Track A — native procedural composition (`compound`)

**Spec audit first.** `docs/U1_SPEC.md` is the normative authority for `u1/v1`
and contains **no** Compound payload syntax and **no** Compound observation
semantics; `Representation::Compound = 0x0A` exists only as a code tag with a
one-line descriptor comment. Implementing a payload behind `0x0A` would be a
silent semantic break, so the capability is a **separate experimental profile**:

| item | value |
| ---- | ----- |
| universe | `vole.audio.u1` (unchanged) |
| profile | `vole.audio.compound.exp1` |
| magic | `vole.compound` |

Bounded deterministic graph over already-exact primitives: silence, constant,
the frozen DDS oscillator (frozen sine table + `eff_incr`), Q16 gain, integer
delay, the frozen analytic ADSR law, and an exact i64 `Add` saturating once.

Court `compound` (known constructions, exact materialization, priced against the
literal floor, FLAC and the bounded VOLE inverse compiler):

| case | compound | literal | FLAC | u1 best |
| ---- | -------- | ------- | ---- | ------- |
| `polyphony-3` | **168 B** | 192 054 B | 70 062 B | 192 830 B |
| `percussion-adsr` | **159 B** | 192 054 B | 22 446 B | 42 709 B |
| `layered-pad` | **207 B** | 192 054 B | 57 575 B | 192 830 B |

Result sha256 `31ddcb66…`.

## Track B — B5 hot path + entropy decode table

`FullObjectReader` now uses a binary search plus a monotonic sequential cursor,
direct-to-destination materialization (`RepresentedLiteral::materialize_into`,
`RepresentedResidual::materialize_closure_into`) with page accounting returned as
a traversal by-product, and no `BTreeSet` second pass.

The measured hot cost turned out to be the per-symbol rANS model lookup, not the
scan/copy. `SymbolModel::slot_table()` derives a `slot -> entry` table over the
frozen 16 384-slot domain from the canonical model (no stored byte changes;
decoded symbols are identical), used above a 4096-symbol threshold.

Measured on the frozen Phase-M runtime court, B5 all-115 population, identical
bytes:

| metric | before | after | factor |
| ------ | ------ | ----- | ------ |
| p50 | 882 ns | 901 ns | — |
| p90 | 37 741 ns | 34 404 ns | 1.10× |
| p99 | 303 310 ns | 117 501 ns | **2.58×** |
| p99.9 | 840 850 ns | 392 618 ns | **2.14×** |
| max | 1 299 311 ns | 591 782 ns | **2.20×** |
| mean | 20 769 ns | 10 087 ns | **2.06×** |

## Report 3 — real-speech campaign (Seal S0 …)

FLAC-5 beats the wired three-family Exp2 real-speech portfolio by ~3.6–4.6 %.
This campaign attributes that gap with evidence before changing any codec.

### Seal S0 — diagnostic (no codec change)

Two new courts, both diagnostic:

* `learned-speech-trace` parses the **actual** frozen B1 FLAC-5 artifact bit for
  bit (`baseline::flac_trace`): frame-header CRC-8, frame-footer CRC-16, and a
  full sample reconstruction that must equal the encoder's round trip. It
  classifies every subframe and records its order, coefficient precision,
  prediction shift, warmup/coefficient bits, residual partition order and
  residual payload bits, beside the VOLE complete-byte waterfall.
* `learned-real-corpus-u1` replays the portfolio under the **frozen U1 s16
  ingest** (`i32 = i16 << 16`) with new identities/hashes in
  `corpus/real_manifest_u1.json`.

**What FLAC-5 actually does on the 16 clips.** All frames are LPC (order 4–8,
dominant 8), coefficient precision 15, prediction shift 12–14, partitioned Rice
with per-block partition orders up to 5. Constant and verbatim subframes are
essentially never selected; fixed predictors appear only occasionally. FLAC's
residual payload is ~98 % of its bytes; framing/metadata is ~130 B/clip.

**Where VOLE's bytes go.** The wired portfolio's winning family is sparse-10 or
hierarchy-2; the model costs 16–70 B and metadata ~110 B, so VOLE's own framing
is *comparable to or smaller than FLAC's*. The entire gap is residual bytes:
over `dev-clean`, FLAC's residual payload is 128 305 B against VOLE's 133 881 B
(+4.3 %), and the totals are 130 331 B vs 135 124 B.

**Conclusion.** The loss is a *prediction/residual-quality* loss, not a
container-overhead loss. FLAC fits a local dense all-pole predictor every
4096 samples; the wired VOLE portfolio fits whole-clip sparse/hierarchical
models. This is exactly the model-class mismatch the campaign targets.

**U1-domain surprise.** Under the true U1 mapping every sample has 16 low zero
bits. FLAC exploits them through wasted bits (its size is unchanged), but the
learned portfolio does not: VOLE's U1 bytes balloon to ~3× FLAC (397 283 B vs
130 006 B over `dev-clean`). This is a real, exact, bounded gap — a
common-factor/wasted-bits mechanism is required before the U1 domain is
competitive on integer-scaled content.

Frozen result hashes: `learned-speech-trace` `0661a292…`,
`learned-real-corpus-u1` `d07cb3e7…`.

Seal S0 also lands the **dense-LPC model vocabulary** (`learned::lpc`, model kind
`12`, coefficients stored exactly as `i32` with a declared precision and
arithmetic shift), with its own exact-closure and accumulator-proof unit tests.
The fitting and portfolio wiring arrive in Seal S2; landing the vocabulary first
keeps S2 a pure add-candidate change.

## Status

Implemented and pushed: Track A, Track B (including the entropy decode table),
and the Report 3 Seal S0 diagnostic courts (`learned-speech-trace`,
`learned-real-corpus-u1`). The remaining tracks from the two whole-repository
optimization reports
(CPU frame-tile multicore + PartialBank vectorization, GPU work decomposition,
CUDA/HIP graphs, entropy model p2 / compatible-model reuse, integer packing,
lifting, page-local LZ, reciprocal rANS encode, PreparedWorld, voice coalescing,
SHA acceleration, borrowed/mapped views, GPUDirect Storage, ALSA hardware-clock
scheduling, radix score assembly, PGO/BOLT, energy counters) are **not yet
implemented** and remain the declared remainder of the campaign.

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

## Status

Implemented and pushed: Track A, Track B (including the entropy decode table).
The remaining tracks from the two whole-repository optimization reports
(CPU frame-tile multicore + PartialBank vectorization, GPU work decomposition,
CUDA/HIP graphs, entropy model p2 / compatible-model reuse, integer packing,
lifting, page-local LZ, reciprocal rANS encode, PreparedWorld, voice coalescing,
SHA acceleration, borrowed/mapped views, GPUDirect Storage, ALSA hardware-clock
scheduling, radix score assembly, PGO/BOLT, energy counters) are **not yet
implemented** and remain the declared remainder of the campaign.

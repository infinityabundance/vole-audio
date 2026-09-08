# ENTROPY NATIVE — the corrected architecture and representation contract

> Owner of the corrected architecture. Companion specs: `docs/RANS.md`
> (frozen codec), `docs/ENTROPY_ACCOUNTING.md` (complete-cost rules),
> `docs/ENTROPYFS.md` (optional persistence), `docs/DSFB_SEARCH.md`
> (zero-authority search governance). This file answers *what is being
> represented and where sample-domain materialization first becomes
> necessary*.

## The architectural correction

VOLE-Audio must not become merely

```
procedural synthesizer state -> PCM samples -> endpoint
```

Its deeper architecture is

```
deterministic explanation
  + entropy/configuration state
  + entropy-coded irreducible residual
    -> bounded observation
      -> endpoint sample codes
```

The defining principle:

> **Store the deterministic explanation. Entropy-code what the explanation
> cannot reproduce. Materialize sample-domain observations only when actually
> required.**

For sampled-origin material, the baked waveform is **not** assumed to remain
the authoritative durable representation merely because the DAC eventually
consumes sample codes. The desired representation path is

```
source audio
  -> deterministic hypothesis H
  -> exact residual R = Residual_rho(X, H)
  -> reversible residual symbolization Psi(R)
  -> entropy model M
  -> canonical native rANS payload C_R
  -> authoritative VOLE-Audio representation
```

and observation is

```
VOLE state + rANS state + residual-page index + voice/event state
  -> decode only required entropy pages
  -> exact intrinsic closure
  -> rate/envelope/filter/mix/...
  -> final endpoint codes
```

## SampleObject semantics are not redefined

`SampleObject` remains the semantic object, with the frozen representation
taxonomy and the frozen U1 closure semantics. What H.2 introduces is an
explicit distinction between

- the **semantic SampleObject** (what the object *is*), and
- the **physical/canonical representation** of that object (how its
  information is stored).

A SampleObject may be represented physically as (H.2.1):

- literal raw samples;
- literal entropy-coded samples;
- procedural state;
- procedural state + entropy-coded residual;
- referenced state / compound state;
- predictor + entropy-coded residual;
- transfer operator + entropy-coded residual;
- (Phase O) learned hypothesis + entropy-coded residual.

Entropy coding is **orthogonal to the hypothesis family**: "rANS object" is
not itself an audio semantics. The exact residual algebra remains whatever
U1 declares (for v1: `X_O = H + R` with sparse replacement records, one i64
sum and a single saturation — see `object/residual.rs` and
`docs/U1_SPEC.md` §14); H.2 never substitutes ordinary arithmetic addition
for the existing residual type, and never changes Phase-E residual semantics
to make entropy coding convenient.

## Sample-domain exposure surfaces (H.2.15 / H.2.38)

Sample values physically exist in bounded, attributable surfaces. The
following surfaces are distinguished in code, counters, and receipts:

| Surface | Meaning | Example |
| --- | --- | --- |
| persistent sample-domain | stored baked waveform | a `Literal` payload's resident i32 samples |
| entropy state | coded residual/sample information | rANS payload bytes, page index |
| entropy model bytes | normalized frequency tables | inline/shared model bytes |
| transient decoded page | bounded decode scratch | one page's reconstructed symbols during observation |
| global decoded waveform | full-object decoded sample buffer | forbidden on native H.2 paths (must be receipted if ever used) |
| D0 output block | VRAM sample block for host transfer | Phase G diagnostic |
| host materialization buffer | application host PCM staging | Phase H D0 hostbuf |
| endpoint ring | ALSA mmap region / FIFO / DMA | registered endpoint-visible region |
| verification buffer | court instrumentation reads | endpoint-region readback for equality proof |

Materialization state is never conflated with verification instrumentation.
A transient decoded page in GPU shared memory is not a persistent full-object
waveform — but it **is** sample-domain state and is counted under the
appropriate surface.

## Native (non-GPU) entropy materialization

`src/entropy/represent.rs` implements physical representations:

- `RepresentedLiteral { descriptor, symbolization, pages, index }` —
  reconstructs the canonical U1 intrinsic sample codes **exactly**.
- `RepresentedResidual { model (semantic), coded sparse residual pages }` —
  reconstructs the identical semantic `object::residual::Residual`.

Both expose:

- full materialization (for oracle equality), and
- partial materialization (`materialize(start_frame, frames, channels)`),
  which decodes only the entropy pages intersecting the requested range.
  Required equality (H.2.13): *partial observation == the same slice of the
  full scalar observation* for every tested object.

Page rules (H.2.6): every page is independently decodable; page size is
explicit representation metadata (no universal optimum; see page-Pareto
court); random access never requires decoding unrelated pages; reverse/loop/
non-unit-rate/seek observations decode at most the pages in the dependency
closure of the requested window, with the halo recorded and accounted.

## The strongest execution path (H.2.18/19)

```
GPU-resident procedural state
  + GPU-resident entropy state
    -> bounded rANS decode (only required pages)
    + procedural evaluation
    + exact residual closure
    + observation transformations
    + mix
      -> final S32 endpoint codes written directly into the
         already-proven D1 ALSA-mapped endpoint region
```

without: a persistent decoded full waveform, a full residual waveform, a host
PCM cache, a voice × frame global PCM matrix, or a D0 global sample block on
the D1 path.

This is **not** a claim that sample values physically never exist — see the
surface table. The objective is eliminating unnecessary baked/persistent/
intermediate waveform representations, not physics.

## Representation container

Physical representation records are versioned, endian-defined,
self-delimiting, integrity-protectable, and dependency-identifiable so Phase
N can embed them without semantic redesign (H.2.44). The complete archive
grammar is **not** frozen in H.2; only the canonical record shapes below are:

```
physical object record  = vole.entropy.p1 profile tag + record kind +
                          semantic descriptor bytes + physical payload +
                          dependency content-ids + integrity
page record            = see RANS.md block container
page index record      = per-page (start_frame, frames, byte offset, length,
                          model reference, payload kind)
model record           = content-addressable normalized model bytes
```

The semantic descriptor bytes are the existing canonical header bytes
(`object::descriptor::canonical_header_bytes`) so content identity of the
semantic object is preserved exactly.

## Observation entry points

- `observe_full` — scalar full reference (semantic authority).
- `observe_partial` — page-bounded entropy observation.
- optimized CPU (page-parallel), SIMD (exact; parallel surfaces only),
  CUDA (page/voice/channel parallel), CUDA D0, CUDA D1 — all must produce
  final sample codes identical to the scalar reference.

The flagship demonstration (H.2.53) runs one sampled-origin object through
all of these paths plus its high-entropy negative control and preserves both
results.

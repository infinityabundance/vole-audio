# PHASE M — production depth / courts

## Mission

The contract's deliverable list (§52):

* frozen ~100-object corpus;
* full baseline ladder;
* depth sweep;
* interference;
* long run;
* energy where measurable;
* crossover results.

It is the phase where the project stops measuring fixtures and starts making
*production* claims, so it is bound by the contract's own ordering rule:

> "Freeze a corpus before tuning." … "Do not tune the corpus after seeing the
> results." (§48)

and by its reporting rule: *"Report crossover surfaces, not just one number"*
and *"say exactly that"* when a backend is only excellent under favourable
conditions (§48, §49).

## What is in this increment (Seal 1): the conventional baselines

Phase M starts with the part that does not need the flagship corpus yet and that
every later comparison depends on: **B1, a real conventional lossless baseline**.

### B1 — conventional lossless codec (contract §47)

| decision | value |
| -------- | ----- |
| input | the **exact** canonical interleaved `i32` domain |
| bit depth | **32 bits/sample** |
| rate | the object's own (no resampling); the generated fixtures are rate-agnostic and evaluated at the u1 default 48 kHz, exactly as the H.2 courts do |
| channels | the object's own, 1..=8 (FLAC's ceiling) |
| conversion | **none** — no `>> 8`, no dither, no normalisation |
| primary preset | **compression level 5** (the `flac` tool's and libFLAC's documented default) |
| secondary controls | level 0 (speed-biased) and level 8 (size-biased) |
| implementation | `libflac-rs = "=0.143.1"`, pinned exactly, in-process, pure Rust |
| authority | **zero VOLE semantic authority** — a comparator only |
| correctness | `decode(encode(x)) == x` for every sample, or the court is `FAILED_CORRECTNESS` |

This is deliberately *not* the historical H.2 comparator. The H.2 court compares
an external `flac` over a **24-bit converted** source and stays frozen as
historical evidence in `courts::entropy_common`. B1 is a new, exact, 32-bit,
in-process baseline that prices the same information problem the VOLE
representation is priced against. B1 also cannot become `UNSUPPORTED`: a
conventional baseline that disappears when a tool is not installed would be an
escape hatch, not a control.

### The reference oracle (non-authoritative)

`baseline::reference` runs the system `flac`, when installed, on the same exact
`i32` domain at the same settings (`-5 --no-padding --force-raw-format --bps=32`)
and checks its own round trip. It is **never** part of the frozen result vector
(its bytes depend on which reference encoder is installed) and its absence is
`NOT_AVAILABLE`, never a failure.

It exists because it immediately found something worth recording. `libflac-rs`
ports libFLAC **1.4.3**, whose constant-signal detection keys off a
`fixed_residual_bits_per_sample[1] == 0` test that the guess predictor only
produces below 28 bits/sample. At 32 bits/sample libFLAC 1.4.3 therefore does
**not** select the CONSTANT subframe, and an all-zero block costs about one bit
per sample; newer reference encoders do select it. On this corpus:

```text
silence         B1  2,178 B   reference    160 B   (13.6x difference)
everything else B1 ~ reference (<2%), e.g. dc 138 vs 160 (B1 smaller)
aggregate       B1 288,791 B  reference 281,087 B  (reference/b1 = 0.973)
```

Without the oracle row that divergence would have been invisible, and the
comparison would have quietly flattered VOLE on exactly the structural class
where VOLE is meant to win. Recording both is the honest form.

### The B0–B9 ladder manifest

The receipt carries every ladder row with its status, and none is deleted:

```text
B0 literal PCM                         MEASURED          court conventional
B1 conventional lossless (FLAC-5)      MEASURED          court conventional
B2 PCM-resident sampler                MEASURED          court runtime (Seal 10)
B3 disk-streaming sampler              MEASURED          court runtime (cold/warm verified,
                                                         read traffic, cache state)
B4 compressed-file decode + playback   MEASURED          court runtime (exact B1 artifact)
B5 VOLE scalar                         MEASURED          court runtime (bounded full object),
                                                         semantic/facts/inverse
B6 VOLE CUDA buffered                  MEASURED_ELSEWHERE court cuda
B7 VOLE ROCm buffered                  MEASURED_ELSEWHERE court rocm-d0 (hardware-gated)
B8 VOLE D1 attempt                     MEASURED_ELSEWHERE court d1 / entropy-d1
B9 VOLE D2 attempt                     NOT_IMPLEMENTED   D2 is future conceptual
```

### Also in this increment

The CUDA context-construction path flagged during the Phase-L freeze is closed:
`Cuda::open` now owns a freshly created context with a `ProvisionalContext`
guard, so an error (or panic) between `cuCtxCreate` and the RAII owner destroys
the context and leaves the caller's context stack exactly as it was.

## What is *not* in this increment

Stated plainly so the gap is visible (updated at Seal 11 — the corpus is frozen,
the conventional and runtime baselines are measured, the full-object container
is frozen and hostile-tested, the negative controls and the observation-depth,
random-access and interference courts exist):

* the **license-clean real-recording stratum is declared and vacant** (see
  `corpus/README.md`); no production claim rests on real recordings yet, and
  `court corpus` records that rather than staying silent;
* **energy is not measurable on this host** (no hwmon power input, no
  NVML/AMDSMI): `court interference` reports `NOT_AVAILABLE` rather than
  inventing a figure, and the probe works on hosts that do expose a source;
* the adversarial load matrix covers only conditions this host can create
  (CPU, memory bandwidth, storage); compositor/display load, competing GPU
  compute, GPU context contention, DVFS, thermal steady state and PCIe power
  saving are `NOT_CONTROLLED`;
* the **long run is a bounded soak** (10 s under CPU contention), not an
  unbounded endurance run;
* `court d2` remains unimplemented (D2 is future conceptual), as do the
  `inspect`/`verify`/`encode`/`observe`/`play`/`bench` CLI surfaces.

## Seal history

### Seal 1 — B1 conventional baseline (2026-09-10)

Seal run (release, `--all-features`, clean tree, version 0.11.0):

- `court conventional` **SUPPORTED**: 14 fixtures over the exact canonical `i32`
  domain, `libflac-rs =0.143.1`, level 5 primary with levels 0 and 8 as controls,
  **42 exact round trips verified** (14 fixtures × 3 levels), every row with its
  STREAMINFO audio MD5 verified.
- Results: B0 raw PCM **1,179,648 B**; B1(level 5) **288,791 B** (0.245× raw,
  0.245× the VOLE u1 literal byte count); B1(level 0) 385,114 B; B1(level 8)
  283,660 B. Reference oracle (system `flac` 1.5.0, 14/14 exact):
  **281,087 B**, reference/B1 **0.973**.
- The divergence above (silence: B1 2,178 B vs reference 160 B; 1.4.3 constant
  subframe selection at ≥28 bits/sample) is recorded per fixture and as a
  limitation, not smoothed away.
- Frozen static result hash
  `11f8683f6ab5df3641ea95fe8349fcbdd4e10e410d4d7d83f328e2d2c23fdd0f`
  (in-process B1 only; the reference oracle is deliberately outside it).
- CUDA construction hygiene (the Phase-L review's narrow error path) closed with
  `ProvisionalContext`; the 9 gated GPU tests still pass.

## Where this goes next

1. **Seals 2–12 — the corpus is frozen, verified and review-closed; the flagship
   B0/B1 conventional baseline is measured; the full-object container mechanism
   and its parser are frozen; the B1-vs-VOLE result exists with clean population
   arithmetic; the entropy complete-cost oracle equals the physical artifact;
   the runtime substrate is measured under a frozen protocol with a stratified
   crossover surface; the negative, random-access, depth and interference courts
   exist with a Phase-M aggregate; and the Seal-11 evidence defects are closed.**
2. **What remains in Phase M** is not another court: the license-clean
   real-recording stratum (currently vacant), energy on a host that exposes a
   readable cumulative counter, an unbounded soak rather than the bounded one,
   and the conditions this host cannot control. Then Phase N — transport/archive;
   Phase O — learned deterministic prediction addendum.
3. **Release status.** 0.19.0 and 0.20.0 are published; 0.21.0 was deliberately
   left unpublished (it carries the Seal-11 evidence defects). 0.22.0 is
   committed and pushed; publish it once the crates.io 24-hour version quota
   permits, then begin Phase N.

### Seal 2 — flagship corpus freeze (2026-09-10)

Seal run (release, `--all-features`, clean tree, version 0.12.0):

- **115 objects, ~3.6 minutes of material** (217.8 s) at 44.1 kHz ×9, 48 kHz
  ×85, 96 kHz ×13, 192 kHz ×8; generated, never stored.
- Populations: **110 B1-comparable**, **5 excluded by format domain**
  (`>8` channels = `NOT_APPLICABLE_BY_FORMAT_DOMAIN`, recorded so the exclusion
  can never become a hidden denominator change).
- Stratification, per axis (counts are the frozen population, not results):
  representation — literal 7, oscillator 29, wavetable 12, exact_repetition 7,
  residual 12, compound 19, noise 29; amplitude — low_byte 5, s16 52, s24 30,
  full 28; channel structure — mono 79, identical 6, correlated 4,
  anti-correlated 4, independent 7, multichannel 15; temporal — stationary 90,
  transient 6, loop 7, one-shot 2, slowly varying 5, strongly modulated 5;
  entropy — highly predictable 28, locally predictable 22, globally periodic 19,
  sparse residual 17, spectrally structured 9, full-width random 14, scrambled 6.
- Hostile controls are hostile by construction and by test: full-width uniform
  and bit-scrambled noise with independent per-channel seeds, independent stereo
  rather than duplicated channels, no shared low-byte structure, no short
  accidental period (checked to p ≤ 1024), no DC bias and no reduced dynamic
  range (all asserted by unit tests).
- `court corpus` **SUPPORTED**: 115 objects regenerated and hash-matched;
  corpus sha256 `c0fc62ff…`, manifest sha256 `a575bf12…`.
- The gate is itself tested by a **mutation battery**: schema, corpus-hash,
  missing-object, extra-object, class, rate, size, content-hash, generator,
  zero-rate, zero-channel, zero-frame and population-count mutations all fail
  with the expected finding kind.
- Duplicate-id defect found and fixed during the freeze (objects differing only
  in channel structure, or in a swept parameter, had collided); ids now carry
  the channel structure, and the membership test asserts uniqueness.
- B1's integrity invariant is now enforced inside `b1_flac` itself (a
  STREAMINFO MD5 failure is an error at every compression level, for every
  caller), not only at the court's primary call site.

### Seal 3 — review-found freeze-integrity closure (2026-09-10)

Still **before any flagship result**. The freeze passed review; the review found
nine freeze-integrity holes that would have let the population's *interpretation*
be adjusted after the fact even though the sample bytes were frozen. All nine are
closed, and the corpus was regenerated under one explicit amendment while the
flagship comparison is still unrun:

1. **B1 eligibility is derived, not trusted.** `verify_manifest` and the courts
derive `b1_comparable` from each object's channel count
(`generate::b1_comparable`) and require the manifest's audit field to equal the
derivation; the comparison denominator can no longer be changed by editing a
flag.
2. **Whole-object canonical verification.** For each frozen `Spec`,
`verify_manifest` regenerates the samples and compares the manifest entry
against the canonical `object_for(spec, samples)` field for field, replacing the
hand-maintained subset comparison; an unknown or mutated label is a mismatch
(`identity_changed`, naming the differing fields), never a silent default.
3. **Identity covers provenance.** `identity_bytes` now includes `class` and
`conversion` (the latter becomes provenance-critical once real recordings
arrive); `duration_ms` is derived and recomputed+verified, and
`expected_inclusion_surfaces` is derived policy.
4. **Root and state validated.** `universe`, `profile` and `state` must be the
frozen values (`root_mismatch` otherwise), not merely the schema.
5. **Duplicates and order rejected.** A repeated object id (`duplicate_id`) and a
manifest whose id sequence differs from the frozen membership (`order_mismatch`)
both fail: benchmark order is experimental state (cache, thermal, GPU-clock
history), so the frozen *sequence*, not merely the set, is preserved.
6. **The anticorrelated-random control is reclassified.** The full-width
`AnticorrelatedStereo` object is kept — temporally random + perfectly
cross-channel structured is a valuable control — but is no longer counted as
incompressible; hostile controls now require genuinely independent channels, and
the hostile invariants (unbiased per channel, both signs, full width, no period
≤ 1024, distinct channel hashes, no `L == R` / `L == -R`) run over every control.
7. **Axis renamed.** The frozen axis is now `source_structure_class`, distinct
from the representation the inverse compiler later *selects* (`oscillator` may
compile to `exact_repeat`; that is a result, not a contradiction).
8. **`corpus freeze` refuses to overwrite a frozen manifest.** Amending a frozen
corpus requires a deliberate `--amend-frozen <reason>` defining a new corpus
identity, so the freeze act can no longer silently retune the population.
9. **Regenerated and resealed with no flagship measurement in existence**, which
is what makes this the one legitimate amendment to a frozen corpus.

The manifest was regenerated under that amendment: corpus sha256
`4c94b841…`, manifest sha256 `f67c73cf…`, populations unchanged at 115 objects
(110 B1-comparable, 5 excluded by format domain). Device artifacts are
byte-identical (PTX `d13d22c3…`, AMDGPU `5c30a4bc…`), and the semantic /
authored / inverse / inverse-search frozen result hashes are unchanged
(`1791816f…`, `f7e103f3…`, `217b09a7…`, `d966d98e…`).

Seal run (release, `--all-features`, clean tree at `9af5a1e`, version 0.13.0):
the **15-row `seal verify` matrix passes** at seal subject `793a07f9…`; 25 fresh
receipts; `court corpus` SUPPORTED (115/115 regenerated and hash-matched);
tests **407 passed / 12 ignored** all-features and **397 passed / 12 ignored**
default-features. No flagship performance claim exists yet.

### Seal 4 — flagship conventional baseline (2026-09-10)

**The box is opened.** `court conventional` no longer measures the H.2 entropy
fixtures; it measures the **frozen flagship population** and produces the first
precommitted flagship measurement. It is deliberately named the *flagship B0/B1
conventional-baseline result*, not the B1-vs-VOLE result.

What the court does:

* it verifies the frozen manifest **before** measuring anything (canonical
  object comparison, membership, order, derived B1 eligibility), then iterates
  the frozen order and requires each regenerated object to match its frozen
  canonical i32 hash;
* all 115 objects get `B0` and the `u1` literal figure; the 110 derived
  B1-comparable objects get FLAC levels 0/5/8 plus the optional reference
  oracle; the five `>8`-channel objects are recorded explicitly as
  `NOT_APPLICABLE_BY_FORMAT_DOMAIN` and never enter a B1 aggregate;
* the static result is bound to the population: manifest sha256, corpus sha256,
  object order, and every object's canonical i32 hash, rate, channels, frames
  and byte rows;
* the receipt reports per-axis surfaces (source structure, amplitude, channel
  structure, temporal, entropy, sample rate) over the B1-comparable subset.

Observed (release, `--all-features`, version 0.14.0):

```text
objects            115   (110 B1-comparable, 5 excluded by format domain)
B0 raw PCM          71,277,600 B
B1 level 5          25,577,431 B      B1(0) 29,901,976 B   B1(8) 24,808,874 B
u1 literal          71,283,810 B
B1 / u1 literal          0.359
B1 / B0 (comparable)     0.390
reference flac 1.5.0     25,319,928 B  reference/B1 0.990   (110/110 exact, non-authoritative)
exact round trips   330  (110 objects x levels 0/5/8)
frozen result       acfdaa32b9b69cdc0ee3ab2c0fb10387233603bc7a13d816575e12f7d2988b9b
corpus sha256       4c94b841…      manifest sha256 f67c73cf…
```

The stratification is the point: instead of one compression number, the frozen
pre-result axes separate cleanly under a *common* exact codec.

```text
entropy class          B1/B0        amplitude class   B1/B0
  highly predictable    0.195         low_byte          0.135
  locally predictable   0.241         s16_like          0.217
  sparse residual       0.286         s24_like          0.379
  globally periodic     0.310         full_i32          0.762
  spectrally struct.    0.653
  full-width random     0.898       channel structure  B1/B0
  scrambled             1.001         identical stereo  0.180
                                       anti-correlated   0.241
source structure       B1/B0          mono              0.379
  literal               0.032         multichannel      0.399
  exact repetition      0.171         correlated        0.415
  compound              0.206         independent       0.751
  oscillator            0.234
  residual              0.360
  wavetable             0.381
  noise                 0.844
```

The negative controls behave as designed (`scrambled` 1.001 — FLAC adds framing
to incompressible 32-bit noise; `full_width_random` 0.898 — the remaining gain
is the cross-channel structure the hostile population deliberately excludes).
The axes are **descriptive surfaces of this frozen population, not controlled
causal effects**: objects differ by rate and structure at once, so e.g. the
per-rate figures reflect which objects sit at each rate, not an isolated rate
effect.

Seal run: the **15-row `seal verify` matrix passes** at seal subject `40c4c8e6…`;
25 fresh receipts; tests **409 passed / 12 ignored** all-features and
**399 passed / 12 ignored** default-features.

**Not yet B1-vs-VOLE.** The `u1_literal_bytes` row is the canonical universal
*fallback*, not the inverse compiler's *selected* representation. The
selected-representation comparison needs an exact full-object inverse container
(deterministic segmentation to the 65,536-frame Phase-K ceiling, complete-cost
accounting of segment framing + index + container metadata, and an exact
total-extent reconstruction); it is the next increment, alongside B2–B4.

### Seal 5 — full-object archival container mechanism (2026-09-10)

Seal 5 freezes the **mechanism** before any flagship result exists. It is
deliberately *not* a new U1 `Representation` tag: the container is an
object-above-objects archive whose segments are ordinary U1 SampleObjects, so
Phase K is untouched.

```text
full frozen object
       │
       ▼
min(65,536, remaining) consecutive intrinsic ranges      ← frozen rule
       │
       ├── exact U1 SampleObject per segment
       └──
       │
       ▼
header ∥ segment index ∥ payloads ∥ integrity            ← real serialized bytes
```

- **Frozen segmentation.** 65,536 frames is inherited *verbatim* from the
  Phase-K observation ceiling (`MAX_INVERSE_FRAMES`), not tuned, and there are
  no content-adaptive boundaries.
- **Frozen selection.** The accepted exact candidate with the minimum Phase-K
  `complete_bytes`, ties broken by the deterministic proposal order. No
  measured quantity and no weighted score — storage economics decides.
- **Real bytes.** `complete_bytes = header.len() + index.len() +` Σ actual
  payload bytes `+ integrity`, with the payloads being the physical canonical
  serialization of each selected representation (entropy container for
  `literal`/`residual`, canonical object bytes otherwise).
- **Standalone pricing.** Every segment compiles against an empty reference
  library, so corpus-level deduplication can never flatter the comparison.
- **Semantics preserved.** The root carries the finite extent and the
  loop/one-shot identity; observation reproduces the finite extent and, for a
  loop root, the declared loop region past it.
- **One encoding, priced and stored.** `cost::best_literal`/`best_residual`
  iterate the frozen encoding universe once and serve both pricing and
  serialization, so selection and storage cannot disagree.

`court fullobj` is the mechanism gate. It runs 36 **non-flagship** fixtures over
the boundary lengths `1`, `65,535`, `65,536`, `65,537`, `131,072`, `131,073`
frames, mono/stereo/3-channel, silence / constant / exact-repeat / full-width
noise / mixed content, with `boundary - 1`, `boundary`, `boundary + 1`, a
window spanning each boundary, the last frame, and the loop continuation:

```text
fixtures                 36
boundary observations    414 exact
hostile containers       180 rejected (truncated/corrupted/version/ceiling/empty)
objects                  20 all-procedural, 8 all-literal, 8 mixed
selected kinds           constant 12, exact_repeat 19, literal 23, silence 13
every extent             reconstructed sample-for-sample; semantics preserved
frozen result           4b517ea0d564662d0b5c004434d1cea5b7358c5e1e563df5f6628a4665993a87
```

Seal run: seal subject `837653dc…`; tests **417 passed / 12 ignored**
all-features and **407 passed / 12 ignored** default-features. No flagship
performance claim exists yet; Seal 6 opens the second box.

### Seal 6 — B1 FLAC versus current bounded VOLE inverse selection (2026-09-10)

The second box is open. Every frozen object is compiled through the Seal-5
container and the 110 B1-comparable objects are priced against their FLAC
level-5 bytes.

**Claim boundary.** This is *B1 FLAC versus the current bounded VOLE inverse
selection* — not "optimal VOLE". The inverse compiler's proposal vocabulary is
deliberately limited (literal, silence, constant, exact-repeat, residual
zero/constant/periodic, shared reference), so a source frozen as `oscillator`
may legitimately compile to `exact_repeat`. Every segment is priced standalone
(empty reference library), exactly as B1 is priced standalone per file.

Binding: the court verifies the frozen manifest before measuring, recomputes B1
in-process, and requires the total to equal the sealed `court conventional`
total (`25,577,431 B`), so the two courts cannot drift. Frozen search budget:
the default bounded search (`max_period_scan` 512, `max_residual_period_candidates`
4, `max_candidates` 64, `Auto` placement).

```text
objects                  115   (110 B1-comparable, 5 format-domain excluded)
segments                 255
B0 raw PCM               71,277,600 B
u1 literal equivalent    71,283,810 B
B1 FLAC level 5          25,577,431 B
VOLE (container)         31,118,702 B
B1 / VOLE                    0.822     VOLE = 1.217x B1 in aggregate
VOLE cheaper / equal / larger than B1      55 / 0 / 55   (within 1%: 15)
objects all-literal / all-procedural / mixed   61 / 49 / 5
selected kinds           literal 133, exact_repeat 103, constant 10,
                         silence 4, residual_zero 3, residual_constant 2
every extent             reconstructed sample-for-sample; semantics preserved
frozen result           76fe5dcff1dec0a173ed67dc69effcc6befc093e340519c81ec2b32ea3b1a17b
```

**Superseded for the aggregate only (Seal 7).** This seal's `VOLE (container)`
total and `B1 / VOLE` ratio mixed populations — all 115 objects for VOLE versus
the 110 B1-comparable for B1. The correct comparable-population figures are
**VOLE 28,257,411 B, B1 / VOLE 0.905 (VOLE 1.105× B1)**. The per-object vector,
the 55/0/55 buckets and every per-axis surface in this receipt were already
computed over the comparable population and stand unchanged.

The aggregate hides the shape, which is the point of reporting the buckets and
the axes. Under a *common* exact test the current compiler wins decisively where
its vocabulary fits and loses where it does not:

```text
B1 / VOLE                 B1 / VOLE                 B1 / VOLE
source literal    51.55   ch identical st.   8.97   entropy globally per.  3.05
source exact_rep   7.08   amp full_i32       1.09   entropy highly pred.   1.64
source wavetable   2.70   amp s16_like       0.71   entropy scrambled      1.00
source oscillator  1.53   ch mono            0.96   entropy full-width rnd 0.91
source residual    0.56   ch correlated      0.46   entropy sparse resid.  0.58
source compound    0.48   ch independent     0.92   entropy locally pred.  0.59
source noise       0.88   ch multichannel    0.76   entropy spectrally st. 0.74
```

(`B1 / VOLE` > 1 means VOLE stores fewer bytes.) VOLE wins on 55 of 110
comparable objects and loses on 55; the aggregate loss is driven by classes the
current vocabulary cannot yet model — compound material, residual-governed
content the periodic model does not capture, independent multichannel, and
full-width random content where both systems pay framing on noise.

Seal run: seal subject `ef8a57d1…`; 26 fresh receipts; the 16-row `seal verify`
matrix passes; tests **417 passed / 12 ignored** all-features and **407 passed /
12 ignored** default-features.

### Seal 7 — flagship aggregate + container-integrity review closure (2026-09-10)

One narrow review closure before B2–B4. **Nothing about the experiment changed**:
no corpus, segmentation rule, search budget, candidate vocabulary or selected
representation was touched, and both frozen static result hashes survive
(`4b517ea0…` container, `76fe5dcf…` flagship).

**1. Population arithmetic (the headline was wrong).** `court flagship`
accumulated `vole_total` over all 115 objects while `b1_total` covered only the
110 B1-comparable objects, so the published headline compared different
populations. Every total is now tracked for both populations, and every B1 ratio
uses the comparable population only. Corrected headline over the **same 110
objects**:

```text
                     all 115        B1-comparable 110
B0                   71,277,600 B        65,517,600 B
literal equivalent   71,283,810 B        65,523,540 B
B1 FLAC-5                    —           25,577,431 B
VOLE complete        31,118,702 B        28,257,411 B
B1 / VOLE                    —                0.905
VOLE / B1                    —               1.105x
```

The corrected result is *stronger* for VOLE: within the same format-comparable
population the current bounded compiler is about **10.5% larger than FLAC-5 in
aggregate**, not 21.7%. The 55/0/55 bucket result, the per-object vector and the
per-axis surfaces were already computed over the comparable population and are
unchanged.

**2. The hostile battery reached less validation than it claimed.** The old
checks mutated structural fields *without* recomputing the trailing digest, so
almost every case proved only "the digest caught it". There are now two classes:

```text
integrity-hostile      mutate, do NOT reseal   -> must fail the outer digest
structurally hostile   mutate AND reseal       -> must reach and fail the
                                                  format validators
```

Seal run reports **144 integrity-hostile** and **432 resealed structural-hostile**
rejections (12 structural mutations × 36 fixtures), so the parser's own
invariants are now actually exercised.

**3. The index could lie about `content_id`.** Materialization never checked it,
so a validly re-sealed container could carry a wrong content identity.
Materialization now reconstructs the semantic U1 object of each segment and
requires `canonical_content_id(descriptor, data) == index.content_id` (for
literal/residual segments this is the *derived semantic* identity, not the
physical entropy-container hash).

**4. Candidate-tag/representation compatibility** is enforced on decode, so
`silence` + `residual_periodic` is rejected rather than decoded.

**5. Canonical payload layout.** Offsets must be the serializer's contiguous
block — first at the payload start, each next at the previous end, final end at
the body end — rejecting gaps, overlaps, aliases and unreferenced trailing
bytes.

Because the serializer itself was already canonical, no stored bytes moved, and
both frozen hashes are preserved. Seal run: seal subject `0e30bdcc…`; tests
**418 passed / 12 ignored** all-features and **408 passed / 12 ignored**
default-features.

*(Seal 8 supersedes the VOLE totals in this seal after the complete-cost framing
fix; see below.)*

### Seal 8 — entropy complete-cost physical framing (2026-09-10)

The full-object layer did exactly what it was built to do: once "accounted
bytes" and "actual bytes" existed side by side, the mismatch became observable.

**Root cause.** `RepresentedLiteral::cost`/`RepresentedResidual::cost` omitted
serialized framing — the container prefix/pool-count/page-count shortfall and
every block's 32-byte header plus its one-byte integrity flag (and optional
digest). An all-RANS literal under-reported by exactly `7 + 33 × rans_blocks`,
which let an entropy `Literal` win a Phase-K selection on a discounted price and
then store a larger physical artifact. `docs/ENTROPY_ACCOUNTING.md` already
required block headers, indexes and integrity in the complete cost; the
implementation did not match its own normative document.

**Fix.** The cost decomposition now sums to exactly the canonical serialized
length — `CompleteCost::complete_bytes == serialized_bytes()` — with every
physical byte given a home:

```text
metadata      container prefix + shared-model pool count
hypothesis    semantic model (residual)
model         pool models + inline models + per-block shared refs
payload       encoded bodies
index         page-count field + page index records
integrity     per-block header + integrity flag + optional digest
```

`serialized_bytes()` is public on both representations; `cost()` carries a debug
invariant of equality; the block framing constants are public. An invariant
battery sweeps every literal symbolization × page size × model mode × integrity
× 1/2 channels and every residual page size, and the full-object layer now
enforces `segment.objective_bytes == segment.stored_bytes` (in
`compile_full_object` and in `court fullobj`). `court entropy-literal` and
`court entropy-residual` require cost == serialized/container bytes. The frozen
reviewer regression (`old objective 3115` / `old physical 7082`,
`7 + 33 × 120 = 3967`) can no longer recur.

**Selection rerun** (nothing else changed — corpus, segmentation, search budget
and candidate vocabulary are untouched):

```text
court inverse    6 -> 7 non-literal explanations        hash 5b836006…
court fullobj    same fixture selections                hash 0969a4f4…
court flagship   VOLE comparable 28,257,411 -> 28,228,300 B
                 B1/VOLE 0.905 -> 0.906 (VOLE 1.104x B1)
                 buckets 55/0/55 -> 57/0/53; within 1% 16
                 all-literal/procedural/mixed 60/52/3
                 kinds literal 128, exact_repeat 104, constant 10,
                 residual_zero 8, silence 4, residual_constant 1
                 hash 8f37fab0…
```

As expected, the corrected physical total is **≤** the previous one (the old
selected candidate remains available; the objective can only become more
honest), and a handful of segments switched representation. Corpus hashes, the
B1 conventional result, semantic behaviour and device artifacts are untouched.
Seal run: seal subject `bb49eb48…`; tests **419 passed / 12 ignored**
all-features and **409 passed / 12 ignored** default-features.

### Seal 9 — runtime mechanism freeze (2026-09-10)

This seal freezes the **runtime protocol and correctness vector** before any
comparative timing exists. One contract, four source architectures, all fed the
same frozen sequential trace and the same caller-owned destination:

```text
B2  PCM-resident sampler          canonical i32 held in memory
B3  raw PCM disk streaming        canonical LE i32 on disk, cold/warm VERIFIED
B4  compressed preload            the exact B1 FLAC-5 artifact, decoded once
B5  bounded VOLE materialization  verified full-object container, page-bounded
```

**Shared-model pool made transactional (H.2 cleanup).** The per-page RANS/RAW
decision in `RepresentedLiteral`/`RepresentedResidual` previously compared block
bytes only, so a page could add shared models to the pool and then fall back to
RAW, leaving orphan models behind that it never paid for. The decision now
includes the **marginal** pool bytes (`choose_page_kind(rans_block_bytes,
marginal_pool_bytes, raw_bytes)`), and a RAW fallback truncates the pool to its
mark. Two regression tests (`page_kind_decision_includes_marginal_pool_cost`,
`shared_pool_is_transactional_with_no_orphan_models`) require the pool to return
exactly to its previous state. This does not affect Seal 8's result, whose
inverse path is `ModelMode::Inline`.

**One artifact for B1 and B4.** `b1_flac_artifact()` returns
`FlacArtifact { bytes, encoding, sha256 }` after the mandatory exact-roundtrip
and STREAMINFO MD5 checks; `b1_flac()` is now a thin compatibility wrapper
returning `.encoding`. B1 and B4 therefore consume the **same** exactly-verified
stream and cannot construct subtly different FLAC files. The sealed B1 stream has
no SEEKTABLE and is never modified to gain one, so B4 is truthfully a
*compressed-storage / decoded-resident* sampler; setup (decode) is reported
separately from steady-state reads.

**Verification outside the hot path.** `VerifiedFullObject::verify(bytes)` runs
once — parse, outer digest, frozen layout, per-segment `content_id` binding to the
reconstructed semantic object, and root semantics — and records `verify_ns`.
`FullObjectReader` then services `[start, frames)` by parsing each segment once
(lazily) and decoding only the pages the window touches through the frozen
`RepresentedLiteral::materialize` / `RepresentedResidual::materialize_closure`
paths; it retains encoded state only, never a decoded waveform. Windows crossing
the 65,536-frame segment boundary concatenate seamlessly.

**Cache claims are verified, not assumed.** Linux documents
`posix_fadvise(DONTNEED)` as an attempt, so eviction is paired with a
`mincore(2)` residency check over the page-aligned mapping; when residency cannot
be established the state is `CACHE_STATE_NOT_CONFIRMED`, never labelled cold or
warm. Global `drop_caches` is never used. `B3` artifacts live on a block-backed
filesystem (`target/`) because `fadvise` cannot evict tmpfs pages — `/tmp` was
tried first and correctly failed to reach cold. Logical (`rchar`) and physical
(`read_bytes`) traffic come from `/proc/self/io`.

**The frozen trace.** Sequential `[0,512), [512,1024), …` plus a final partial
window, at each object's native rate and channel count, with no gain, pan, filter,
resampling or random seek. Deadlines follow the native rate (512 frames @ 48 kHz
= 10,666,666 ns; @ 44.1 kHz = 11,609,977 ns; @ 96 kHz = 5,333,333 ns;
@ 192 kHz = 2,666,666 ns; asserted in a unit test). This court measures **source
materialization**, not endpoint behaviour.

`court runtime` runs all 115 frozen objects (B4 for the 110 B1-comparable
objects and `NOT_APPLICABLE_BY_FORMAT_DOMAIN` otherwise): **23,229 trace windows
per source, every requested window exact on every source**;
`COLD_VERIFIED 115`, `WARM_VERIFIED 115`; frozen static result
`a66774555f11bd9506ccfbf1f14e9e4404e6fdc8aa8478518cdee3c705d5ea84`. The hash
covers the protocol, artifact identities, deterministic counters and the
correctness vector; measured latency and physical storage-read traffic are
recorded as evidence but deliberately excluded from it. Informational (not
frozen) totals from the seal run:

```text
B2        total   1.4 ms   max   1.6 µs   storage reads 0
B3-cold   total  55.1 ms   max   1.7 ms   storage reads 71,462,912 B (logical 71,277,600)
B3-warm   total   5.7 ms   max  14.5 µs   storage reads 0
B4        total   0.9 ms   max   1.3 µs   storage reads 0
B5        total 480.7 ms   max   1.3 ms   encoded examined 60,750,412 B, parsed 31,063,791 B,
                                           pages 12,062, segments 23,229
deadline misses: 0 on every source
```

No comparative headline is drawn here. B2/B3/B4/B5 measurement — repeated
traversals with deterministically rotated source order, raw per-quantum
latencies, derived distributions and the stratified crossover table — is
**Seal 10**. Frozen hashes are unchanged from Seal 8 (semantic `1791816f…`,
authored `f7e103f3…`, inverse `5b836006…`, inverse-search `d34708c3…`,
conventional `acfdaa32…`, fullobj `0969a4f4…`, flagship `8f37fab0…`, corpus
`4c94b841…`, manifest `f67c73cf…`, PTX `d13d22c3…`, AMDGPU `5c30a4bc…`). Seal
run: seal subject `1d27510c…`; the **18-row** `seal verify` matrix passes; tests
**426 passed / 12 ignored** all-features and **416 passed / 12 ignored**
default-features.

**Documentation correction.** `court inverse-search` reports `d34708c3…`, and has
done so since Seal 8 — the complete-cost oracle changed candidate pricing, which
the court's static result includes. Prose that carried `d966d98e…` as the current
value was repeating the pre-Seal-8 number (Seal 7 and earlier, where it was
correct); Seal 9 itself changed nothing. Every Seal-9 receipt was compared
field-for-field against its Seal-8 predecessor and all 27 pre-existing courts are
identical.

### Seal 10 — runtime measurement protocol (2026-09-10)

The Seal-9 mechanism was sound but its **measurement instrumentation was not
comparable**, and the review found four things to close before any official
runtime number could be reported.

**1. The latency boundary.** B3 previously allocated a byte buffer and read
`/proc/self/io` *before* starting its stopwatch, stopped it after
`read_exact_at()`, and only then converted bytes to `i32` — so B3 was effectively
timed as “kernel file read” while B2/B4 timed their copy to `dst` and B5 timed its
materialization. Latency is now **harness-owned**: the stopwatch wraps the whole
`RuntimeSource::read` call in `run_traversal`, and `ReadEvidence` no longer has
timing fields at all, so a source cannot choose which part of its work counts.
`/proc/self/io` moved to traversal boundaries, outside every timer. A timer
calibration is reported (`timer_overhead_ns`, min 10 ns / median 20 ns) so
sub-100 ns rows are read as near the apparatus floor.

**2. Residency accounting.** `SourceInfo::persistent_sample_domain_bytes`
counted B3's whole on-disk file as “resident in the process”. It is now split:

```text
artifact_storage_bytes       bytes the artifact occupies in storage
resident_sample_domain_bytes PCM persistently resident in the process
resident_encoded_bytes       encoded bytes persistently resident
```

So B3 has storage 71,277,600 B and **zero** resident PCM; B4 holds its decoded
PCM resident but not its FLAC artifact; B5 holds its container as resident
*encoded* bytes and zero resident PCM.

**3. Setup separation.** B3's `prepare` serialized, wrote, `sync_all`ed and
opened the file inside `setup_ns`, so B3's setup included authoring while B4/B5's
did not. `DiskPcmArtifact::create` (authoring) is now separate from
`DiskPcmSource::open` (runtime setup), and artifact build / verification /
runtime setup are reported as distinct quantities.

**4. Repeat, state and population.** The aggregate counted only B2's windows and
labelled them “per source”. Window counts are now per source; **every aggregate
is computed inside an explicit population** (all 115 with B2/B3/B5/B5-prepared,
and the 110 B1-domain objects adding B4), and no ratio mixes them. The trace is
repeated 3 times with **deterministically rotated source order**; each repeat
starts from equivalent state (B3 re-verifies cold/warm; B5 first-play gets a
fresh bounded reader). B3 eligibility is per repeat: only `COLD_VERIFIED` /
`WARM_VERIFIED` repeats count. B5 first-play is primary; a **prepared** control
(all segments parsed before measurement) is reported separately and never
averaged with it. Timed correctness is checked against window digests
precomputed during preparation, so verifying a window never walks the canonical
vector.

**Frozen protocol.** `PROTOCOL_SCHEMA = vole.audio.runtime.protocol.v2`;
quantum 512; 3 repeats; rotated source order; no gain/pan/filter/resample/seek;
native rate and channels. Static result:

```text
d7d11681141f5f92baf1a22d572b40d5cb7e1813f902d2f5067a2e1dbd54ba79
```

covering protocol, artifact identities, deterministic counters and the
correctness vector — 115 objects, 23,229 windows per traversal, 2,055
traversals, every window exact; latency, physical storage traffic and setup
times are measured evidence excluded from it and written to a trace under
`receipts/traces/`.

**Measured** (pooled over all 115 objects; p50 / p99 / max in ns):

```text
B2           120 / 701 / 29095        resident PCM 71.28 MB, storage 0
B3-cold      141 / 27572 / 1545604    storage 71.28 MB, resident 0, disk reads 214 MB (3x object)
B3-warm      141 / 632 / 28904        storage 71.28 MB, resident 0, disk reads 0
B4           60 / 511 / 85150         storage 25.58 MB, resident PCM 65.52 MB, setup 323 ms
B5           912 / 305774 / 1283111   storage 31.09 MB, resident encoded 31.09 MB, 0 resident PCM
B5-prepared  911 / 305614 / 1336461   same, setup 2.25 ms (all segments parsed before measurement)
```

0 deadline misses on every source. Over the 110 B1-domain objects B5 stores
28,228,300 B against B4's 25,577,431 B — a byte ratio of **1.104×**, the same
figure the flagship court reports. Crossover surfaces (p50 and stored bytes per
frozen axis) are in the receipt; the structure is emphatic — B5 is 15× B4 at
p50 overall but stores **0.019×** B4 for `literal`, **0.11×** for
`identical_stereo`, **0.33×** for `globally_periodic`, while paying ~200–900×
B2's p50 on `correlated_stereo`, `multichannel` and residual/transient classes.
The prepared control's p50 is within 0.2% of first-play, so at the median the
lazy per-segment parse is not the cost — page materialization is — and the parse
cost appears in setup (2.25 ms vs 8 µs), not the tail.

Seal 9's `a6677455…` receipt is retained unchanged as historical mechanism
evidence; this seal supersedes its residency/window-count/timing instrumentation.
That is the **only** row whose hash changed: every other court's Seal-10 receipt
was compared field-for-field against its Seal-9 predecessor and is identical.
Seal run: seal subject `3e7e6e2a…`; tests **430 passed / 12 ignored**
all-features and **420 passed / 12 ignored** default-features.

**Real-time load, depth, random-access, interference, long run and energy are
not measured here** — see “What is *not* in this increment”.

### Seal 11 — remaining Phase-M courts (2026-09-10)

With the runtime substrate measured, the rest of the Phase-M court set lands.

**`court random-access`.** The runtime court measures the sequential trace; this
measures the other half. Deterministic per-object random windows (seeded from
the canonical hash, never from measurement) always include the first frame, the
last frame, a window straddling the 65,536-frame container boundary and a final
full quantum, then pseudo-random windows of 1/64/256/512 frames, repeated with
rotated source order and compared against a sequential pass. Five rows: B2
resident PCM, B3-warm, B4 decoded-resident, **B4-seek** (the same artifact via
stateless `decode_seek` — the sealed stream has no SEEKTABLE, so each seek
decodes forward from the first frame) and B5 bounded VOLE. New
`runtime::FlacSeekSource`. Frozen static result
`d9259f18a4fdc731c34ac7cc1106bd612a0b993df4c7ba6ca473803ad527132f`.

**`court negative`.** The incompressible controls (entropy classes
`full_width_random` and `scrambled`) measured against B0/B1/VOLE with every
ratio recomputed from the stored bytes — **VOLE 11,248,166 B vs B0 11,348,400 B
vs B1 10,046,098 B** over 20 objects, i.e. neither codec compresses
incompressible material and VOLE does not lose to raw framing there (0.991×) but
does lose to FLAC (1.12×). Plus a hostile-archive battery (truncation, bit flip,
**resealed** structural mutation, allocation bomb, garbage) run under
`catch_unwind`: **21 candidates rejected with a typed error, 0 panics**. Frozen
result `6994a97a…`.

**`court depth`.** Minimum stable observation depth from measured per-window
latencies at quanta 64/128/256/512/1024: the smallest buffered lookahead `k`
with `completion[k-1] >= max_i(completion[i] - i*deadline)`. On this idle host
the worst depth over all 115 objects is **1 quantum** at every quantum and for
every source — production is far faster than realtime — with the first
non-trivial sign at B5's 64-frame quantum (1.33 ms deadline), where **3 windows
miss their individual deadline** without yet forcing a deeper prefill. Frozen
result `3e616cde…`.

**`court interference`** (contract §49). The frozen workload under idle,
`cpu_burn` (one spinner per logical CPU), `memory_bandwidth` and `storage_io`,
with idle-relative tails and per-condition depth:

```text
condition          B2      B3-warm   B4      B5      (p50 ns / idle ratio)
idle               40      150       40      911
cpu_burn           50 1.25 200 1.33  50 1.25 1723 1.89
memory_bandwidth   822 20.6 2385 15.9 762 19.1 2915 3.20
storage_io         40 1.00 150 1.00  40 1.00 911 1.00
```

Memory bandwidth is the dominant interference (B2/B4 ~19–21×, B3-warm ~16×, B5
~3.2× — B5 is already materialization-bound); CPU contention ~1.3–1.9×; storage
pressure does not move the warm-cache sources. One B3-warm window misses its
deadline under memory pressure. A **bounded soak** under CPU contention runs
121 passes with **0 deadline misses** and drift 1.0. `energy` is
`NOT_AVAILABLE` (no hwmon power input, no NVML/AMDSMI — no figure is invented),
and seven contract conditions this host cannot manipulate are reported
`NOT_CONTROLLED`. New `runtime::load` and `runtime::energy`. Frozen result
`9a5777bc…`.

**`court all`.** The Phase-M aggregate runs conventional, corpus, fullobj,
flagship, runtime, random-access, negative, depth and interference in sequence
and is `SUPPORTED` when all nine are (they are). Frozen result `c3d4a47c…`.

Shared `courts::measure` now owns the traversal runner, the timer calibration
and the depth computation, so every measurement court uses one latency boundary.
Every pre-existing court's receipt is field-for-field identical to Seal 10.
Seal run: seal subject `6b000562…`; the **23-row** `seal verify` matrix passes;
tests **437 passed / 12 ignored** all-features and **427 passed / 12 ignored**
default-features.

*Superseded by Seal 12 for the affected evidence (Git history and receipts are
preserved unchanged): the `court negative` aggregate above mixes the 20-object
B0/VOLE population with the 19-object B1 population, so “VOLE … vs B1 …” is not
an apples-to-apples ratio; `court interference`'s frozen cells repeat a
corpus-level accumulator rather than per-object geometry; the soak is a
CPU-contention soak; and `court random-access`'s frozen vector records
`exact: false` and zero-window “MEASURED” rows for the five >8-channel objects.*

### Seal 12 — evidence-contract closure (2026-09-10)

A narrow closure. No data path changed; six evidence-contract defects are fixed.

**`court random-access`.** `Acc` derived `Default`, so `exact` started `false`
and `&= t.exact` could never recover it: the independent court-level gate still
passed, but every cell recorded `"exact": false` and the frozen hash bound that
false value. It is now constructed only through `Acc::measured()` /
`Acc::not_applicable()` with `exact: true` (and a `started` flag for the first
record). The five `>8`-channel objects now serialize B4/B4-seek as
`NOT_APPLICABLE_BY_FORMAT_DOMAIN` (`exact: null`) instead of zero-window
“MEASURED”, and the pooled surface reports each source's own window total
(B2/B3-warm/B5 7806; B4/B4-seek 7471). Resealed: `3a76aef7…`. B4-seek's random
p50 is **1.45 ms** — the honest cost of stateless seeking on a no-SEEKTABLE
stream, against 131 ns for B4's resident path.

**`court negative`.** The 12-channel `noise-stress` full-width-random object is
outside FLAC's domain, so B0/VOLE covered 20 objects while B1 covered 19 and the
reported `vole/B1` mixed populations. Two explicit populations are now formed:

```text
all (20 objects)          B0 11,348,400 B   VOLE 11,248,166 B   (VOLE/B0 0.991)
b1_comparable (19)        B0 10,772,400 B   B1 10,046,098 B      VOLE 10,671,752 B
                          VOLE/B0 0.991     VOLE/B1 1.062
```

The honest figure is **VOLE/B1 1.062**, not the mixed 1.12. Every B1 ratio is now
formed only inside the comparable population. Resealed: `cca0f162…`.

**`court interference`.** The frozen cells were built from one corpus-level
accumulator per (condition, source) copied into all 115 object cells, so they
repeated aggregate state (the first object's window count) rather than binding
each object's own traversal geometry. Per-object deterministic records
(`ObjectCond { exact, windows, deadline_ns }`) are now captured during
measurement and the static projection is built from them; the pooled latency
matrix remains the measured evidence. Resealed: `f23c70c1…`. The bounded soak is
now described as what it is — a **CPU-contention soak** — rather than “the
heaviest controllable load”, which this court's own matrix contradicts (memory
bandwidth is more disruptive); the soak workload was not changed after seeing
the result.

**`court depth`.** The finite prefill search always resolves at `k = n` (the
whole object is prefetched), so an `UNSTABLE` state was unreachable for a
non-empty trace and a producer slower than real-time could still read
“stable, depth=n”. Two quantities are now reported: the
`finite_object_min_prefill_quanta` (the escape it is) and `streaming_stability`,
which marks `UNSTABLE_STREAMING` when average production exceeds the deadline
(`sum(latency) > n * deadline`). The measured result is unaffected — depth 1
everywhere, with B5's 64-frame quantum still the first to miss individual
deadlines. Frozen result unchanged (`3e616cde…`) because the projection holds
request geometry and exactness only; depth is evidence.

**Energy.** The probe claimed to have searched NVML/AMDSMI and never did; it is
now described as what it probes (a hwmon instantaneous power source) and a
present-but-unreadable powercap counter is not reported as available. Cumulative
`energy_uj` (powercap) is the supported instrument for joules over an interval,
with wraparound handled against `max_energy_range_uj`; a hwmon spot reading is
recorded as evidence and never converted into workload energy. On this host
`energy_uj` exists but is root-only, so energy is truthfully `NOT_AVAILABLE`.

**`court runtime`.** B4's `artifact_build_ns` was set from `runtime_setup_ns`
(the decode/preload) rather than the B1 encode; it now reports the FLAC
`encode_ns`. Frozen result unchanged (`d7d11681…`) — the field is measured
evidence, not part of the static projection.

Seal run: seal subject `2a9ce87e…`; the **23-row** `seal verify` matrix passes;
tests **439 passed / 12 ignored** all-features and **429 passed / 12 ignored**
default-features. Random-access, negative and interference are resealed; every
other court's receipt is field-for-field identical to Seal 11.

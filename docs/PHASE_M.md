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
B2 PCM-resident sampler                NOT_IMPLEMENTED   Phase M
B3 disk-streaming sampler              NOT_IMPLEMENTED   Phase M (must record read
                                                         traffic, cache state, underruns)
B4 compressed-file decode + playback   NOT_IMPLEMENTED   Phase M
B5 VOLE scalar                         MEASURED_ELSEWHERE court semantic/facts/inverse
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

Stated plainly so the gap is visible (updated at Seal 6 — the corpus is frozen,
the conventional baseline is measured, the container mechanism is frozen, and
the true B1-vs-VOLE result now exists):

* B2–B4 (sampler / disk-streaming / compressed-file playback) are
  `NOT_IMPLEMENTED`; now that a canonical full-object VOLE artifact exists, the
  runtime ladder (B2 / B3 / B4 / B5 selected full-object VOLE scalar) becomes
  comparable rather than fragmented;
* `court depth`, `court random-access`, `court negative`, `court interference`,
  `court all` are not implemented;
* energy and the adversarial real-time load matrix (§49) are not measured;
* the **license-clean real-recording stratum is declared and vacant** (see
  `corpus/README.md`); no production claim rests on real recordings yet, and
  `court corpus` records that rather than staying silent.

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

1. **Seals 2–6 — the corpus is frozen, verified and review-closed; the flagship
   B0/B1 conventional baseline is measured; the full-object container mechanism
   is frozen; and the true B1-vs-VOLE result exists.**
2. B2–B4 (PCM-resident / disk-streaming / compressed-file playback) beside the
   selected full-object VOLE artifact, then the `depth` / `random-access` /
   `negative` courts and the crossover surface.
3. Adversarial real-time load (§49) and energy where measurable.

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

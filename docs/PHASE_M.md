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

Stated plainly so the gap is visible:

* the **flagship corpus is not frozen yet**. This increment measures the frozen
  **H.2 entropy corpus** (14 generated fixtures). Its negative controls are built
  for 8-bit symbolization and are not all incompressible at 32 bits/sample (for
  example `random-control` is 2-channel with identical channels, so mid-side
  decorrelation halves it); the flagship corpus must be stratified by
  representation *and* amplitude class;
* B2–B4 (sampler / disk-streaming / compressed-file playback) are
  `NOT_IMPLEMENTED`;
* `court depth`, `court random-access`, `court negative`, `court interference`,
  `court all` are not implemented;
* energy and the adversarial real-time load matrix (§49) are not measured.

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

1. **Freeze the flagship corpus** (~100 objects, several minutes, 44.1/48/96/192
   kHz, one-shots/loops/reverse/rate/ADSR/pan/automation/bounded filters,
   stratified by representation and amplitude), with a manifest and
   `corpus verify`.
2. B2–B4 and the `depth` / `random-access` / `negative` courts.
3. The crossover surface: `court all` stratified by representation.
4. Adversarial real-time load (§49) and energy where measurable.

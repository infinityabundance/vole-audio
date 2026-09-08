# PROJECT STATE — exact status ledger

> The ledger. Updated at every phase boundary. This file tells the truth about
> what exists, what is evidenced, what is blocked, and what is next. It is the
> first file to read after the README.

## Architectural invariant

```
A SampleObject is the authoritative media object.
PCM/sample-domain values are observation surfaces.
Literal representation is valid and mandatory as a universal fallback.
Sample-domain materialization is bounded and attributable.
CPU/GPU semantics are identical (same shared semantic core).
Directness is measured and proven or rejected per hardware.
```

## Toolchain & environment anchor (captured at Phase A)

- rustc `1.99.0-nightly (89c61a754 2026-07-23)` pinned in `rust-toolchain.toml`
  (channel `nightly-2026-07-24`).
- Host: CachyOS, Linux `7.2.2-1-cachyos`, AMD Ryzen 7 9800X3D (16 threads,
  AVX2 + AVX-512), 125 GiB RAM, governor `performance`.
- GPU: NVIDIA at PCI `0000:01:00.0` (vendor `0x10de`, driver `nvidia`,
  RTX 4080-class; CUDA UMD 13.3, driver 610.57.04; toolkit at `/opt/cuda`).
- Audio endpoints (ALSA): card 0 `Onyx Artist 1-2` USB (S16_LE/S32_LE,
  MMAP_INTERLEAVED) — primary D1 candidate; card 1 NVIDIA HDMI; card 2
  ALC897 analog (`snd_hda_intel`).
- ROCm: **not installed**; `amdgcn-amd-amdhsa` court = hardware-unavailable
  evidence (never manufactured).
- Kernel device targets verified: `nvptx64-nvidia-cuda` (PTX, `extern
  "ptx-kernel"` + `feature(abi_ptx)`; needs `llvm-bitcode-linker`);
  `amdgcn-amd-amdhsa` (`extern "gpu-kernel"` + `feature(abi_gpu_kernel)`,
  HSA/ELF shared object output, `-Z build-std=core` from `rust-src`).

## Completed phases

### Phase A — Evidence constitution (complete)

_... see git history / earlier ledger entries ..._ (Phase A summary retained in
`PROJECT_STATE` at commit `efaebf7`.)

### Phase B — vole.audio.u1 (complete)

Exit criteria: frozen reference vectors; `U1_SPEC.md` matches code; unit
tests enforce the freeze.

Delivered (`src/universe/`, all `no_std`, device-compiled):

- `u1.rs` — universe/profile identity (`vole.audio.u1`, `u1/v1`).
- `sample.rs` — i32 canonical code domain; exact u8/s16/s24/s32 ingest.
- `time.rs` — typed frame coordinates (MediaFrame/ObjectFrame/EndpointFrame),
  `EpochId`, `NominalRate`; no f64 media time.
- `clock.rs` — `XrunPolicy` vocabulary; integer host frame<->time rounding.
- `arithmetic.rs` — Q24 positions/rates, Q16 multipliers, round-half-away
  (`rnd_shift`), `sat_i32`, `mul_q16`, `lerp_i32`, mixing-bound proof test.
- `phase.rs` — u64-mod-2^64 oscillator phase, exact 64-bit `freq_to_incr`
  (u128 only in test oracle), 12-bit index + 24-bit frac addressing,
  Q30 sine table freeze.
- `prng.rs` — VOLE-SPLITMIX64-STREAM (per-frame noise, random-access pure),
  VOLE-XOSHIRO256STARSTAR-1 with splitmix64 seeding + jump.
- `event.rs` — total order (frame, class priority, sequence); priority table.
- `layout.rs` — Mono/Stereo/N channels; canonical interleave.
- `observation.rs` — canonical interleaved LE-i32 form; streaming
  `observation_sha256` (alloc-free).
- Frozen asset `assets/u1/sine_q30_4096.bin` (4096 x i32 LE, Q30,
  structurally symmetric, SHA-256 pinned in spec + test).
- `docs/U1_SPEC.md` rewritten as the normative freeze.

Verified: 70 unit tests pass (incl. FIPS vectors, i128 oracle sweeps, u128
increment oracle, symmetry + hash pin); `clippy -D warnings` clean; `fmt`
clean; library still compiles `no_std` for `nvptx64-nvidia-cuda`.

### Phase C — Scalar oracle (complete)

Exit criteria: SampleObject model, voices, triggering, loop/reverse/rate,
gain/pan, ADSR, mix, interpolation, narrow filter, observation ranges;
deterministic repeated hashes; chunked == contiguous; seek == sequential;
reference vectors frozen; `court semantic` executable.

Delivered:

- `object/` — SampleObject (descriptor + content identity + literal +
  referenced + dependency-graph validation with depth/cycle bounds, hostile
  ceilings enforced), canonical per-representation byte forms.
- `sampler/` (pure semantics, no_std): analytic piecewise-linear ADSR;
  Q16 gain chain (unity-identity, +6 dB ceiling); linear equal-gain pan with
  exact `L + R == unity`; signed-Q24 rate/position + Euclidean loop wrap
  (forward = reverse tape) + exact one-shot end frames; linear/nearest reads
  with frozen continuation rules; i64 per-(frame,channel) mixer with one
  final saturation; narrow exact filter set (integer-coefficient Butterworth
  biquad from the sine table, one-pole smoother) with explicit state.
- `sampler/voice.rs|world.rs|scheduler.rs` (host): immutable voice specs,
  functional world (voices = spawn records), validated timeline assembly.
- `sampler/resampler.rs` — frozen 64×1024 polyphase FIR: asset
  `assets/u1/resampler_bh64_p1024_q15.bin`, exact-DC rows, measured
  passband ripple ≤ 1.5e-4 / stopband ≥ 210 dB, hash pinned.
- `eval/scalar.rs` — the scalar oracle (semantic authority).
- `court semantic` — first executable court: repeated-hash, chunked ==
  contiguous, seek == sequential, hostile-note-off rejection; emits immutable
  receipts; reference hash frozen and enforced by test.
- CLI `court` + `receipt show` wired.
- U1_SPEC updated: sampler transforms (§12), resampler freeze (§9),
  Phase C reference vectors (§11).

Verified: 136 tests green; `clippy -D warnings` clean; `fmt` clean;
`court semantic` SUPPORTED with receipt
`receipts/semantic/semantic-*.json`.

### Phase D — Procedural objects (complete)

Exit criteria: authored objects observe without resident full-object PCM.

Delivered:

- `object/simple.rs` — Silence / Constant / Noise payloads (endless, extent 0,
  canonical bytes; representation tag `Noise = 0x0C` added to the frozen
  taxonomy).
- `object/wavetable.rs` — `Cycle` payload for `Wavetable`/`SingleCycle`/
  `ExactRepeat` (resident-cycle tables, always-wrap playback, table bytes
  bounded by `MAX_TABLE_BYTES`).
- `object/oscillator.rs` — `Oscillator` (base freq + Q16 amp) and
  `PartialBank` (ascending harmonic list) payloads with domain validation
  and canonical bytes.
- `sampler/procedural.rs` (no_std, device-shared) — frozen generator
  semantics: `eff_incr` (exact, clamped, i128 host), modulo-2^64 phase
  accumulation, sine-table oscillator sample, i64-accumulate + single-
  saturation partial bank, partial validation.
- `ResolvedVoice` reworked around play-source classes (literal content,
  cycle content, endless procedural) with per-class validation; references
  resolve to any class. `SampleObject::resolve_target` replaces the literal-
  only resolver; store insert validates payload/descriptor consistency.
- `resident_sample_bytes()` accounting (endless = 0) for §41 exposure math.
- `court authored` — procedural battery (determinism, chunk equality, zero
  resident bytes, hostile spec rejection); reference hash frozen & enforced.
- `mix` gated host-only; library still compiles `no_std` for the device
  target. Semantic court reference hash unchanged across the refactor
  (1791816f...), demonstrating semantic stability.

Verified: 143 tests green; clippy/fmt clean; `court semantic` and
`court authored` SUPPORTED with receipts.

### Phase E — Exact residual / literal + exact WAV ingest (complete)

Exit criteria: intrinsic closure; literal fallback; residual variants; exact
WAV ingest; every accepted E1 input has exact intrinsic closure.

Delivered:

- `object/residual.rs` — `PredictorResidual` payload: v1 models (Zero /
  Constant / mono Periodic cycle) + sparse residual records (frame, channel,
  i32 delta; unique, sorted, bounded) with canonical bytes/identity,
  validation, `closing_residual` (compute the exact residual that closes an
  intrinsic under a model; uncloseable gaps rejected), and binary-search
  closure samples. `resident_sample_bytes` counts residual deltas.
- Closure semantics frozen (U1_SPEC §14): `X_O = H + R` precedes
  observation; no `T(H)+T(R)` commutation; residual-governed reads
  interpolate reconstructed neighbors.
- `format/wav.rs` — narrow exact WAV ingest (u8/s16/s24/s32 integer PCM
  only; float/extensible/other tags rejected explicitly), hostile-input
  tests (truncation, duplicate chunks, lying lengths, absurd rates/depths,
  bad magic, zero channels), metadata-chunk skipping, container bytes
  outside the equality claim.
- End-to-end proof: `residual_closure_equals_literal_through_the_world`
  (closure observation == literal observation, incl. interpolated reads).

Verified: 154 tests green; clippy/fmt clean; device no_std compile clean.

### Phase F — SIMD (complete)

Exit criteria: scalar == SIMD (bit parity on every floor); serious CPU
baseline exists (measured, not asserted); runtime dispatch; docs/ledger.

Correctness ledger entry — Phase D latent bug fixed during Phase F:

- `World::observe` used a standalone `voice_end_frame` pre-check that ignored
  `endless`/`periodic` voices (it only honored loop regions). Endless
  procedural voices have extent 0, so `one_shot_end_frame(0, extent 0)`
  returned `Some(t0)` and every constant/oscillator/noise/partial-bank voice
  was skipped as "already ended" — silent since Phase D, and no test asserted
  endless audibility. Fixed: the world loop now uses the canonical
  `ResolvedVoice::end_frame` (pub(crate)). Regression test
  `endless_procedural_voices_are_audible_and_class_distinct` added. The
  authored court fixture contains these classes, so its frozen reference hash
  (`f91b5b42...` → `f7e103f3...`, documented in `courts/authored.rs`
  and re-frozen with fresh receipts). Semantic court hash unchanged
  (`1791816f...`).

Second correctness fix (same phase): `object::compose_transpose` formed its
product in i64 before the documented saturation clamp, so in-domain inputs at
the top of the rate × accumulated-transpose ranges (up to 2^40 · 2^47 = 2^87)
could wrap and produce an in-domain-looking *garbage* effective rate in
release builds (debug asserted first). It now composes in i128 and saturates
at `MAX_ACCUMULATED_TRANSPOSE_Q24` exactly as documented; out-of-domain
composed rates are rejected by `rate::checked_rate` at voice resolution.
Regression: `compose_transpose_saturates_instead_of_wrapping`.

Delivered:

- `eval/backend.rs` — `Backend` (scalar/simd/auto) + `Isa` (scalar/avx2/
  avx512) runtime detection via `is_x86_feature_detected`; concrete backend
  resolution; evidence-facing labels. Selection is never semantic.
- `eval/simd.rs` — planned engine: per-voice window plans decomposed into
  envelope segments (`plan_window`, segment level formulas property-tested
  against `EnvelopeParams::level_at`); hoisted per-voice object handle, end
  clamp, release window. Bit-identical to the scalar oracle by construction
  for the scalar floor (parity batteries: 900-seed random worlds + frozen
  fixtures).
- `eval/x86.rs` + `eval/x86_ops.rs` (x86-64 only) — frame-blocked vector
  kernels (lanes = frames) generated from one macro source per floor:
  `avx512` (8 lanes, native `vpmullq`) and `avx2` (4 lanes, 64-bit ops
  emulated: mul via 32-bit multiplies, arithmetic shift / min / max via
  select). Content (literal/cycle, linear + nearest, loop/cycle wrap incl.
  single-correction fast path and exact Euclidean fallback) and endless
  (oscillator DDS + table, vectorized VOLE-SPLITMIX64-STREAM noise,
  constant, silence) classes. Partial banks and residual-governed voices
  stay on the exact shared scalar path (documented; measured 1.0×). Envelope
  division remains scalar-per-lane (no vector integer division); loads and
  mix extraction are scalar-per-lane by design. Ops emulations are
  unit-tested against scalar references (random + edge vectors, cross-floor
  agreement).
- Tail frames (< block width) render through the scalar oracle's own
  `contribution_at` — partial blocks never read past a segment boundary.
- Differential tests: frozen fixture hashes reproduced through every floor
  (`vector_floors_reproduce_frozen_fixture_hashes`), fixture-window parity,
  and a 400-seed random-world battery per vector floor. All green on this
  host (avx2 + avx512 both exercised).
- `court simd` — parity battery over semantic/authored/mixed worlds on every
  available floor + fixture-level timing; immutable receipts under
  `receipts/simd/`; verdict SUPPORTED.
- CLI help updated; `docs/PERFORMANCE.md` now contains the measured Phase F
  fixture-level rows (scalar/avx2/avx512, exact-parity enforced).

Verified: 173 tests green; `clippy --all-targets --all-features -D warnings`
clean (lib+bin+tests; example probe removed after measurement); `fmt` clean;
device `no_std` compile clean; `court semantic`, `court authored`,
`court simd` SUPPORTED with fresh receipts.

Measured (Phase F fixture-level; method in docs/PERFORMANCE.md): AVX-512
kernels 2.0–5.3× faster than the planned scalar floor (literal-interp 256v
2.9×; noise 512v 5.3×; wavetable 256v 2.6×; osc/env churn 2.0×); AVX2
1.1–2.8×; partial-bank class 1.0× (exact shared scalar path, documented).

## Known blockers

- None for Phase F. ROCm hardware absent (evidence row only). ALSA D1 court
  needs a user decision on audible output (courts default to silence-safe
  probes; `--emit-audio` opt-in flag will gate audible content).

## Next work (exact order — the implementation contract is executed in sequence)

1. Phase G — CUDA (Rust PTX evaluator, GPU-resident world, buffered
   diagnostic D0; scalar == CUDA differential).
2. Phase H — CUDA D1 falsification (ALSA mmap + registration).
3. Phase I — ROCm (hardware-unavailable evidence + clean amdgcn build).
4. Phase J — ROCm D1.
5. Phase K — inverse compiler; Phase L — GPU inverse search;
6. Phase M — production depth/courts/corpus; Phase N — transport/archive.

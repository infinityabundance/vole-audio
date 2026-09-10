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
  passband deviation ≤ 1.1e-3 (frozen Q15 table, all 1024 rows; DTFT) with
  the continuous-design prototype (analog, pre-quantization) measured
  separately (ripple 8.3e-5, stopband ≈ 210 dB — a prototype property only;
  the critically sampled rows have no digital stopband), hash pinned.
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

### Phase F seal amendments (evidence-chain hardening)

Applied right after the phase commit, before Phase G:

- **Receipt anchors**: receipts now record the committed source-tree hash
  (`git rev-parse HEAD^{tree}`) alongside the commit SHA, and the source
  dirty flag excludes `receipts/` writes by pathspec — writing evidence never
  marks the attested tree dirty by itself. Optional fields are appended and
  omitted when absent, so archived receipts still self-verify
  byte-identically. Seal receipts are produced on a clean committed tree
  (commit SHA + tree hash + `git_dirty: false`).
- **MSRV**: `rust-version = "1.89"` in Cargo.toml, verified at that seal by
  running the full suite (175 tests then, incl. the AVX-512 kernels that need
  `stdarch_x86_avx512`, stabilized in 1.89) on 1.89.0 and on stable 1.98.0.
  Host builds are stable-only; the pinned nightly is used for device/GPU
  artifact cross-compilation only (README documents the split). Suite size
  grows with each phase — the current verified count is recorded in each
  phase seal ledger (`docs/PHASE_H2.md` seal 3: 287 total all-features,
  debug and release, on the pinned nightly and on MSRV 1.89.0).
- **Resampler wording corrected**: the frozen 64×1024 Q15 table's response is
  now measured and reported separately from the continuous pre-quantization
  prototype. The rows are critically sampled (cutoff == source Nyquist), so
  they have **no digital stopband** — the earlier "stopband ≥ 210 dB" figure
  described the analog prototype only. Honest frozen-table figures (DTFT over
  all 1024 rows): worst passband deviation `max |H−1| = 1.1e-3` (edge
  F = 0.45) and the identical worst deviation in the beyond-Nyquist image
  band (exact mirror); coefficient quantization is quoted as amplitude
  error, never as a stopband figure. Enforced by
  `quantized_table_response_bounds`; prototype numbers are labeled as
  design-only in U1_SPEC.
- Test count at seal: 175 green (debug + release), clippy `-D warnings`
  clean, `fmt` clean, `no_std` host + `nvptx64-nvidia-cuda` device builds
  clean.

### Semantic facts court — Phase G prerequisite (complete)

Committed immediately after the Phase F seal, before Phase G begins:

- **Independent-oracle coverage** (`src/facts.rs`, `court facts`, receipts
  under `receipts/facts/`). Differential parity proves backend equality only;
  a bug shared by every backend survives it. Each fact (stable ids F01–F15)
  is an independent semantic statement about `vole.audio.u1` whose expected
  samples are derived from first principles — closed-form integer math,
  hand-enumerated boundary sequences, or an independent implementation — and
  observed through the full `World` path on every host surface (scalar
  authority, SIMD scalar floor, AVX2, AVX-512).
- Coverage: silence (F01), constant identity/half-gain (F02/F03), oscillator
  exact phase points at 12 kHz (F04), noise frozen vectors from an
  independent Python `VOLE-SPLITMIX64-STREAM` oracle (F05), wavetable
  fractional interpolation + periodicity (F06/F07), reverse one-shot and
  loop-boundary hand-enumeration (F08/F09), pan endpoints/center (F10),
  envelope exact ADSR/release knots (F11), mix i64-sum/single-final-+
  voice-bus saturation (F12), residual closure H+R == X_O (F13), reference
  unity transpose (F14), transpose vs an independent i128 oracle incl.
  ceiling saturation and out-of-domain rejection (F15).
- Wiring: `crate::facts` (std-gated), `court facts` in the court registry
  (verdict `SUPPORTED` only when every fact passes on every available
  surface; otherwise an honest `FAILED_CORRECTNESS` receipt), CLI usage
  updated, README current-status updated, `docs/SEMANTIC_FACTS.md` written
  (coverage + surface matrix, stable-id rule: every new representation must
  ship a fact).
- Drafting the facts caught one real wiring defect before commit: the
  envelope-knot fact originally set `VoiceSpec.note_off` directly, but the
  world scheduler only honors `VoiceOff` timeline events, so the note never
  released (sustain forever). The fact world now pushes the timeline event;
  this is exactly the class of bug the facts are designed to expose.
- Test count now 177 green (debug + release; +2 facts tests), clippy
  `-D warnings` clean, `fmt` clean.

### Phase G — CUDA D0 (complete)

The first GPU phase, executed exactly as contracted: one package, PTX
cross-compiled from `src/lib.rs`, GPU-resident flat world, D0 buffered
diagnostic only (no D1 claim anywhere), bit-exact scalar == CUDA, honest
throughput surfaces with a crossover sweep.

Delivered:

- **`device/` (no_std, shared host/device)**: `kernel_shared.rs` — flat
  `#[repr(C)]` records (`FlatVoice` 128 B, `FlatState`, `FlatEnvelope`) and
  the exact per-(frame, channel) evaluation mirroring
  `ResolvedVoice::contribution_at`/`World::observe` over the same frozen
  `no_std` functions; `nvptx_entry.rs` — `extern "ptx-kernel"
  vole_render_d0` (one thread per output sample slot, i64 register mix,
  single final saturation, coalesced write; grid-stride). `amdgcn_entry.rs`
  placeholder for Phase I.
- **`backend/flatten.rs`**: store+world → flat state; capability matrix
  (`PredictorResidual` = `CUDA_FALLBACK`, closure materialized host-side and
  counted; `Referenced` resolved before upload). Bit-exactness anchored on
  the host by a 3000-seed random battery (`flat == scalar`).
- **`backend/cuda/`**: audited dlopen FFI (legacy export names), RAII
  driver/context/module/stream/event/buffer/graph, probe, `KernelWorld`
  (world uploaded once; no per-quantum device allocation), D0 renders over
  standard / highest-priority / captured-graph submission.
- **`scripts/build-cuda-device.sh`** (live): rustc → PTX
  (`--crate-type cdylib` for nvptx64-nvidia-cuda, panic=abort, opt-3) +
  SHA-256 + provenance JSON (rustc/LLVM/target/source tree).
- **`court cuda`**: probe → artifact binding → scalar == CUDA parity on the
  frozen semantic/authored/mixed fixtures across all available strategies →
  semantic facts F01–F14 re-verified on the device → 32-seed random
  differential subset on the GPU → per-cell exactness check → fixture-level
  throughput cells incl. a voices × quantum crossover sweep. Verdict
  `SUPPORTED` only when every row passes; absent driver/device or artifact
  writes `UNSUPPORTED_BY_HARDWARE`/`INCONCLUSIVE` receipts.

Driver findings recorded (driver 610.57.04 / CUDA 13.3): the legacy
`cuCtxCreate_v2` export yields a context that rejects later allocation with
rc 201; the exported `cuCtxCreate` maps to `cuCtxCreate_v4` with the
**four-argument** ABI `(CUcontext*, CUctxCreateParams*, unsigned int flags,
CUdevice)` (NULL params = regular context); the meaningful stream-priority
range comes only from `cuCtxGetStreamPriorityRange` (there are no
MIN/MAX stream-priority device attributes); the JIT error-log options are
`CU_JIT_ERROR_LOG_BUFFER = 5` + `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6`;
plain `cuModuleLoadData` fails PTX JIT with rc 218 where
`cuModuleLoadDataEx` + the correct error-log options succeeds. The runtime
sets `CUDA_MODULE_LOADING=EAGER` and `CUDA_CACHE_DISABLE=1` (when unset),
JITs at load (setup, never a real-time path), and routes all loads through
`cuModuleLoadDataEx` with a real JIT log.

### Phase G review amendment (external review, before Phase H)

Every finding from the Phase G review was fixed, re-sealed, and recorded:

1. **cuCtxCreate ABI corrected** to the documented four-argument v4 form
   (the earlier five-argument binding was the `cuCtxCreate_v3` ABI and only
   worked for ordinal 0 by register accident). Binding + header citation in
   `backend/cuda/ffi.rs`, documented in the ledger.
2. **JIT option constants corrected** from cuda.h (`ERROR_LOG_BUFFER = 5`,
   `SIZE_BYTES = 6`, both passed; the earlier `= 2` was `CU_JIT_WALL_TIME`
   and silently disabled the error log the comments claimed). Every frozen
   numeric CUDA constant is now pinned by the `frozen_constants` unit test
   table with header citations.
3. **Stream priority range** now comes from `cuCtxGetStreamPriorityRange`:
   on this device the context reports (least 0, greatest −5); the
   high-priority stream is created at −5 and the driver-assigned value is
   read back and receipted (assigned −5; a +1 request clamps to 0 — the
   earlier receipt's `[1, 1]` came from invented attributes and produced a
   default-priority stream).
4. **D0 traffic counters now instrument the render path**, not the court:
   every launch increments `kernel_launches`, every completed observation
   quantum increments `quanta_submitted`, each DtoH transfers exactly the
   rendered window's bytes (prefix copy) and increments
   `gpu_to_host_pcm_bytes`, the host staging→output copy increments
   `host_pcm_copy_bytes`, and the D0 VRAM block + its host staging mirror
   are recorded (`device_sample_block_bytes`, resident peak). `staging`
   stays 0 by definition: the copy engine (not the host) writes the staging
   region. Sealed receipt: 166 quanta, 243 launches, 2 790 568 gpu→host
   bytes (host copy identical).
5. **CPU timing fairness**: CPU evaluators are built once outside the timed
   interval (the analogue of the GPU world staying resident). Per-observation
   semantic planning inside the CPU engines remains a documented limitation
   (a pre-resolved flat CPU baseline is future work).
6. **Receipt wording exact**: facts on device are F01–F14 (windowed); F15 is
   authority-level and surface-independent. The random CUDA battery records
   every seed: compared or skipped with the exact reason (seed 28:
   "effective rate −251 outside frozen domain after transpose" — a
   legitimately rejected random draw, preserved).
7. **Receipt→Markdown renderer**: `vole-audio receipt perf <receipt>`
   renders a receipt's `throughput_cells` as Markdown, so PERFORMANCE.md
   tables are generated from the receipt instead of hand-transcribed.

Measured (fixture-level, 48 kHz mono; RTX 4080 SUPER + driver 610.57.04 vs
Ryzen 9800X3D; receipts under `receipts/cuda/`). Mean wall ms; the full
cell table is generated in PERFORMANCE.md from the sealed receipt; key rows
at the 8192-frame window:

| cell | scalar | avx2 | avx512 | cuda-d0 wall | cuda kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| literal-256v-q8192 | 39.1 | 11.4 | 6.2 | 2.5 | 2.6 |
| noise-512v-q8192 | 96.6 | 13.3 | 6.9 | 2.5 | 2.3 |
| wavetable-256v-q8192 | 38.3 | 11.4 | 6.1 | 2.2 | 2.1 |
| oscillator-1024v-q8192 | 143.6 | 40.2 | 21.1 | 11.1 | 14.0 |
| partials-32v-q8192 | 110.5 | 106.6 | 106.8 | 24.2 | 19.4 |

Crossover cells (mean wall ms): quantum 512 favors CPU everywhere (noise
64v: avx512 0.058 vs cuda 0.192; noise 1024v: 0.926 vs 3.079; partials
64v: 13.524 vs 19.397); quantum 8192 favors CUDA (noise 1024v: 13.926 vs
3.118; noise 64v: 0.870 vs 0.197; oscillator 64v: 1.308 vs 0.297; partials
32v: 106.800 vs 24.209 — the class where the CPU SIMD floors gain nothing,
now a legitimate GPU workload). GPU cells carry run-to-run variance
(thermal/clocks) — the crossover shape, not any single number, is the
result. Every cell verified bit-exact (cuda == scalar) before timing; all
observations across floors hash-identical. Fixture-level surfaces only
(Phase M owns the flagship corpus).

- Test count now 185 green (debug + release; +2 since the Phase G commit:
  frozen-constants audit, counters add_from), clippy `-D warnings` clean,
  `fmt` clean, `no_std` lib check clean, device PTX build clean. GPU-gated
  smoke tests exist but are `#[ignore]`d (require hardware).

### Phase H — CUDA D1 (complete)

The first direct-endpoint falsification court. Question answered: can the GPU
write final sample-domain observations *directly* into the actual
ALSA-mapped endpoint region — the memory the endpoint DMA reads — with no D0
VRAM block, no device→host transfer, and no host PCM copy?

Delivered:

- **`audio/` (Linux-only, Phase A module now live)**: `alsa_ffi.rs` — audited
dlopen(`libasound.so.2`) bindings (every constant frozen from
`/usr/include/alsa/pcm.h`, signatures transcribed from the header, pinned by
tests) and `alsa.rs` — playback-endpoint discovery (/proc/asound parse, sysfs
driver evidence), the frozen D1 shape (`hw:` only, MMAP_INTERLEAVED + S32_LE +
48 kHz + stereo + 512-frame period / 1024-frame buffer = 8 KiB page-aligned
ring), explicit `snd_pcm_prepare`/`mmap_begin`/`commit`/`start`/`drain`
discipline, xrun/recover with an explicit discontinuity policy, and honest
failure classification (a busy device is never a fabricated hardware verdict).
- **`backend/cuda/direct.rs`**: `cuMemHostRegister` (DEVICEMAP) of the *exact
existing mapping* — page-boundary analysis, exact rc + driver-string capture,
pointer-attribute evidence (`cuPointerGetAttribute`), rc→verdict classifier
(never guesses: 801/1/800 → `UNSUPPORTED_BY_API` with reason; resource/state
codes stay `INCONCLUSIVE`). `KernelWorld::render_direct` — fused kernel write
into a caller device pointer with honest counters (`kernel_launches`,
`quanta_submitted`, `endpoint_observation_bytes`; **no** `gpu_to_host`/host-
copy increments — that is the D1 claim). New FFI constants frozen from cuda.h
13.3 and pinned by the frozen-constants audit (note: `WRITE_COMBINED` no
longer exists in current headers; `IOMEMORY=0x04`, `READ_ONLY=0x08`).
- **`court d1`** (registry + CLI; `--emit-audio` opt-in sets
`VOLE_D1_EMIT_AUDIO`): probes CUDA + the PTX artifact; enumerates `hw:`
playback endpoints (VOLE_D1_DEVICE overrides the documented preference order:
HDA analog → other HDA → USB → rest); runs a paced real-time session where
each contiguous mmap chunk is rendered by the kernel directly into the
registered region, stream-synchronized, shadow-verified **in place** against
the scalar oracle, and only then committed; and a controlled **D0-mmap
baseline** (D0 render → DtoH → CPU copy into the region) on the same endpoint
shape. Per-device trial rows record open/mmap/format/registration outcomes;
registration failures are classified exactly and never "fixed" by a
substitute pinned buffer. Default content is silence-safe (peak codes ≤ 2^16,
content-rich); `--emit-audio` runs an audible demo.

Measured (RTX 4080 SUPER, driver 610.57.04, Linux 7.2.2-cachyos; receipt
under `receipts/d1/`):

- **`cuMemHostRegister` on the actual `snd_hda_intel` (ALC897 analog) DMA ring
succeeded** (8 KiB region, page-aligned; device pointer 0x9800000). The GPU
wrote **48 000 frames / 94 chunks directly into the ALSA-mapped region**,
byte-exact vs the scalar oracle on every chunk (in-place verification, no
shadow sample buffer), zero xruns, clean drain; endpoint depth 512–1024
frames paced by `snd_pcm_avail_update`. Verdict **SUPPORTED**
(`D1_ENDPOINT_MAPPED` / `HOST_MAPPED`). `cuPointerGetAttribute` on the
registered range returned rc 1 (invalid argument) for every attribute and
both address targets on this driver — recorded per query; the registration,
device pointer, and successful direct kernel writes are the evidence (no
pointer-attribute value is claimed).
- **The bytes D1 removes, measured on equal frame counts**: the D0-mmap
baseline on the same endpoint (48 000 frames, the same window as the D1
session) uses the stronger `render_into` path — the DtoH transfer lands
directly in one host buffer (no internal staging copy) and is copied once
into the endpoint region: 384 000 B device→host + 384 000 B host copy; the
D1 path moved **0 B** device→host and **0 B** host copies while writing the
same window into the endpoint region (`endpoint_observation_bytes` =
384 000 B). Top-level receipt counters describe the verdict-bearing D1
path; the D0 baseline and an `experiment_aggregate` are separate named
surfaces. Measured per-chunk wall across seals (variance is real): D0 mean
0.14–0.19 ms, D1 mean 0.20–0.55 ms (an earlier seal measured 0.55 vs 0.19;
a warm re-measure 0.20 vs 0.14). D1 is a directness/traffic result, not a
latency optimization; Phase M owns the crossover question.
- Every candidate device gets its own trial row: after the D1 session,
remaining endpoints are probed for open/mmap/format/registration
(`playback_attempted: false`). In the sealed receipt the other HDA rings
(NVIDIA HDMI 1,3/7/8/9 and ALC897 Digital 2,1) registered but were not
played, and the PipeWire-held USB interface is `INCONCLUSIVE` (busy).
- Implementation bugs found and fixed during the court bring-up (recorded
here because they are exactly what an evidence-driven hardware court is
for): `snd_pcm_mmap_begin` takes the requested frame count as an *input*
value in `*frames` (a zero-initialized request returns zero frames with rc 0);
the stream must be `snd_pcm_prepare`d and explicitly `snd_pcm_start`ed (the
sw start-threshold does not auto-start on every driver; verified against a C
probe before trusting the FFI); the mmap area base is only meaningful from
PREPARED onward.
- Test count now 199 green (debug + release; +14 over the Phase G seal:
registration classifier, page rounding, frozen ALSA constants + channel-area
layout, /proc/asound parsers, failure classification incl. rate/geometry,
interleaved-geometry expectation, D1 fixture bounds/parity), clippy
`-D warnings` clean, `fmt` clean, `no_std` lib check clean.

### Phase H review amendment (external review, before reseal)

Every finding from the Phase H review was fixed and re-sealed:

1. **Exact commit transfer check**: `snd_pcm_mmap_commit` now verifies the
driver-reported transferred frame count equals the requested count; a
short commit (ALSA's documented xrun-class condition) is an explicit
`FAILED_DEADLINE`-class event with the exact requested/transferred counts
in the receipt detail.
2. **Top-level counters describe the verdict-bearing path**: a SUPPORTED D1
receipt's counters now report the D1 path (0 gpu→host, 0 host copies, 94
quanta, 94 launches, 384 000 endpoint-observation bytes) instead of the D0
baseline; the baseline and an explicit `experiment_aggregate` are separate
named surfaces.
3. **D0 and D1 run the same 48 000-frame window**, so the byte comparison
is direct. The D0 baseline uses `KernelWorld::render_into` (DtoH lands
directly in one host buffer, then one buffer→region copy): D0 = 384 000 B
DtoH + 384 000 B host copy vs D1 = 0 B / 0 B. (An earlier amendment counted
two host copies against the old staging-path baseline; switching the
baseline to the stronger single-copy form is what makes any later D1
latency claim harder to attack.)
4. **D0 host-copy accounting counted both real copies of the then-current
path** (the internal staging→buffer copy instrumented in `KernelWorld` and
the court's buffer→region copy). Superseded by the stronger `render_into`
baseline in the second amendment (single DtoH destination, one host copy).
5. **Verification is copy-free and named**: the D1 verifier compares the
mapped endpoint region against the oracle slice in place — no shadow
sample buffer — and receipts split `verification_host_read_bytes` /
`verification_shadow_copy_bytes` (0) from materialization traffic.
6. **Pointer evidence records every query**: each `cuPointerGetAttribute`
call (attribute × device-pointer/host-pointer target) records rc + driver
string + value. On this driver every query returns rc 1 (invalid
argument); the earlier "memory_type HOST" prose was removed — registration
success, the device pointer, and the working direct writes are the
claims, nothing more.
7. **Every candidate gets a trial row**: after the first successful D1
session the remaining endpoints are probed for open/mmap/format/
registration (`playback_attempted: false` rows), so the sealed receipt
covers all seven candidates (five more HDA rings registered-but-not-played;
the PipeWire-held USB endpoint is busy/`INCONCLUSIVE`).
8. **Mandatory exact rate**: `AlsaPcm::open` refuses an endpoint that cannot
grant the exact 48 kHz (a nearby rate would silently resample the frozen
timeline); the previously-unused `require_exact_rate` helper was removed.
9. **Full channel-area geometry validation**: at open every channel's area
is checked (shared `addr`, `first == ch*32`, `step == channels*32` bits)
before the simplified `base + offset * frame_bytes` arithmetic is used;
receipts record the per-channel layout (`ch0 first=0 step=64; ch1
first=32 step=64` on the sealed ALC897 ring).

### Phase H second review amendment (external review, before reseal)

The follow-up review confirmed the D1 result and the equal-work accounting,
and found seven further fixes, all applied:

1. **Structural teardown order (must-fix)**: the D1 session now binds the
`HostRegistration` and the `KernelWorld` (which owns the CUDA context) in
one `DirectSession` struct whose field order is the lifetime contract —
Rust drops fields in declaration order, so `cuMemHostUnregister` always
runs before context teardown. The earlier `drop(kw); drop(r)` ordering
destroyed the context before the unregister. The `open_with` error path
re-establishes a context before unregistering.
2. **Top-level depth extremes**: the receipt counters now carry the session's
recorded min and max endpoint depth (512..1024), not a single
`observe_endpoint_depth(max)` call that collapsed both to 1024.
3. **D0 residency measured as a peak**: the host materialization buffer is
reported at its peak allocated capacity (4 096 B), not the trailing short
chunk's length; the KernelWorld D0 diagnostic staging (unused on the
measured paths) is disclosed in the receipt limitations rather than
miscounted.
4. **Probe rows retain `RegisterRange`**: the post-success capability-probe
trials now record exact base/length/page alignment.
5. **Short commits are xrun-class**: a nonnegative-but-short
`snd_pcm_mmap_commit` increments `xruns` (explicit discontinuity) and
carries the exact requested/transferred counts; `AlsaFailure::is_xrun_class`
and the session expected-slice derivation from the committed media position
keep recovery deterministic.
6. **Geometry re-proved on every `mmap_begin`**: each chunk must still refer
to the same registered interleaved ring (shared addr + per-channel
first/step), not just the initial mapping.
7. **Stronger D0 baseline**: the D0-mmap baseline uses `render_into` — the
DtoH transfer lands directly in one host buffer (no internal staging copy)
followed by one buffer→region copy — so the comparison D0: 384 000 B DtoH
+ 384 000 B host copy vs D1: 0 B / 0 B is the harder-to-attack form. The
court also surfaced a genuinely useful tradeoff: D1 eliminates intermediate
sample movement but mapped-host GPU stores are slower than VRAM renders in
this 512-frame court (D1 mean chunk wall ≈ 0.5 ms vs D0 ≈ 0.2 ms). D1 is a
directness/residency/traffic result; latency crossover belongs to Phase M.

Test count now 200 green (debug + release; +1 xrun-class detection), clippy
`-D warnings` clean, `fmt` clean, `no_std` lib check clean, MSRV 1.89
verified.

### Phase-H entry gate (closed before H.2, per the H.2 brief)

Pre-H.2 edge cases closed on the D1 court and re-sealed:

1. Every `snd_pcm_mmap_begin` now proves the returned base equals the base
registered at open (`self.area_base`) — all channels sharing some base is
not enough; a changed mapping terminates rather than writing through a
stale registration.
2. Falsification-court policy frozen: ANY xrun / short commit / suspend
event TERMINATES the session as `FAILED_DEADLINE` (zero-xrun success is the
criterion). `frames_committed`, `chunks`, and the media position never
advance past a failed commit; expected slices derive from the committed
media position, so a failed chunk cannot desync later verification. No
mid-session recovery (a recovered stream would silently drop endpoint
timeline position).
3. Consecutive-stall counters reset on actual progress.
4. Evidence promoted into standard fields where the schema supports it:
`provenance.exact_equality` is set for the verdict-bearing path, and a
`residency` extras block separates materialization residency from
verification residency (0 for both paths: in-place verification, no shadow
buffers).
5. Stale D0/D1 timing prose refreshed from sealed receipts, including
run-to-run variance (D1 mean 0.20–0.55 ms vs D0 0.14–0.19 ms per chunk).

### Phase H.2 — Entropy-native audio core (complete)

Inserted between H and I (no renumbering). The phase corrects an architectural
omission: VOLE-Audio must not be only `procedural state -> PCM -> endpoint`;
it now embodies

```
deterministic explanation
+ entropy/configuration state
+ entropy-coded residual not reproduced by the chosen deterministic explanation
 -> bounded observation
   -> endpoint sample codes
```

Normative documents (one owner per decision): `docs/PHASE_H2.md` (charter +
seal ledger), `docs/ENTROPY_NATIVE.md` (representation contract, exposure
surfaces), `docs/RANS.md` (frozen native codec), `docs/ENTROPY_ACCOUNTING.md`
(complete cost, declared/unique/physical, methodology), `docs/ENTROPYFS.md`
(optional persistence boundaries), `docs/DSFB_SEARCH.md` (zero-authority
search governance). ADRs `0001–0005` record the five normative decisions.

Delivered (all inside the one package):

- `src/entropy/` — native deterministic byte rANS (32-bit state,
`scale_bits = 14`, `MODEL_TOTAL = 16384`, `STATE_L = 2^23`, checked
integer arithmetic, division-based scalar authority; byte-parity against
the ryg-rans-rs oracle in dev-dependencies only), canonical
largest-remainder model normalization with min-1 present-symbol
guarantee, reversible symbolizations (identity, lane4 plain/zigzag,
delta-lane4), self-describing blocks and pages, mandatory complete-cost
RAW fallback, literal + residual representations (existing Phase-E
residual semantics untouched), frozen corpus with byte-flat negative
controls, hostile-input corpus, complete-cost accounting, EmbeddedStore.
- `src/entropy/entropyfs_store.rs` + feature `entropyfs-store` — adapter to
the real published entropyfs 0.7.17 engine (default-off; optional).
- `src/entropy/search/dsfb.rs` + feature `dsfb` — deterministic observer +
adapter to the published dsfb crate (default-off; encoder search
governance only, zero decoder authority).
- `src/backend/entropy_flat.rs` — host flat-job builder (page-aligned
bounded windows; models deduped by canonical bytes; residual RAW payloads
rewritten page-local) and the host parity decoder.
- `src/device/entropy_shared.rs` — the no_std shared decoder (pages decode
one thread per page on device; repr(C) records size-pinned by tests).
- `src/backend/cuda/entropy.rs` — `EntropyWorld` GPU runtime: upload once,
bounded per-window decode jobs, `decode_into` for the fused path.
- Device entries: `vole_entropy_decode` and `vole_upmix_mono_dup` (mono ->
stereo duplication at the sampler boundary, on-device) added to the PTX
artifact (`scripts/build-cuda-device.sh`; artifact also strips the DWARF
debug sections rustc emits into PTX even at `-C debuginfo=0`).

Courts (all registered; `court h2` runs the aggregate):

- `entropy-rans` — codec battery: canonical determinism, model sweep,
hostile corpus (796 typed cases).
- `entropy-literal` — RAW vs native rANS literal vs canonical U1 vs FLAC
(pinned external tool; `NOT_AVAILABLE` when absent).
- `entropy-residual` — exact-residual entropy coding, byte-identical
closure reconstruction, complete costs.
- `entropy-pages` — page-size Pareto 64..4096, seek latency, corruption
locality.
- `entropy-partial` — partial materialization == full-slice equality, pages
touched, decode halo.
- `entropy-simd` — CPU parallel decode surface: scalar == page-parallel
decode (2 and N threads), measured sequential vs parallel wall;
instruction-level SIMD decode honestly recorded `NOT_IMPLEMENTED`
(single-state rANS is serial per stream; no fabricated vectorization).
- `entropy-cuda` — scalar == CUDA on literal delta-lane4, RAW-fallback
noise, and procedural residual closure jobs.
- `entropy-d1` — flagship fused path (below).
- `entropyfs` / `dsfb-entropy` — feature-gated store and search courts
(`INCONCLUSIVE` + limitation without their features).
- `h2` — aggregate: SUPPORTED only when all ten H.2 courts are.

Flagship `court entropy-d1` (H.2.18–H.2.20, H.2.53): entropy-coded literal
(rANS delta-lane4), procedural mono+residual (Periodic hypothesis + sparse
exact corrections), and a high-entropy RAW control ride the Phase-H D1
mechanism on the RTX 4080 SUPER + `snd_hda_intel` hw:2,0 ring. Each 512-frame
observation window decodes only its intersecting pages (two 256-frame pages,
one thread per page) and writes the exact final S32 codes straight into the
registered ALSA mmap ring before commit:

- D0-literal: 32 768 B GPU→host + 32 768 B host copies; D1-literal: 0 B / 0 B.
- D0-residual: 16 384 B GPU→host + 32 768 B host copies (mono->stereo
expansion included); D1-residual: 0 B / 0 B (device-side expansion via
`vole_upmix_mono_dup` into the ring).
- Verification reads the committed ring in place (32 768 B/session,
separately accounted; never conflated with materialization).
- All sessions shadow-exact vs the scalar oracle, zero xruns, clean drain;
the measured D1 chunk walls (seal runs: ~0.47 ms mean literal, ~0.11 ms
residual, ~0.06 ms RAW) leave clear deadline margin against the 10.67 ms
period. Methodology recorded in the receipt: 512-frame period into a
4096-frame buffer, sustained-clock warm-up launches (the serial rANS
decode is latency-chain bound; named wall regimes — ~20 ms/window
idle-first-launch, ~2–3 ms court-warmup, ~0.4–0.5 ms aggregate-hot — see
PERFORMANCE.md), and the PTX module-load fix (NUL-terminated image; the
driver's ptxas otherwise parses heap garbage past an unterminated buffer —
the cause of intermittent rc-218 JIT failures that reproduced only
in-process).

Robustness fix in the CUDA driver layer: `cuModuleLoadDataEx` input is now
NUL-terminated, eliminating process-state-dependent PTX JIT failures; the
build script strips DWARF sections from the PTX text (rustc emits them even
at `-C debuginfo=0`) and drops the matching `debug` target flag.

Seal: see the ledger in `docs/PHASE_H2.md` and `receipts/` (immutable
receipts under `receipts/<court>/`; negative rows preserved).

### Phase I — ROCm (complete)

Charter + seal ledger: `docs/PHASE_I.md`. Scope is the contract's own
statement — **ROCm (hardware-unavailable evidence + clean amdgcn build)** —
executed to the same evidence standard as every earlier phase.

- `device/amdgcn_entry.rs` (was an empty placeholder): three
  `extern "gpu-kernel"` entries mirroring the NVPTX surface exactly —
  `vole_render_d0`, `vole_entropy_decode`, `vole_upmix_mono_dup` — thin
  wrappers over the same `device::kernel_shared` / `device::entropy_shared`
  no_std semantics; AMDGCN intrinsics (`workitem_id_x`, `workgroup_id_x`,
  `feature(stdarch_amdgpu)`) with launch geometry passed as kernel
  parameters (AMD exposes no workgroup-count/size intrinsic).
- `scripts/build-rocm-device.sh` (was `NOT_IMPLEMENTED`): clean code-object
  build on the pinned nightly — rustup ships no prebuilt amdgcn std, so
  core is built from source (`-Z build-std=core`, rust-src component); the
  amdgcn cdylib link ICEs in fat-LTO with incremental (documented;
  `CARGO_INCREMENTAL=0`), and cargo's `--config` cannot override a manifest
  crate-type, so the code object is linked by direct rustc against the
  build-std rlibs (mirrors the CUDA script's direct-rustc style). Artifact:
  `scripts/out/vole_audio.amdgcn.elf` (ELF AMDGPU code object) + sha256 +
  provenance sidecar; per-ISA (`-C target-cpu=$VOLE_ROCM_GFX`, baseline
  gfx906) — code objects are not portable like PTX text.
- `backend/rocm/` (`mod.rs`, `loader.rs`, `probe.rs`, `elf.rs`): a loader
  probe walking the Phase-J runtime chain (AMD GPU -> amdgpu driver ->
  `/dev/kfd` accessible -> HIP/HSA dlopen + symbols) with the corrected
  taxonomy (userspace absence = `UNSUPPORTED_BY_API`, never hardware), plus
  self-defending ELF validation of the code object (magic/machine/entries
  from the bytes, no external tools). `court rocm` is two-dimensional and
  fails closed: an unsatisfied compile surface (artifact missing/malformed/
  unverifiable) is `INCONCLUSIVE` whatever the runtime says; with the
  surface satisfied, this host reports `UNSUPPORTED_BY_HARDWARE` (typed
  cause). The launch/direct-path runtime is Phase J scope, where ROCm
  hardware can validate it.
- Evidence-binding amendment (repo-wide, review): `build.rs` stamps
  compile-time source identity into the host binary; receipts record both
  compiled-from and executed-in-worktree, and a seal requires them to match
  (`Environment::source_bound`; `court-all.sh` rebuilds unconditionally and
  refuses a non-bound battery). PTX/AMDGPU provenance sidecars are consumed
  into the CUDA/entropy/rocm court receipts. `device::geom` holds the
  frozen launch contract with zero-geometry rejection and the pathological
  geometry unit battery.
- `scripts/build-rocm-device.sh` builds in a fresh isolated per-toolchain/
  per-gfx target dir (`target/vole-rocm/<rustc-commit>/<gfx>`, removed
  first, exactly-one rlib enforced); `--verify-deterministic` proves
  `SHA256(A1) == SHA256(A2)` across two isolated builds
  (`vole_audio.amdgcn.determinism.json`). Artifact:
  `scripts/out/vole_audio.amdgcn.elf` sha256 `5092e129…` (gfx906;
  byte-deterministic).
- Review-5 closure (Seal 6, ADR 0006, version 0.6.0): receipts now carry a
  **seal subject** — SHA-256 over the tracked source excluding the
  evidence/governance trees (`receipts/`, `target/`, `scripts/out/`,
  `docs/`, `.git/`) — and the default seal invariant is `verifier seal
  subject == receipt seal subject`, replacing the git-tree equality that
  committing receipts could never satisfy. Committing receipts/ledgers no
  longer invalidates a seal; the release head verifies the sealed subject
  without `--historical`. `git_commit`/`git_tree_sha` remain as exact
  battery-tree provenance; `vole-audio seal subject` prints the current
  subject. All 18 Seal-6 receipts share subject `8890404e…` (tree
  `57cc954`).
- Review-6 micro-hardening (Seal 7, version 0.6.1): the subject is derived
  from Git **index** entries and each includes the Git **mode**
  (`mode || NUL || path || NUL || content`) — a `100644`→`100755` change
  alters the subject even when bytes do not, symlinks contribute their
  link-target blob, and gitlinks contribute their pinned oid. All 18
  Seal-7 receipts share subject `4b809236…` (tree `28063ba`).
- Review-7 closure (Seal 8, version 0.6.2): the subject parser fails
  **closed** — malformed index records and in-progress merges (`stage !=
  0`) are explicit errors, never silently dropped records; the subject
  covers the whole index or does not exist. All 18 Seal-8 receipts share
  subject `8a4043c3…` (tree `1144433`).

### Phase J — ROCm D1 (complete)

Charter + seal ledger: `docs/PHASE_J.md`. Scope is the contract's own
statement — **ROCm D1: differential scalar == ROCm battery + the D1
endpoint experiment** — on top of the frozen Phase-I ABI contract. This
host has no AMD GPU / ROCm userspace, so the runtime is code-complete and
the courts record typed causes (`UNSUPPORTED_BY_HARDWARE` with the full
chain) — no kernel execution is pretended.

- `backend/rocm/ffi.rs`: dlopen'd HIP bindings over exactly the frozen
  `HIP_D0_REQUIRED` / `HIP_D1_ADDITIONAL` tables (+ optional evidence
  extras). No symbol outside the tables is required.
- `backend/rocm/runtime.rs`: RAII `Rocm` session, `Module` (AMDGPU code
  object from bytes), `Function` (launch with the `device::geom`
  contract), `DeviceBuffer`, and D1 `HostRegistration`
  (`hipHostRegister(hipHostRegisterMapped)` / `hipHostGetDevicePointer` /
  unregister) with typed attempts and rc classification.
- `backend/rocm/kernel.rs`: `RocmWorld` (render / render_direct / render_to
  / upmix) and `EntropyWorldRocm` (decode / decode_into) mirroring the
  CUDA host side; AMD launch geometry is passed as kernel parameters and
  equals the actual launch geometry.
- `court rocm-d0`: differential scalar == ROCm (frozen fixture windows +
  entropy literal/residual decode jobs + mono->stereo upmix), byte-compared
  against the scalar oracle; typed negatives without a D0-ready device.
- `court rocm-d1`: D1 endpoint experiment (D0 baseline + stereo-direct +
  mono-upmix sessions on the registered ALSA mmap ring, frozen zero-xrun
  policy, per-session traffic/exposure cells incl. the 2048-byte bounded
  mono intermediate); D1 readiness split is preserved — a D0-ready stack
  without host registration is `UNSUPPORTED_BY_API`, never "unavailable".
- Review closures (Seal 2, version 0.7.1; Seal 3, version 0.7.2): resource
  identity is **API lifetime AND device affinity** — `HipDevice { api,
  ordinal }` is retained by every resource, and every current-device-
  dependent operation re-selects its owner ordinal first (`make_current()`,
  no ABI expansion). This closes the thread-local-device hole
  (launch/alloc/copy/register/synchronize could otherwise follow whichever
  device a later `Rocm::open` left current). Registration failure to
  re-select the owner is reported with the veol-side sentinel rc, never a
  fabricated HIP code.

### Phase K — inverse compiler (complete)

Charter + seal ledger: `docs/PHASE_K.md`; the compiler's contract (candidate
families, accounting rules, frontier semantics, non-claims) is
`docs/INVERSE.md`. The inverse compiler answers the dual question — *what is
the cheapest exact deterministic SampleObject explanation of these observed
samples?* — as a **bounded deterministic proposal search** with zero decoder
authority.

- `inverse/observe.rs`: identity-voice scalar observation and intrinsic
  reconstruction for every candidate class (both must be exact — "close" is
  never accepted).
- `inverse/cost.rs`: complete dependency accounting in four separate
  measurements — **storage** (the eight H.2 components, with `complete_bytes`
  taken verbatim from the H.2 `CompleteCost` for entropy-carrying
  representations, and the canonical object length, truthfully decomposed, for
  the rest), **representation persistence**
  (`persistent_sample_domain_bytes`: 0 for entropy-coded representations,
  which bake no samples), **transient materialization**
  (`decoded_sample_state_bytes` / `decoded_residual_state_bytes` /
  `decoded_window_state_bytes`), and **baseline** (`raw_sample_bytes`,
  `canonical_literal_bytes`); only the eight storage components are summed.
- `inverse/propose.rs`: bounded deterministic proposals — literal (always),
  silence, constant (mode), exact-repeat (minimal KMP frame period), residual
  zero/constant/periodic (bounded scan), and exact shared references against a
  reference library. `max_candidates == 0` is rejected; a zero period scan
  disables the periodic family literally.
- `inverse/frontier.rs`: a genuine Pareto set over static objectives
  `(complete_bytes, total_ops, seek_ops)` with a validity self-check; measured
  wall times are reported per candidate but are deliberately not objectives,
  so the frontier is reproducible.
- `court inverse`: 14 fixtures (frozen H.2 corpus window), every candidate
  exact on intrinsic closure + scalar observation + a bounded seek window;
  with the corrected H.2 cost adaptation **6/14** fixtures have a non-literal
  explanation (silence 46 B vs literal 183 B, DC 50 B vs 327 B, single-sine
  574 B at an exact period of 64, quasi-periodic 15937 B, am-signal 1420 B at
  residual period 128); `impulse-train`/`transient-heavy` and all three
  negative controls are honestly cheapest as entropy-coded literals (the
  earlier residual “wins” were an artefact of double-charging the deltas);
  frontier validity/coverage, determinism, archive dedup (32 dependency bytes,
  0 sample-domain bytes) and a procedural-library reference all gated; frozen
  static-result hash `217b09a7…`.
- `court flattening`: `flat == scalar` bit-for-bit over the frozen fixtures
  and 252 adversarial battery worlds (764 windows), with host residual-closure
  materialization and upload bytes accounted rather than hidden.
- Deferred (documented, not approximated): delta/linear-predictor and
  partial/harmonic hypotheses require a residual-model vocabulary extension
  beyond the frozen u1 v1 models, so they are a universe amendment, not a
  Phase-K implementation detail.
- Evidence integrity (found while sealing K, because `court inverse` is the
  first float-bearing receipt in the seal matrix): receipt self-hashing is now
  stable under (a) `serde_json` float parsing — the `float_roundtrip` feature
  is required, or a shortest-decimal `f64` re-parses 1 ULP off and the
  receipt cannot reproduce its own hash — and (b) **additive schema growth**:
  the hash covers the receipt body exactly as the file carries it
  (`preserve_order`), so a receipt written before a new field existed still
  re-hashes its own field set. Previously 180 of 348 archived receipts failed
  `receipt show`; now all 348 verify.

## Known blockers

- None for Phases F/G/H/H.2/K/L. ROCm hardware absent (evidence row only). The
  D1 result is per-device/per-driver: another host's endpoint may register,
  refuse registration, or lack mmap — the court records whichever happens.
- Phase L measurement caveat: GPU-side period-scan timing on this host is
  clock-state-dependent (≈0.21×–1.78× the sequential scan across release runs).
  The placement policy does not depend on it — the host-parallel surface is
  consistently faster — and the receipt reports the observed ratios with an
  explicit "no stable device comparison" limitation.

### Phase L — GPU inverse search (complete)

Charter + seal ledger: `docs/PHASE_L.md`. Phase L places the inverse
compiler's dominant search cost — the bounded period scan — across four
possible surfaces and proves that placement is a **performance** question only:

- `device/search_shared.rs`: one shared `no_std` implementation
  (`period_records`, `scan_into`) that is *exactly* the p-dependent part of
  `Residual::closing_residual` for the mono periodic model (equivalence is
  asserted against the normative construction over the corpus).
- Device surface grows by exactly one entry (`vole_period_scan`, NVPTX +
  AMDGCN) with no per-backend semantics; the AMDGPU ELF validator and both
  build scripts require four entries now.
- `backend/cuda/search.rs` / `backend/rocm/search.rs`: host wrappers
  (`SearchWorld`, `SearchWorldRocm`) driving the entry; the AMD path follows
  the frozen `device::geom` launch contract.
- `inverse/search.rs`: `PeriodScan`, the scalar and host-parallel scan
  surfaces, `SearchPlacement`, and the frozen ranking rule (`rank_periods`).
- `inverse::compile_with` accepts an externally ranked period list and preserves
  the caller's rank order (filter → dedup-by-first-occurrence → take `keep` →
  canonicalize); it can change *which* periodic hypotheses are proposed, never
  whether one is accepted. `inverse::compile` ranks through
  `SearchBudget::placement` (`Auto` by default, which selects the measured-faster
  host surface; `Scalar` is the reference placement).
- `court inverse-search`: 14 fixtures × 512 periods; CUDA counts exactly equal
  the scalar counts on all 14 fixtures, 80 accepted candidates re-verified by
  the exact evaluator, and every accepted set identical between the
  sequential scan and the externally fed ranking. Work per period is
  `O(frames - p)`, so a scan of periods `1..=P` costs `P*F - P*(P+1)/2`
  comparisons. The host-parallel surface is consistently faster than
  sequential (≈0.22–0.25×); the CUDA ratio is **not stable** on this host
  (≈0.21×–1.78× across release runs, dominated by GPU clock state), so no
  stable GPU comparison is claimed. The default placement keeps the host
  surface and the device path is kept as a verified-equal surface. The receipt
  states the executed-surface count explicitly
  ("3 execution surfaces + 1 compile-only hardware-pending ROCm surface") and
  records the ratios with explicit limitations. ROCm is a typed
  `UNSUPPORTED_BY_HARDWARE` row; when AMD hardware is present the ROCm row runs
  the same 14-fixture battery as CUDA.
- Artifacts are unchanged from Seal 1 (the closure was host-side): PTX
  `d13d22c3…`, AMDGPU `5c30a4bc…` (still byte-deterministic across isolated
  builds).
- Seal 2 (review-1 closure, v0.10.0): CUDA resources now retain a shared
  `Arc<CudaContext>` (driver + context + device) so a context can never be
  destroyed under a live resource — the HIP model, applied to CUDA;
  `Function` retains its module. Gated ownership regressions pass on the RTX
  4080. `SearchBudget::placement` (`Auto`) drives production placement; external
  period rankings preserve the caller's rank order.
- Seal 3 (review-2 closure, v0.10.1): CUDA resource identity is now **lifetime
  *and* current-context affinity**. Every context-dependent operation enters
  its owner (`cuCtxSetCurrent`, previous value restored on drop; entering is a
  single `cuCtxGetCurrent` when the owner is already current), and stream-taking
  APIs take `&Stream` with an `Arc::ptr_eq` context check instead of a raw
  handle. `cuCtxPushCurrent` is deliberately not used: a context created by
  `cuCtxCreate` is already on the calling thread's stack and pushing it again
  returns rc 201 (reproduced on driver 610.57.04). Gated regressions: two-context
  isolation with restore, foreign-thread use, cross-context rejection.
- Seal 4 (review-3 driver hygiene, v0.10.2): `Cuda::open` no longer touches the
  process environment (a safe public API cannot assume a single-threaded
  process); the runner sets `CUDA_MODULE_LOADING`/`CUDA_CACHE_DISABLE` and each
  CUDA receipt records the observed `cuda_environment`. The new context is
  popped immediately, so it is *floating* on return (non-invasive creation; the
  final `Arc` can be dropped on any thread). `CurrentContextGuard` is `!Send`
  by construction plus a compile-time assertion. Gated regression:
  `context_is_floating_after_open_and_safe_to_drop_elsewhere`.

### Phase M — production depth / courts (in progress)

Charter + seal ledger: `docs/PHASE_M.md`. Increment 1 (Seal 1) delivers the
conventional baseline ladder's foundation:

- `src/baseline/flac.rs`: **B1**, an in-process conventional lossless baseline
  over the **exact canonical interleaved i32 domain** at **32 bits/sample**,
  `libflac-rs = "=0.143.1"` pinned exactly, level 5 primary with levels 0/8 as
  secondary controls, and `decode(encode(x)) == x` enforced sample-for-sample
  (an inexact row is `FAILED_CORRECTNESS`, never a smaller number). No `>> 8`,
  no dither, no normalisation, no resampling. Zero VOLE semantic authority.
- `src/baseline/reference.rs`: a **non-authoritative** reference oracle that
  runs the system `flac` on the same exact domain at identical settings when
  installed (absent = `NOT_AVAILABLE`), outside the frozen result vector. It
  records a real divergence: `libflac-rs` ports libFLAC 1.4.3, which does not
  select the CONSTANT subframe at ≥28 bits/sample, so all-zero 32-bit blocks cost
  ~1 bit/sample there (B1 2,178 B vs reference 160 B on `silence`), while the
  rest of the corpus agrees within ~2% (aggregate reference/B1 0.973).
- `court conventional`: B0 raw PCM + B1 (primary + controls) per fixture, the
  full B0–B9 ladder manifest with every row's status, and a frozen static result
  `11f8683f…`. Measured over the frozen H.2 entropy corpus: B0 1,179,648 B, B1(5)
  288,791 B (0.245× raw), B1(0) 385,114 B, B1(8) 283,660 B, u1 literal
  1,180,404 B; 42 exact round trips verified.
- CUDA construction hygiene (the deferred Phase-L review item): `Cuda::open`
  owns a freshly created context with `ProvisionalContext` so a fallible setup
  step (or panic) between `cuCtxCreate` and the RAII owner destroys the context
  and leaves the caller's context stack unchanged.

Not yet in Phase M: the B1-vs-VOLE comparison (Seal 4 now measures the flagship
conventional B0/B1 baselines over the frozen corpus, but the selected-
representation VOLE side needs the exact full-object inverse container), B2–B4,
`court depth|random-access|negative|interference|all`, the crossover surface,
energy, and the adversarial real-time load matrix.

### Phase M Seal 2 — flagship corpus freeze (v0.12.0)

The corpus is now frozen and verifiable:

- `src/corpus/specs.rs` is the frozen membership (115 objects, ~3.6 minutes,
  44.1/48/96/192 kHz) with orthogonal stratification across representation,
  amplitude occupancy, channel structure, temporal structure and entropy
  character, plus hostile full-width/scrambled controls with independent
  per-channel seeds.
- `src/corpus/generate.rs` regenerates every object deterministically
  (integer-only); `corpus/manifest.json` is the frozen identity-bearing manifest
  (identity bytes cover classes, rate, channels, frames, semantics, generator
  parameters, source and canonical i32 hash; `corpus_sha256` covers all of it).
- `court corpus` / `vole-audio corpus verify` is the gate: it fails on a bad
  schema, a corpus hash that does not cover its objects, membership drift in
  either direction, a mutated class/rate/size, a generator whose output no
  longer matches the frozen hash, and population-count drift. A mutation battery
  tests each class.
- Populations are explicit: 110 B1-comparable, 5 excluded by FLAC's format
  domain (`>8` channels, never entering a B1-vs-VOLE aggregate).
- The license-clean real-recording stratum is declared and **vacant**, with a
  documented admission path; `court corpus` records it as
  `real_audio_stratum: VACANT_DECLARED`.
- `b1_flac` now enforces the STREAMINFO MD5 invariant internally, at every
  compression level, so no caller can accept a B1 encoding without it.

Measured: corpus sha256 `c0fc62ff…`, manifest sha256 `a575bf12…`, 115/115
objects regenerated and hash-matched. The flagship comparison itself is the next
increment.

### Phase M Seal 3 — freeze-integrity closure (v0.13.0)

Still before any flagship result: the freeze passed review, and the review found
nine ways the frozen population's *interpretation* (not its bytes) could have
been adjusted after the fact. All are closed:

- **B1 eligibility is derived** (`corpus::generate::b1_comparable(channels)`) by
the verifier and the courts; the manifest's `b1_comparable` field must equal the
derivation and never feeds the comparison denominator
(`corpus::derived_b1_counts`).
- **Whole-object canonical verification**: each frozen `Spec` is regenerated and
the manifest entry is compared field for field against
`corpus::object_for(spec, samples)`; the hand-maintained subset comparison and
the fail-open class parsers are gone. `duration_ms` is recomputed and verified,
`expected_inclusion_surfaces` is derived policy, and `identity_bytes` now covers
`class` and `conversion`.
- **`universe` / `profile` / `state` are validated** (`root_mismatch`), duplicate
ids (`duplicate_id`) and reordered manifests (`order_mismatch`) are rejected, and
`corpus freeze` refuses to overwrite a `FROZEN` manifest without a deliberate,
reason-bearing `--amend-frozen`.
- The frozen axis is renamed **`source_structure_class`**: it names the material
an object was designed as, not the representation the inverse compiler later
selects. The full-width `AnticorrelatedStereo` object is kept but excluded from
the incompressible hostile population (`R = -L` is cross-channel structured);
hostile invariants now run per channel over every control.
- Manifest regenerated under the amendment: corpus sha256 `4c94b841…`, manifest
sha256 `f67c73cf…`; populations unchanged (115 objects, 110 B1-comparable, 5
excluded by format domain).

Measured (release, `--all-features`, clean tree at `9af5a1e`): the 15-row
`seal verify` matrix passes at seal subject `793a07f9…` (25 fresh receipts);
`court corpus` SUPPORTED; device artifacts byte-identical (PTX `d13d22c3…`,
AMDGPU `5c30a4bc…`); semantic/authored/inverse/inverse-search frozen hashes
unchanged; 407 passed / 12 ignored all-features, 397 passed / 12 ignored
default-features. The flagship comparison is still the next increment.

### Phase M Seal 4 — flagship conventional baseline (v0.14.0)

**The box is opened.** `court conventional` now verifies the frozen manifest and
then measures the **flagship population** (115 objects; 110 B1-comparable, 5
`NOT_APPLICABLE_BY_FORMAT_DOMAIN`), bound to manifest sha256, corpus sha256,
object order, and each object's canonical i32 hash / rate / channels / frames.
It is the **flagship B0/B1 conventional-baseline result**, not the B1-vs-VOLE
result (the `u1` literal row is the universal fallback, not the selected
representation).

Measured: B0 71,277,600 B, B1(level 5) 25,577,431 B, u1 literal 71,283,810 B
(B1/u1 0.359, B1/B0 over the comparable subset 0.390), 330 exact round trips,
reference `flac 1.5.0` 110/110 exact (reference/B1 0.990, non-authoritative).
Per-axis surfaces (B1/B0, B1-comparable subset): entropy — highly predictable
0.195, scrambled 1.001; amplitude — low_byte 0.135, full_i32 0.762; channel
structure — identical stereo 0.180, independent stereo 0.751; source structure —
literal 0.032, noise 0.844. Frozen result
`acfdaa32b9b69cdc0ee3ab2c0fb10387233603bc7a13d816575e12f7d2988b9b`; seal
subject `40c4c8e6…`; 409 passed / 12 ignored all-features, 399 passed / 12
ignored default-features. The axes are descriptive surfaces of this frozen
population, not controlled causal effects.

### Phase M Seal 5 — full-object archival container mechanism (v0.15.0)

The bounded Phase-K compiler (65,536-frame window) now explains a whole object
through an **object-above-objects** container; no new U1 `Representation` tag is
introduced and Phase K is unchanged.

- `src/fullobj/`: canonical container v1 `header ∥ segment index ∥ payloads ∥
integrity`, with real serialized bytes (`complete_bytes = header + index +` Σ
payload `+ integrity`). Segmentation is the frozen rule
`min(65,536, remaining)`, inherited verbatim from `MAX_INVERSE_FRAMES`;
selection is minimum Phase-K `complete_bytes` with deterministic tie order; each
segment is priced against an **empty** reference library (standalone, no
corpus-level dedup).
- Root semantics (finite extent + loop/one-shot identity) are preserved and
observation reproduces both, including the declared loop region past a loop
root's extent.
- `src/inverse/serialize.rs` prices and serializes the same encoding
  (`cost::best_literal`/`best_residual` share one iteration).
- `court fullobj`: 36 non-flagship fixtures over the boundary lengths 1, 65,535,
  65,536, 65,537, 131,072, 131,073 frames (mono/stereo/3-channel; silence /
  constant / exact-repeat / noise / mixed), 414 boundary observations exact, 180
  hostile containers rejected; every extent reconstructed sample-for-sample.

Measured: frozen result
`4b517ea0d564662d0b5c004434d1cea5b7358c5e1e563df5f6628a4665993a87`; seal
subject `837653dc…`; 417 passed / 12 ignored all-features, 407 passed / 12
ignored default-features. The true B1-vs-VOLE result is Seal 6.

### Phase M Seal 6 — B1 FLAC versus current bounded VOLE inverse selection (v0.16.0)

The second box is open. `src/courts/flagship.rs` compiles every frozen object
through the Seal-5 container and prices the 110 B1-comparable objects against
their FLAC level-5 bytes. It is named for the compiler that exists today, not
"optimal VOLE": the proposal vocabulary is bounded and deterministic, and every
segment is priced standalone (empty reference library).

- Bound to the frozen population (manifest/corpus sha256, object order, canonical
  hashes); B1 is recomputed in-process and required to equal the sealed
  `court conventional` total (25,577,431 B), so the courts cannot drift.
- Measured (default bounded search; 115 objects, 255 segments): B0 71,277,600 B,
  B1 25,577,431 B, VOLE 31,118,702 B (**VOLE 1.217x B1 in aggregate**; B1/VOLE
  0.822). VOLE is cheaper on **55**, equal on 0, larger on **55** of the 110
  comparable objects (within 1% on 15). Objects all-literal 61, all-procedural
  49, mixed 5; segments selected literal 133, exact_repeat 103, constant 10,
  silence 4, residual_zero 3, residual_constant 2.
- The aggregate hides the shape. VOLE wins where the vocabulary fits
  (`literal` 51.5x, `exact_repetition` 7.1x, `identical_stereo` 9.0x,
  `globally_periodic` 3.05x, `wavetable` 2.70x, `oscillator` 1.53x) and loses
  where it does not (`compound` 2.10x larger, `residual` 1.79x,
  `sparse_residual` 1.74x, `independent_stereo` 1.09x, `noise` 1.13x),
  with full-width random and scrambled near parity.
- Frozen result
`76fe5dcff1dec0a173ed67dc69effcc6befc093e340519c81ec2b32ea3b1a17b`; seal
subject `ef8a57d1…`; 417 passed / 12 ignored all-features, 407 passed / 12
ignored default-features.

## Next work (exact order — the implementation contract is executed in sequence)

Phase H.2 is complete (entropy-native core: all ten H.2 courts SUPPORTED on
this machine, aggregate `court h2` SUPPORTED; fused entropy->CUDA->D1 endpoint
path sealed). Phase I (ROCm) is complete: clean amdgcn code-object build +
`backend/rocm` probe + `court rocm`/`probe rocm` hardware-unavailable evidence
(no AMD GPU / KFD / ROCm userspace on this host). Phase J (ROCm D1) is
complete as code + courts: `court rocm-d0` (differential scalar == ROCm) and
`court rocm-d1` (D1 endpoint experiment) exist and record typed causes on this
host; their positive paths execute when a D0/D1-ready ROCm stack + AMD device
are present. Phase K (inverse compiler) is complete: bounded deterministic
proposals, exact acceptance through two independent reconstructions, complete
H.2-priced dependency accounting, and a deterministic Pareto frontier, sealed
by `court inverse` + `court flattening`. Phase L (GPU inverse search) is
complete: the bounded period scan runs on scalar / host-parallel / CUDA /
ROCm surfaces with identical counts, the device-ranked proposals are
re-verified by the exact evaluator (`court inverse-search`), and the placement
policy keeps the measured-faster host surface (`SearchBudget::placement`,
`Auto`).

1. Phase M — remaining increments (B2–B4 beside the selected full-object VOLE
   artifact; depth / random-access / negative / interference courts; crossover
   surface; energy);
   Phase N — transport/archive
   (embeds H.2 canonical records); Phase O — learned deterministic prediction
   addendum (judged by the H.2 complete-cost API).

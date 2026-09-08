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
- **MSRV**: `rust-version = "1.89"` in Cargo.toml, verified by running the
  full suite (175 tests, incl. the AVX-512 kernels that need
  `stdarch_x86_avx512`, stabilized in 1.89) on 1.89.0 and on stable 1.98.0.
  Host builds are stable-only; the pinned nightly is used for device/GPU
  artifact cross-compilation only (README documents the split).
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

## Known blockers

- None for Phases F/G. ROCm hardware absent (evidence row only). ALSA D1
  court needs a user decision on audible output (courts default to
  silence-safe probes; `--emit-audio` opt-in flag will gate audible
  content).

## Next work (exact order — the implementation contract is executed in sequence)

Phase G is complete. Phase H begins the direct-endpoint falsification work:

1. Phase H — CUDA D1 falsification: ALSA `hw:` mmap region registration
   (`cuMemHostRegister` DEVICEMAP against the actual mapped endpoint
   region), fused final writes, memory-provenance + synchronization
   evidence; SUPPORTED or an explicit negative status both complete the
   court. Never substitute a new pinned buffer for the endpoint region.
2. Phase I — ROCm (hardware-unavailable evidence + clean amdgcn build).
3. Phase J — ROCm D1.
4. Phase K — inverse compiler; Phase L — GPU inverse search;
5. Phase M — production depth/courts/corpus; Phase N — transport/archive.

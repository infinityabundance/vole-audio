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

Exit criteria: one-package repo; docs; exact status ledger; evidence schema;
environment capture; non-claims; corpus manifest; limits; `cargo test`;
`cargo clippy -D warnings`; no fake performance claims.

Delivered:

- One Cargo package `vole-audio` (lib + bin), pinned toolchain, no workspace,
  no internal crates. Host `std` feature gates host-only modules; device
  targets compile the same crate `--no-default-features`.
- Evidence constitution: `vole.audio.evidence.v1` receipts (immutable,
  self-hashing, `create_new` writes), verdict vocabulary
  (`status.rs` — the nine classes + `NOT_IMPLEMENTED`, shared `u8` codes with
  device kernels), counters with defined measurement boundary, monotonic
  timing + percentile policy (p99.9 needs N ≥ 20 000), environment/hardware
  capture (uname, os-release, cpuinfo, meminfo, sysfs PCI scan, git
  identity, rustc version), trace files with incremental SHA-256, energy
  adapter interface (honest `none` default).
- Hostile-input ceilings (`limits.rs`, no_std): frames ≤ 2^40, voices ≤ 4096,
  channels ≤ 32, objects ≤ 2^20, events ≤ 2^20, dependency depth ≤ 64, graph
  nodes ≤ 2^16, file ≤ 2^40 bytes, etc. Mixing-bound proof test
  (|mix| < 2^43 « 2^63).
- In-repo SHA-256 validated against FIPS 180-4 vectors.
- Directness (D0..D3) and Topology vocabularies (audio/), docs set, corpus
  manifest (empty-draft, honest), receipts dir, scripts (device builds +
  court driver exit `NOT_IMPLEMENTED` until their phases).
- CLI: `probe`, `receipt show`, `version`, `help`; unimplemented commands
  exit with explicit `NOT_IMPLEMENTED`-style errors (exit code 3 semantics),
  never implying support.
- Verified: `cargo test` 27 passing; `cargo clippy --all-targets
  --all-features -- -D warnings` clean; library compiles `no_std` for host and
  for `nvptx64-nvidia-cuda` with `-Z build-std=core`.

Current evidence files: (probe receipts are produced by `vole-audio probe`;
first court receipts arrive with Phase C.)

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

## Known blockers

- None for Phase D. ROCm hardware absent (evidence row only). ALSA D1 court
  needs a user decision on audible output (courts default to silence-safe
  probes; `--emit-audio` opt-in flag will gate audible content).

## Next work (exact order — the implementation contract is executed in sequence)

1. **Phase F — SIMD (AVX2 baseline; scalar == SIMD)** on the host: the same
   frozen semantics vectorized with runtime dispatch; differential parity
   tests vs the scalar oracle.
2. Phase G — CUDA (Rust PTX evaluator, GPU-resident world, buffered
   diagnostic D0; scalar == CUDA differential).
3. Phase H — CUDA D1 falsification (ALSA mmap + registration).
4. Phase I — ROCm (hardware-unavailable evidence + clean amdgcn build).
5. Phase J — ROCm D1.
6. Phase K — inverse compiler; Phase L — GPU inverse search;
7. Phase M — production depth/courts/corpus; Phase N — transport/archive.

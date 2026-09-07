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

## Known blockers

- None for Phase B. ROCm hardware absent (evidence row only). ALSA D1 court
  needs a user decision on audible output (courts default to silence-safe
  probes; `--emit-audio` opt-in flag will gate audible content).

## Next work (exact order — the implementation contract is executed in sequence)

1. **Phase B — vole.audio.u1**: sample domain (i32 canonical code); fixed-point
   semantics (Q24 position/rate, Q16 gain/pan, u64 phase); time/event total
   order; canonical serialization; content identity; frozen-table policy.
   Exit: U1_SPEC.md matches code; reference vectors frozen.
2. Phase C — scalar oracle (SampleObject, voices, ADSR, mix, interpolation,
   random access).
3. Phase D — procedural objects; Phase E — residual/literal + exact WAV ingest;
   Phase F — SIMD (AVX2 baseline); Phase G — CUDA; Phase H — CUDA D1;
   Phase I — ROCm (hardware-unavailable evidence); Phase J — ROCm D1;
   Phase K — inverse compiler; Phase L — GPU inverse search; Phase M —
   production depth/courts/corpus; Phase N — transport/archive finalization.

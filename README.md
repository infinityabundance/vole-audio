# VOLE-Audio

<p align="center"><img src="assets/vole.png" alt="VOLE-Audio" width="314" /></p>

Procedural sampling and **direct audio materialization from deterministic
state** — a rigorous native-Rust implementation of the VOLE-Audio v1.0
research architecture.

> de Beer, R. (2026). *VOLE-Audio: Procedural Sampling and Direct Audio
> Materialization from Deterministic State — Broad Prior-Art Technical
> Disclosure and Research Architecture* (Version v1.0). Zenodo.
> https://doi.org/10.5281/zenodo.22649073

> One Cargo package. PCM is an **observation view**. `SampleObject` is
> authoritative. Literal fallback exists. Everything is measured; nothing is
> assumed.

## What this project is

VOLE-Audio treats pulse-code modulation as a **bounded sample-domain
materialization requested from persistent deterministic audio state** — not as
the audio object itself. A `SampleObject` may be procedural, hybrid,
referenced, residual-governed, or literal. Playback is evaluation of a
deterministic observation function; the GPU residency and endpoint-mapping
experiments (D1/D2) are about pushing the observation boundary as close to the
physical DAC as the hardware actually permits.

## Non-negotiable reading order (what to know first)

- **PCM is an observation view**, never the authoritative object.
- **`SampleObject` is authoritative** and may be procedural / hybrid /
  referenced / residual-governed / literal.
- **Literal representation is valid and mandatory** as the universal fallback.
  If literal wins the Pareto frontier, that is the result.
- **CUDA and ROCm/HIP are the current execution targets**; the scalar
  reference owns semantic authority; the SIMD backend makes the GPU comparison
  honest.
- **D0–D3** name the materialization path: D0 buffered (diagnostic),
  D1 endpoint-mapped (GPU writes the actual endpoint region), D2 peer-device,
  D3 endpoint-native. **D3 is future conceptual work** — it exists in the
  vocabulary and returns `NOT_IMPLEMENTED`.
- **Directness is measured, never inferred**: an mmap exists ≠ D1; a PCIe link
  exists ≠ D2.
- **Negative results are results.** Unsupported paths stay visible as
  `UNSUPPORTED_BY_*` / `INCONCLUSIVE` evidence rows. Nothing is silently
  replaced by an easier buffered path and reported as success.
- **Exact is exact**: byte/hash equality, not listening. WAV *sample-domain*
  reconstruction equality is distinct from WAV *container byte* equality.
- Every court emits an immutable, self-verifying evidence receipt
  (`vole.audio.evidence.v1`) under `receipts/`.

## Repository map

```
Cargo.toml            one package; std feature gates host-only modules
rust-toolchain.toml   pinned nightly (device/GPU artifact builds only)
src/
  status.rs           the nine verdict classes + NOT_IMPLEMENTED (shared u8 codes)
  limits.rs           hostile-input ceilings (no_std, shared with device)
  hash/               in-repo SHA-256 (FIPS 180-4), no_std
  universe/           vole.audio.u1 exact semantics        (Phase B)
  object/             SampleObject model                    (Phase C)
  sampler/            voices/world/scheduler                (Phase C)
  eval/               scalar + SIMD + shared battery   (Phase C/F)
  device/             flat kernel semantics (no_std); nvptx entry live (G);
                      amdgcn entry (Phase I)
  backend/            flatten (G); cuda/ host runtime live (G); rocm/ (I)
  audio/              ALSA endpoint, directness, topology   (Phase H+)
  format/             canonical archive + WAV ingest        (Phase E)
  inverse/            bounded inverse-proceduralization     (Phase K+)
  transport/          deterministic framing                 (Phase N+)
  evidence/           receipts/counters/timing/environment  (Phase A)
  courts/             executable courts                     (Phase C+)
  main.rs             the vole-audio CLI                    (grows by phase)
docs/                 spec + evidence + non-claims (repo only)
corpus/               frozen flagship corpus, Phase M (repo only)
receipts/             immutable evidence outputs (repo only)
assets/u1/            frozen deterministic tables (resampler, sine)
scripts/              device build + court drivers (repo only)
```

> Package vs repository: the crates.io tarball ships the library, binary,
> `assets/` tables, licenses, and this README. `docs/`, `corpus/`, `receipts/`,
> `scripts/`, and `rust-toolchain.toml` live in the GitHub repository only
> (they are excluded from the published package to keep it lean and
> stable-buildable). See the links under [Status](#current-status) and
> [Building](#building) for the repository-only material.

## Current status

Phases A–H are complete: A–E the exact representation model on the scalar
oracle, F the honest SIMD baseline (scalar == AVX2 == AVX-512), G the CUDA D0
buffered-diagnostic backend (scalar == SIMD == CUDA bit-for-bit; semantic
facts F01–F14 verified on the device), and H the CUDA D1 falsification court
against the real ALSA `hw:` mmap endpoint (the first direct-endpoint evidence;
see the summary below). Executable evidence today:

- `cargo run -- court semantic` — scalar oracle determinism battery
  (reference SHA-256 `1791816f…`);
- `cargo run -- court authored` — procedural SampleObject battery
  (`f7e103f3…`);
- `cargo run -- court simd` — Phase F SIMD parity: scalar == SIMD on every
  available ISA floor (AVX-512 / AVX2 / scalar) over frozen worlds, with
  fixture-level timing;
- `cargo run -- court facts` — independent semantic facts (F01–F15):
  first-principles oracles for every representation/transform, verified on
  every host surface (see
  [SEMANTIC_FACTS.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/SEMANTIC_FACTS.md));
- `cargo run -- court cuda` — Phase G CUDA D0: `scalar == CUDA` bit-exact on
  the frozen fixture worlds across standard / high-priority / captured-graph
  submission, semantic facts F01–F14 re-verified on the device, a random
  differential subset on the GPU, and fixture-level CPU vs CUDA throughput
  cells incl. a voices × quantum crossover sweep. Needs the PTX artifact
  (`scripts/build-cuda-device.sh`) and a CUDA device; absent hardware or
  artifact yields an honest `UNSUPPORTED_BY_HARDWARE` / `INCONCLUSIVE`
  receipt, never a manufactured result;
- `cargo run -- court d1` — Phase H CUDA D1 falsification: register the
  **actual** ALSA `hw:` mmap region (`cuMemHostRegister` DEVICEMAP + device
  pointer), render each contiguous mmap chunk's final codes directly into
  that region (kernel write → stream sync → in-place shadow verify vs the
  scalar oracle → `snd_pcm_mmap_commit`), and compare against a D0-mmap
  baseline on the same endpoint shape — measuring the exact bytes D1
  removes. Default content is silence-safe; `--emit-audio` opts into an
  audible demo. Verdicts are per-device and honest: the sealed run registered
  the on-board HDA ring (`snd_hda_intel`) and played byte-exact with zero
  xruns (`D1_ENDPOINT_MAPPED`, `HOST_MAPPED`); devices that refuse open,
  mmap, format, or registration stay visible as their own negative rows.
  Requires Linux + ALSA + the PTX artifact + a CUDA device.

The exact ledger — completed phases, evidence, blockers, and the next work
item — is
[PROJECT_STATE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PROJECT_STATE.md)
(repository-only). Fixture-level measurements are in
[PERFORMANCE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PERFORMANCE.md);
the spec is
[U1_SPEC.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/U1_SPEC.md),
and
[NON_CLAIMS.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/NON_CLAIMS.md)
says what this repository does *not* claim.

## Building

```sh
cargo build     # host build; needs no CUDA/ROCm/ALSA development files
cargo test      # unit + property + differential tests
cargo clippy --all-targets --all-features -- -D warnings
```

Toolchain split (documented precisely because it is easy to blur):

- **Host builds (the published crate) are stable-Rust**: `rust-version` in
  `Cargo.toml` names the minimum stable release this package is verified
  against. `cargo build`/`cargo test` succeed with that stable toolchain and
  no GPU toolchain installed.
- **Device/GPU artifact builds are nightly-only**: `rust-toolchain.toml` in
  the repository pins the exact nightly used to cross-compile this same
  package to `nvptx64-nvidia-cuda` (PTX) and `amdgcn-amd-amdhsa` (HSA/ELF).
  The pin is repository-only — it is excluded from the published package so
  the crates.io tarball never forces nightly on consumers.

The CUDA device artifact (Phase G, live) is produced by cross-compiling this
same package via
[build-cuda-device.sh](https://github.com/infinityabundance/vole-audio/blob/main/scripts/build-cuda-device.sh)
— one package, no second crate — emitting `scripts/out/vole_audio.ptx` plus
SHA-256 and provenance metadata. `court cuda` loads that PTX through the CUDA
driver API (dlopen'd; nothing links against CUDA at build time). The ROCm
artifact is produced by the Phase I script
[build-rocm-device.sh](https://github.com/infinityabundance/vole-audio/blob/main/scripts/build-rocm-device.sh)
(repository-only; active in Phase I). GPU support is opt-in and runtime
probed; CPU-only machines run every non-GPU court unchanged.

## Evidence constitution

Every claim this repository makes is backed by an immutable receipt. Run
`vole-audio probe` to see the environment capture and `vole-audio receipt show
<file>` to verify a receipt's self-hash. Each receipt records its git commit,
a source-tree hash, and a dirty-state that excludes receipt-output writes —
see the
[EVIDENCE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/EVIDENCE.md)
measurement-boundary notes (repository-only). Courts arrive with their phases;
the court list is fixed in the implementation contract (semantic, authored,
simd, facts, inverse, flattening, cuda, rocm, d1, d2, depth, conventional,
random-access, negative, interference, all).

## Non-claims

See
[NON_CLAIMS.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/NON_CLAIMS.md)
(repository-only). Notably: no *flagship* performance figure exists yet —
fixture-level Phase F measurements exist and are labeled as such in
[PERFORMANCE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PERFORMANCE.md);
no real-time/deadline claim is made before the Phase M courts. This
repository is hostile to self-deception by design.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) /
  https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) /
  https://opensource.org/licenses/MIT)

at your option.

The VOLE-Audio v1.0 paper (DOI 10.5281/zenodo.22649073) remains a separate
work with its own disclosure terms.

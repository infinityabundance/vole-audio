![VOLE-Audio](assets/vole.png)

# VOLE-Audio

> de Beer, R. (2026). *VOLE-Audio: Procedural Sampling and Direct Audio
> Materialization from Deterministic State — Broad Prior-Art Technical
> Disclosure and Research Architecture* (Version v1.0). Zenodo.
> https://doi.org/10.5281/zenodo.22649073

Procedural sampling and **direct audio materialization from deterministic
state** — a rigorous native-Rust implementation of the VOLE-Audio v1.0
research architecture
(DOI [10.5281/zenodo.22649073](https://doi.org/10.5281/zenodo.22649073)).

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
rust-toolchain.toml   pinned nightly (reproducible kernels + artifacts)
src/
  status.rs           the nine verdict classes + NOT_IMPLEMENTED (shared u8 codes)
  limits.rs           hostile-input ceilings (no_std, shared with device)
  hash/               in-repo SHA-256 (FIPS 180-4), no_std
  universe/           vole.audio.u1 exact semantics        (Phase B+)
  object/             SampleObject model                    (Phase C+)
  sampler/            voices/world/scheduler                (Phase C+)
  eval/               scalar/SIMD evaluators                (Phase C+)
  device/             GPU ABI + nvptx/amdgcn entry points   (Phase G+)
  backend/            cuda/, rocm/ host runtimes            (Phase G+)
  audio/              ALSA endpoint, directness, topology   (Phase 21+)
  format/             canonical archive + WAV ingest        (Phase E+)
  inverse/            bounded inverse-proceduralization     (Phase K+)
  transport/          deterministic framing                 (Phase N+)
  evidence/           receipts/counters/timing/environment  (Phase A)
  courts/             executable courts                     (Phase C+)
  main.rs             the vole-audio CLI                    (grows by phase)
docs/                 spec + evidence + non-claims
corpus/               frozen flagship corpus (Phase M)
receipts/             immutable evidence outputs
assets/u1/            frozen deterministic tables (resampler, Phase C)
scripts/              device build + court drivers
```

## Current status

Phase A (evidence constitution) is complete. See
[docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) for the exact ledger —
completed phases, evidence, blockers, and the next work item. Phase B
(`vole.audio.u1`) is next and is in progress.

## Building

```sh
cargo build     # host build; needs no CUDA/ROCm/ALSA development files
cargo test      # unit + property tests
cargo clippy --all-targets --all-features -- -D warnings
```

GPU artifacts are produced by cross-compiling this same package
(`scripts/build-cuda-device.sh`, `scripts/build-rocm-device.sh`) and are
opt-in. `cargo build`/`cargo test` succeed on machines with no GPU toolchain.

## Evidence constitution

Every claim this repository makes is backed by an immutable receipt. Run
`vole-audio probe` to see the environment capture and `vole-audio receipt show
<file>` to verify a receipt's self-hash. Courts arrive with their phases; the
court list is fixed in the implementation contract (semantic, authored,
inverse, flattening, cuda, rocm, d1, d2, depth, conventional, random-access,
negative, interference, all).

## Non-claims

See [docs/NON_CLAIMS.md](docs/NON_CLAIMS.md). Notably: no performance number
exists yet — [docs/PERFORMANCE.md](docs/PERFORMANCE.md) says exactly
`NOT YET MEASURED`. This repository is hostile to self-deception by design.

## License

No license is asserted by this repository's maintainers at this time; see
the upstream paper for disclosure terms.

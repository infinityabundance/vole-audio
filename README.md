# VOLE-Audio

<p align="center"><img src="assets/vole.png" alt="VOLE-Audio" width="314" /></p>

Procedural sampling and **direct audio materialization from deterministic
state** — a rigorous native-Rust implementation of the VOLE-Audio research
architecture, currently specified by v1.1.

> de Beer, R. (2026). *VOLE-Audio: Procedural Sampling and Direct Audio
> Materialization from Deterministic State — Entropy-Native Layer and
> Late-Materialization Architecture* (Version v1.1). Zenodo.
> https://doi.org/10.5281/zenodo.22666746

> Original v1.0 broad prior-art disclosure:
> https://doi.org/10.5281/zenodo.22649073
>
> v1.1 is the current architecture paper — it makes the entropy-native layer
> explicit (persist the deterministic explanation, entropy-code the exact
> residual, materialize samples only when an observation requires them), and
> Phase H.2 of this repository implements the first concrete entropy-native
> core of that layer. v1.0 remains the earlier broad disclosure.

> One Cargo package. PCM is an **observation view**. `SampleObject` is
> authoritative. Literal fallback exists. Everything is measured; nothing is
> assumed.

## The central idea (as of Phase H.2)

**Persist deterministic state and entropy-coded innovation. Materialize
waveform samples only when an observation requires them.**

VOLE-Audio's durable representation of an audio object is a *deterministic
explanation* (procedural state — or a literal when no explanation wins) plus
a *native entropy-coded exact residual* for whatever the explanation cannot
reproduce, organized as block-addressable pages so any bounded observation
decodes only the pages it touches. PCM — the final sample codes a DAC
consumes — remains a legitimate observation surface and a literal
representation is always the mandatory universal fallback; it is simply **not
universally privileged as authoritative durable state**. The strongest path
executed in this repository renders entropy-coded objects on the GPU and
writes the exact final sample codes directly into the registered ALSA mmap
endpoint region (see the Phase H/H.2 summaries below).

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
  entropy/            native rANS + models + pages + residual coding + corpus
                      + EmbeddedStore + DSFB observer       (Phase H.2)
  device/             flat kernel semantics (no_std); nvptx entry live (G);
                      amdgcn entry live (I; code object per gfx target);
                      shared period-scan search primitive (L)
  backend/            flatten (G); cuda/ host runtime live (G) + entropy (H.2)
                      + search (L); rocm/ probe + artifact (I), runtime (J),
                      search (L); entropy_flat (H.2) host flat-job builder
  audio/              ALSA endpoint, directness, topology   (Phase H+)
  format/             canonical `.volea` archive + WAV ingest   (Phase E/N)
  inverse/            bounded inverse-proceduralization + search placement (K/L)
  learned/            Phase O learned deterministic prediction (experimental profile)
  transport/          deterministic framing + receiver + clock  (Phase N)
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

Phases A–L are complete: A–E the exact representation model on the scalar
oracle, F the honest SIMD baseline (scalar == AVX2 == AVX-512), G the CUDA D0
buffered-diagnostic backend (scalar == SIMD == CUDA bit-for-bit; semantic
facts F01–F14 verified on the device; F15 is authority-level and
surface-independent), H the CUDA D1 falsification court
against the real ALSA `hw:` mmap endpoint (the first direct-endpoint
evidence), H.2 the entropy-native core — a deterministic native rANS
codec with canonical models, block-addressable pages and mandatory RAW
fallback; literal + exact-residual entropy representations; optional
EntropyFS persistence and DSFB search governance; CUDA entropy decode; the
flagship **fused entropy -> CUDA -> D1 endpoint** court — I the ROCm
device surface: a clean `amdgcn-amd-amdhsa` code-object build
(`scripts/build-rocm-device.sh`, thin kernels over the same shared no_std
semantics; byte-deterministic across isolated builds), the `backend::rocm`
loader probe (GPU -> amdgpu -> KFD -> HIP/HSA) and `court rocm` /
`probe rocm` evidence (two-dimensional, fail-closed; hardware-unavailable
on this host), J the ROCm D0/D1 runtime (`backend::rocm` HIP host runtime
with structural resource lifetimes and device affinity; `court rocm-d0`
differential scalar == ROCm and `court rocm-d1` endpoint experiment, both
typed to this host's missing device), and K the **inverse compiler**
(`src/inverse/`, `court inverse`, `court flattening`) — bounded
deterministic proposal search for the cheapest *exact* deterministic
`SampleObject` explanation of an observed window, priced with the H.2
complete-cost oracle and reported as a deterministic Pareto frontier
(see
[INVERSE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/INVERSE.md)),
and L the **GPU inverse search** placement of that compiler's bounded period
scan on scalar / host-parallel / CUDA / ROCm surfaces — identical rankings on
all surfaces, with every device-ranked proposal re-verified by the exact
evaluator (see
[PHASE_L.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PHASE_L.md)).
Executable
evidence today:

- `cargo run -- court semantic` — scalar oracle determinism battery
  (reference SHA-256 `1791816f…`);
- `cargo run -- court authored` — procedural SampleObject battery
  (`f7e103f3…`);
- `cargo run -- court simd` — Phase F SIMD parity: scalar == SIMD on every
  available ISA floor (AVX-512 / AVX2 / scalar) over frozen worlds, with
  fixture-level timing;
- `cargo run -- court facts` — independent semantic facts (F01–F15):
  first-principles oracles for every representation/transform, verified on
  every host surface (F01–F14 additionally device-verified in `court cuda`;
  F15 is authority-level and surface-independent — see
  [SEMANTIC_FACTS.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/SEMANTIC_FACTS.md));
- `cargo run -- court inverse` — Phase K inverse compiler: over the frozen
  corpus window, every accepted candidate reproduces the observed samples
  exactly through **both** its intrinsic closure and the scalar evaluator
  (plus a bounded seek window); storage costs come from the H.2 complete-cost
  oracle and are reduced to a deterministic Pareto frontier over static
  objectives. On the frozen corpus 6/14 fixtures have a cheaper exact
  non-literal explanation (silence 46 B vs literal 183 B, DC 50 B vs 327 B,
  single-sine 574 B at an exact period of 64, quasi-periodic 15937 B,
  am-signal 1420 B at residual period 128), while `impulse-train`,
  `transient-heavy`, `harmonic-tone`, `fm-signal`, `stereo-correlated` and the
  three negative controls are honestly cheapest as entropy-coded literals. The
  explanation search, archive deduplication and a procedural-library reference
  are reported separately;
- `cargo run -- court flattening` — Phase K host flat-evaluator parity:
  `flat == scalar` bit-for-bit over the frozen fixtures and an adversarial
  battery, with residual-closure materialization and upload bytes accounted
  rather than hidden;
- `cargo run -- court inverse-search` — Phase L search placement: the bounded
  period scan over 14 fixtures × 512 candidate periods on scalar,
  16-thread parallel and CUDA surfaces, requiring identical per-period counts
  on every surface and requiring the device-ranked proposals to reproduce
  exactly the sequential accepted set (re-verified by the exact evaluator).
  Measured ratios are reported as-is, with no claim that the GPU wins; the
  ROCm row is a typed negative on this host while the `vole_period_scan` entry
  is part of the AMDGPU compile surface;
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
  scalar oracle — no shadow sample buffer → `snd_pcm_mmap_commit` with an
  exact transferred-frame check), and compare against a D0-mmap baseline
  that runs the **same 48 000-frame window** on the same endpoint shape —
  measuring the exact materialization bytes D1 removes (D0: 384 KB DtoH +
  384 KB host copy; D1: 0 B / 0 B — D1 is a directness/traffic result, not
  a latency claim, in this court). Verification reads are a separately
  named surface. Default content is silence-safe; `--emit-audio` opts into
  an audible demo. Every candidate endpoint gets its own trial row (the
  first registered device runs the session; the rest are probed for
  open/mmap/format/registration with `playback_attempted: false`), and the
  endpoint must grant the exact 48 kHz rate with validated interleaved
  channel geometry. Requires Linux + ALSA + the PTX artifact + a CUDA
  device; the sealed run registered the on-board HDA ring
  (`snd_hda_intel`) and played byte-exact with zero xruns
  (`D1_ENDPOINT_MAPPED`, `HOST_MAPPED`); devices that refuse open, mmap,
  format, or registration stay visible as their own negative rows.
- Phase H.2 entropy courts — `court entropy-rans` (native codec battery incl.
  hostile corpus), `court entropy-literal` (RAW vs native rANS vs U1 vs FLAC
  baselines), `court entropy-residual` (exact-residual entropy coding),
  `court entropy-pages` (page-size Pareto 64..4096 + seek/corruption),
  `court entropy-partial` (partial == full slice), `court entropy-simd`
  (CPU page-parallel surface, exact; instruction-SIMD decode honestly
  recorded `NOT_IMPLEMENTED`), `court entropy-cuda` (scalar == CUDA decode
  on literal/RAW/residual jobs), `court entropyfs` / `court dsfb-entropy`
  (optional store persistence and zero-authority search governance;
  feature-gated, `INCONCLUSIVE` without), and `court entropy-d1` — the
  flagship fused path: entropy-coded literal, procedural mono+residual, and
  a high-entropy control are decoded on the GPU per bounded 512-frame window
  (only the window's pages) and written directly into the registered ALSA
  ring beside an equal-work D0 baseline — D1 removes 32 768 B GPU→host +
  32 768 B host copies (literal) and 16 384 B + 32 768 B (residual) with
  zero xruns and byte-exact ring codes; verification is separately
  accounted. `court h2` runs the whole H.2 battery as an aggregate.
  `court rocm` (Phase I) is two-dimensional and fails closed: the
  `compile_surface` (artifact present + ELF-valid AMDGPU code object with the
  required kernel entries + provenance sidecar matching the attested source
  tree; built by `scripts/build-rocm-device.sh`, byte-deterministic across
  isolated builds) and the `runtime_surface` (AMD GPU -> amdgpu driver ->
  KFD -> HIP/HSA dlopen + symbols; missing userspace is `UNSUPPORTED_BY_API`,
  never a hardware verdict). An unsatisfied compile surface is
  `INCONCLUSIVE`; with it satisfied, this host reports
  `UNSUPPORTED_BY_HARDWARE` with the typed cause. No kernel is executed by
  the Phase I court; the differential scalar == ROCm battery is Phase J on
  ROCm hardware. `vole-audio probe rocm` prints the same classification.
  Receipts bind the binary to its source (compiled-from ==
  executed-in-worktree) and the GPU artifacts to their provenance sidecars.

The exact ledger — completed phases, evidence, blockers, and the next work
item — is
[PROJECT_STATE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PROJECT_STATE.md)
(repository-only). The H.2 phase charter and seal ledger live in
[PHASE_H2.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PHASE_H2.md)
with its normative documents (`ENTROPY_NATIVE.md`, `RANS.md`,
`ENTROPY_ACCOUNTING.md`, `ENTROPYFS.md`, `DSFB_SEARCH.md`, ADRs 0001–0005);
the Phase K inverse compiler and its seal ledger live in
[INVERSE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/INVERSE.md)
and
[PHASE_K.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PHASE_K.md);
the Phase L search placement is in
[PHASE_L.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PHASE_L.md).
Fixture-level measurements are in
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
random-access, negative, interference, all, archive, transport). `inverse`,
`flattening` and `inverse-search` are Phases K/L; `d2`, `depth`, `conventional`,
`random-access`, `negative`, `interference` and `all` arrive with Phase M; the
canonical `.volea` archive and deterministic transport (`archive`, `transport`,
`phase-n`) arrive with Phase N. The learned deterministic prediction courts
(`learned-determinism`, `learned-residual-codec`, `learned-linear`,
`learned-intrinsic`, `learned-transfer`, `learned-residual`,
`learned-quantization`, `learned-capacity`, `learned-shared`,
`learned-random-access`, `learned-gpu`, `learned-training-cost`,
`learned-inverse`, `learned`) arrive with Phase O under the experimental
`vole.audio.learned.exp1` profile.

## Non-claims

See
[NON_CLAIMS.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/NON_CLAIMS.md)
(repository-only). Notably: no *flagship* performance figure exists yet —
fixture-level Phase F/H.2 measurements exist and are labeled as such in
[PERFORMANCE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PERFORMANCE.md);
no real-time/deadline claim is made before the Phase M courts. The entropy
phase adds its own non-claims: rANS is never presented as a generator, the
DAC does not "consume compressed audio", D1 is not a latency optimization in
the Phase-H court, EntropyFS/DSFB are optional and never enter playback, and
the GPU never owns semantic authority (scalar does). This repository is
hostile to self-deception by design.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) /
  https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) /
  https://opensource.org/licenses/MIT)

at your option.

The VOLE-Audio papers — v1.1, the current architecture paper (DOI
10.5281/zenodo.22666746), and v1.0, the original broad prior-art disclosure
(DOI 10.5281/zenodo.22649073) — remain separate works with their own
disclosure terms.

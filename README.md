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
> core of that layer. Later phases extend it: Phase O adds learned deterministic
> prediction, and the speech campaign plus the Phase-6 mechanisms push the
> exact portfolio past FLAC-8 on the frozen speech corpus. v1.0 remains the
> earlier broad disclosure.

> One Cargo package. PCM is an **observation view**. `SampleObject` is
> authoritative. Literal fallback exists. Everything is measured; nothing is
> assumed.

## The central idea

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
endpoint region (Phase H/H.2).

Since Phase O the *explanation* may also be a **learned deterministic
hypothesis** (a quantized predictor proposed by fitting and admitted only after
exact residual closure and complete physical-byte accounting), and since the
speech campaign the exact portfolio is measured *past FLAC-8* on a frozen
LibriSpeech-derived corpus. The exact-residual closure contract never changes:
the literal fallback remains mandatory, and every candidate is scored by its
complete serialized bytes.

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
  learned/            learned deterministic prediction: Phase O (exp1), the Exp2
                      addendum, the Exp3 speech/residual mechanisms, and the
                      Phase-6 parsing/representation models (experimental profiles)
  transport/          deterministic framing + receiver + clock  (Phase N)
  evidence/           receipts/counters/timing/environment  (Phase A)
  courts/             executable courts                     (Phase C+; registry in
                      src/courts/mod.rs)
  main.rs             the vole-audio CLI                    (grows by phase)
docs/                 spec + phase charters + evidence + non-claims + CHANGELOG (repo only)
corpus/               frozen flagship corpus, Phase M (repo only)
receipts/             immutable evidence outputs (repo only)
assets/u1/            frozen deterministic tables (resampler, sine)
scripts/              device build + court drivers (repo only)
```

> Package vs repository: the crates.io tarball ships the library, binary,
> `assets/` tables, licenses, and this README. `docs/`, `corpus/`, `receipts/`,
> `scripts/`, and `rust-toolchain.toml` live in the GitHub repository only
> (they are excluded from the published package to keep it lean and
> stable-buildable). See the links under [Current status](#current-status) and
> [Building](#building) for the repository-only material.

## Current status

Phases A–N are complete. **Phase O** (learned deterministic prediction, profile
`vole.audio.learned.exp1`) is complete; the **Exp2 addendum** imports every Exp1
candidate and adds the v2 residual codecs and mechanism families; the
**speech campaign** (Seals S0–S8, v0.29.0–v0.37.0) took the exact learned
portfolio past FLAC-8 on a frozen LibriSpeech-derived corpus; the **fourth-pass
entropy/adaptation ladder** (E0–F0, v0.42.0–v0.53.0) added the signed/FSM range
coders, the natural-gradient predictor and BGMC; and **Phase 6** is in progress,
sealing one mechanism per release.

**Measured position (as of v0.57.0).** On the frozen real-speech corpus
(LibriSpeech, CC BY 4.0), the exact learned portfolio is:

| split | VOLE portfolio | FLAC-5 | FLAC-8 | record |
| ----- | -------------- | ------ | ------ | ------ |
| effectiveness (dev-clean, 8 clips) | **126 620 B** | 130 331 B | 129 713 B | 7/8 wins vs both |
| held-out Mode C (test-clean, 8 clips) | **136 852 B** | 144 145 B | 142 714 B | 8/8 wins |

That is a speech-corpus result on this host — not a general-audio flagship
claim, and not a real-time/deadline claim.

Phase 6 so far: `StatefulSyntaxParse` (v0.54.0), `IterativeReprice` (v0.55.0),
`EntropyReblock` (v0.56.0 — it moved effectiveness 126 899 → 126 620 B and
Mode C 137 362 → 136 852 B), and `SampleExpertMux` (v0.57.0).

**Executable evidence today.** `cargo run -- court <name>`. The authoritative
court list is the registry in `src/courts/mod.rs`; the aggregate courts are
`all` (Phase M), `h2` (Phase H.2), `phase-n` (Phase N) and `learned` (Phase O),
and every court emits an immutable receipt under `receipts/`. The exact seal
expectation is checked by `vole-audio seal verify --receipts receipts`
(59 rows at v0.57.0). The full per-phase evidence narrative — the Phase H/H.2
endpoint measurements, the Phase K/L inverse results, and every seal's numbers —
lives in
[CHANGELOG.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/CHANGELOG.md).

The exact ledger — completed phases, evidence, blockers, and the next work
item — is
[PROJECT_STATE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PROJECT_STATE.md)
(repository-only). The phase charters (`PHASE_H2.md`, `PHASE_K.md`,
`PHASE_L.md`, `PHASE_M.md`, `PHASE_N.md`, `PHASE_O.md`, `PHASE_O_EXP2.md`) and
the normative documents (`ENTROPY_NATIVE.md`, `RANS.md`,
`ENTROPY_ACCOUNTING.md`, `ENTROPYFS.md`, `DSFB_SEARCH.md`, ADRs 0001–0005) live
under `docs/`. Fixture-level measurements are in
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
a source-tree hash, and a dirty-state that excludes receipt-output writes; the
**seal subject** it binds is every tracked source file except
`receipts/`, `target/`, `scripts/out/`, `docs/` and `.git/`, so committing
evidence or documentation cannot invalidate a seal while any code change does.
See the
[EVIDENCE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/EVIDENCE.md)
measurement-boundary notes (repository-only).

The court list is not fixed in this README: the authoritative registry is
`COURT_NAMES` in `src/courts/mod.rs`, and the newest receipt per court is what
`seal verify` checks. Courts arrived with their phases — A–M (oracle, SIMD,
CUDA/ROCm, inverse, conventional and runtime courts), N (`archive`,
`transport`, `phase-n`), O (`learned-*` under `vole.audio.learned.exp1`), and
the Exp2/Exp3/Phase-6 courts under the experimental `learned` profiles. The
exact per-seal matrix and every frozen result hash are recorded per receipt and
summarized in
[CHANGELOG.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/CHANGELOG.md).

## Non-claims

See
[NON_CLAIMS.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/NON_CLAIMS.md)
(repository-only). In particular: the speech-corpus result above is exactly
that — a frozen-speech-corpus result on this host, not a general-audio flagship
claim, and not a real-time/deadline claim. Fixture-level measurements in
[PERFORMANCE.md](https://github.com/infinityabundance/vole-audio/blob/main/docs/PERFORMANCE.md)
are labeled as fixture-level. The entropy phase adds its own non-claims: rANS is
never presented as a generator, the DAC does not "consume compressed audio", D1
is not a latency optimization in the Phase-H court, EntropyFS/DSFB are optional
and never enter playback, and the GPU never owns semantic authority (scalar
does). Learned prediction is an experimental candidate family, never truth: a
learned object participates only after canonical quantization, exact residual
closure and complete dependency accounting, and the literal fallback stays
mandatory. This repository is hostile to self-deception by design.

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

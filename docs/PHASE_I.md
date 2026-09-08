# PHASE I — ROCm (AMD device surface)

> Phase charter and seal ledger. Phase I follows the sealed Phase H.2 and
> precedes Phase J (ROCm D1), exactly as the implementation contract
> sequences them. Phase I scope is the repository's own statement — **ROCm
> (hardware-unavailable evidence + clean amdgcn build)** — executed to the
> same evidence standard as every earlier phase: no manufactured result, no
> invented data, negative outcomes preserved forever.

## Mission

Deliver the AMD device surface of the one-package architecture:

```
one semantic core (scalar = SIMD = CUDA = ROCm)
  -> device::kernel_shared / device::entropy_shared (shared no_std)
  -> amdgcn-amd-amdhsa kernel entry   (this phase)
  -> backend::rocm presence probe     (this phase)
  -> scripts/build-rocm-device.sh     (this phase: clean code-object build)
  -> court rocm / probe rocm          (this phase: honest evidence)
  -> differential battery + D1 endpoint (Phase J, on ROCm hardware)
```

The development host has **no AMD GPU and no ROCm userspace**, so Phase I
cannot execute kernels. What it must deliver instead is the *complete,
correct, clean-building* AMD surface plus evidence that says exactly why
device execution is unavailable — with the same typed-cause discipline the
CUDA courts use for absent hardware.

## Toolchain reality (captured at entry freeze, I.0)

Measured on this host with the pinned nightly (`nightly-2026-07-24`,
rustc 1.99.0-nightly, LLVM 22.1.8):

| fact | measurement |
| --- | --- |
| `amdgcn-amd-amdhsa` target | known to rustc (target list), **no prebuilt rust-std** from rustup (`rustup target add` fails: "no prebuilt artifacts") |
| std for the target | built from source: `cargo -Z build-std=core` (requires the `rust-src` component of the same toolchain) |
| cdylib link (incremental) | rustc **ICE** `cannot find embedded bitcode` (fat-LTO over bitcode-less build-std rlibs) — avoided by `CARGO_INCREMENTAL=0` (documented, stable invocation) |
| `cargo --config 'lib.crate-type=…'` | does **not** override a manifest field → the code-object link uses direct `rustc` after the cargo build-std rlib step (mirrors the CUDA script's direct-rustc style) |
| kernel ABI | `extern "gpu-kernel"` (`feature(abi_gpu_kernel)`, already gated in `src/lib.rs` for `amdgpu`) |
| thread indexing | `core::arch::amdgpu::{workitem_id_x, workgroup_id_x, …}` under `feature(stdarch_amdgpu)` (added to the lib gates this phase); **no workgroup-count/size intrinsic exists** → launch geometry (`blocks_x`, `threads_x`) is passed as kernel parameters |
| artifact | ELF shared object, `AMD GPU` machine, e.g. `scripts/out/vole_audio.amdgcn.elf`; **per-ISA** (`-C target-cpu=gfx…`, baseline `gfx906`, env `VOLE_ROCM_GFX` overrides) — unlike the portable PTX text the CUDA path emits |

## Work items (executed in order)

- **I.1 Kernel entry** — `src/device/amdgcn_entry.rs` (was an empty Phase-I
  placeholder): three `extern "gpu-kernel"` wrappers mirroring the NVPTX
  surface exactly — `vole_render_d0`, `vole_entropy_decode`,
  `vole_upmix_mono_dup` — over the same shared `kernel_shared` /
  `entropy_shared` semantics; grid-stride loops with AMDGCN intrinsics and
  explicit launch geometry (`device::geom`).
- **I.2 Build script** — `scripts/build-rocm-device.sh` (was
  `NOT_IMPLEMENTED`): clean code-object build. Every build runs in a **fresh
  isolated per-toolchain/per-gfx target dir**
  (`target/vole-rocm/<rustc-commit>/<gfx>`, removed first), so stale
  core/compiler_builtins rlibs can never be linked and the glob is enforced
  to resolve to exactly one rlib. `--verify-deterministic` builds twice in
  two isolated dirs and requires `SHA256(A1) == SHA256(A2)`, writing
  `scripts/out/vole_audio.amdgcn.determinism.json` — byte-determinism is
  measured evidence. Artifact + `sha256` + provenance sidecar (entries,
  rustc/LLVM + commit hash, target/gfx, source-tree hash, dirty flag,
  determinism state, loadability disclaimer).
- **I.3 Backend probe** — `src/backend/rocm/` (`mod.rs`, `loader.rs`,
  `probe.rs`, `elf.rs`): a **loader probe** walking the exact Phase-J chain
  — AMD display GPU (sysfs) → bound to `amdgpu` → `/dev/kfd` accessible →
  HIP/HSA actually dlopen (ld cache + `ROCM_LIB_PATH` + `/opt/rocm*`
  candidates) with the required symbols → `INCONCLUSIVE_PENDING_EXECUTION`.
  Corrected taxonomy: userspace absence is `UNSUPPORTED_BY_API`, KFD
  present-but-inaccessible is `INCONCLUSIVE`, only device/driver absence is
  `UNSUPPORTED_BY_HARDWARE`; rocm-smi is auxiliary telemetry only.
  `elf.rs` validates code-object images from their bytes (ELF magic,
  ELF64 LE, `EM_AMDGPU`, required FUNC entries in `.symtab`) with hostile-
  file safety and no external tools.
- **I.4 Court + CLI** — `court rocm` is **two-dimensional and fails
  closed**: `compile_surface` (artifact present + ELF-valid + entries +
  provenance sidecar + artifact↔source-tree correspondence) and
  `runtime_surface` (the loader chain). A missing/malformed/unverifiable
  artifact is incapable of satisfying Phase I → `INCONCLUSIVE` regardless of
  the runtime verdict; an arbitrary file at `VOLE_ROCM_ARTIFACT` is never
  merely hashed. `probe rocm` prints the same classification.
- **I.5 Docs** — this charter; `PROJECT_STATE` Phase I section + next-work
  renumbering; `ARCHITECTURE` module map; `README`; `NON_CLAIMS`;
  `SEMANTIC_FACTS` rocm row note.
- **I.6 Evidence-binding amendment (review)** — repo-wide: `build.rs`
  stamps **compile-time** source identity into the host binary
  (compiled-from commit/tree/dirty/rustc/profile; `Environment.source_bound`
  requires compiled-from == executed-in-worktree, both clean — receipts
  record both and a seal requires them to match; `court-all.sh` rebuilds
  unconditionally and refuses a non-bound battery); `evidence::artifact`
  consumes the PTX/AMDGPU provenance sidecars into the CUDA/entropy court
  receipts; launch-geometry guards (`device::geom`) with the pathological
  battery (zero rejected, 1×1, 1×64, non-divisible, blocks>work, page ±1,
  max geometry).

## Claim boundary (what Phase I does NOT claim)

- No ROCm kernel was executed anywhere; no device observation exists.
- The code object is **compile evidence**: ELF AMDGPU code object for the
  recorded baseline `gfx`, byte-deterministic across isolated builds
  (measured by `--verify-deterministic`); loadability on a real device is
  **Phase J** evidence (needs ROCm runtime + matching hardware) and is never
  assumed.
- `court rocm` on a full ROCm stack classifies `INCONCLUSIVE` with the
  Phase-J reason — it does not run the differential battery, and Phase I
  never reports device `SUPPORTED`. Missing userspace is `UNSUPPORTED_BY_API`,
  never a hardware verdict.
- AMDGCN kernels share the scalar semantics exactly by construction (thin
  wrappers over the same no_std core); the scalar == ROCm *proof* is the
  Phase-J differential battery, not this phase's word count.
- Receipts bind the binary to its source: compiled-from (build.rs) must
  equal executed-in-worktree for a seal; artifact sidecars bind the GPU
  artifact back to the tree/toolchain that built it.

## Exit criteria (Phase I complete only when ALL hold)

1. `docs/PHASE_I.md` exists; the entry-freeze toolchain facts above are
   measured, not assumed.
2. `device/amdgcn_entry.rs` implements the three kernels as pure wrappers
   over shared semantics (no duplicated math, no per-backend forks),
   guarded by the host-tested `device::geom` launch contract (zero geometry
   rejected; pathological cases in the unit battery).
3. `scripts/build-rocm-device.sh` emits a real AMDGPU code object from a
   fresh isolated per-toolchain/per-gfx target dir on the pinned nightly
   from a clean tree (`git_dirty: false`), with exactly-one
   core/compiler_builtins enforcement, and `--verify-deterministic` proves
   byte-equality of two isolated builds (recorded in
   `vole_audio.amdgcn.determinism.json`).
4. `backend/rocm` loader/probe/elf modules compile under `--all-features`;
   the probe walks the runtime chain with the corrected taxonomy
   (unit-tested: no-GPU / not-amdgpu-bound / no-KFD → hardware; KFD
   inaccessible → inconclusive; HIP/HSA not loadable → API; full chain →
   inconclusive-pending-execution; rocm-smi alone is not a compute
   runtime).
5. `court rocm` fails closed: unsatisfied compile surface → `INCONCLUSIVE`;
   satisfied surface + hardware absence → typed runtime verdict; never a
   manufactured result. `probe rocm` reports the same classification.
6. All pre-existing A–H / H.2 courts remain SUPPORTED with frozen hashes
   unchanged; the full battery + `court rocm` re-seal clean-tree receipts
   with compiled-from == executed-in-worktree.
7. Host tests green (all-features), clippy `-D warnings` clean, fmt clean;
   MSRV 1.89.0 green.
8. Docs (`PROJECT_STATE`, `ARCHITECTURE`, `README`, `NON_CLAIMS`,
   `SEMANTIC_FACTS`, usage text) describe the phase without overclaiming.

## Seal ledger

Seals are appended in order; every seal is a clean-tree run with immutable
receipts committed separately; negative receipts are kept.

### Seal 1 — Phase I implementation + clean-tree battery (2026-09-08)

Implementation commits (in order): `df2f793` (amdgcn kernels, build script,
`backend::rocm` probe, `court rocm`, CLI), `c887850` (docs: this charter,
PROJECT_STATE section, ARCHITECTURE/README/NON_CLAIMS/SEMANTIC_FACTS).

Seal run (release, `--all-features`, clean tree `c887850`, `git_dirty: false`):

- 18 receipts sealed under `receipts/` (six A–H courts + ten H.2 courts +
  the `h2` aggregate + `rocm`), committed separately at `69838e8`.
- All pre-existing courts re-sealed SUPPORTED with frozen hashes unchanged
  (semantic `1791816f4b93…`, authored `f7e103f3a97d…`) — Phase I adds no
  semantic delta (exit criteria 6).
- `court rocm`: `UNSUPPORTED_BY_HARDWARE` with typed causes — no AMD display
  GPU in the sysfs PCI walk; no `/sys/class/kfd` or `/dev/kfd`; no ROCm
  userspace soname found. The clean-tree code object
  `scripts/out/vole_audio.amdgcn.elf` (93 040 bytes, sha256
  `e2ae95d0c15fb9aef4dc8a3f55a7ea11ccd1027d373e18267b256c53603eb69f`,
  entries `vole_render_d0`/`vole_entropy_decode`/`vole_upmix_mono_dup`,
  baseline gfx906) is receipted as compile evidence with its SHA-256.
- PTX artifact unchanged (the lib delta is amdgpu-inert): sha256
  `8b23325d03700847b056b29df4f4d4afd1a0c67386458512986c52f4fca7896b`.
- Host tests: 292 total (287 passed, 5 ignored) all-features on the pinned
  nightly; MSRV 1.89.0 green with the same all-features count; clippy
  `-D warnings` and `cargo fmt --check` clean.

### Seal 2 — review amendment: evidence binding + hardening (2026-09-08)

Delta since Seal 1 (five external-review items):

1. **Stale-binary receipt binding (repo-wide)** — `build.rs` stamps
   compile-time source identity into the host binary (compiled-from:
   commit/tree/dirty/rustc/profile; reruns every build so HEAD moves are
   tracked); every receipt now records both compiled-from and
   executed-in-worktree, and this seal is the first in which **every
   receipt is source-bound** (compiled-from == executed-in-worktree, both
   clean). `court-all.sh` rebuilds unconditionally and refuses a non-bound
   battery.
2. **ROCm probe taxonomy + loader chain** — the probe now dlopens the
   compute runtime (HIP/HSA) with required symbols (ld cache +
   `ROCM_LIB_PATH` + `/opt/rocm*` candidates); missing userspace is
   `UNSUPPORTED_BY_API`, KFD present-but-inaccessible is `INCONCLUSIVE`,
   rocm-smi is telemetry only.
3. **`court rocm` two-dimensional + fail-closed** — `compile_surface`
   (artifact present + ELF-valid AMDGPU code object + required kernel
   entries + provenance sidecar matching the attested tree) and
   `runtime_surface`; an unsatisfied compile surface is `INCONCLUSIVE`
   whatever the runtime says (self-defending `elf` validator, no external
   tools).
4. **Build isolation + determinism evidence** — fresh per-toolchain/per-gfx
   target dirs with exactly-one rlib enforcement;
   `--verify-deterministic` built twice in isolated dirs and proved
   `SHA256(A1) == SHA256(A2)` = `5092e129…`
   (`vole_audio.amdgcn.determinism.json`).
5. **Launch-geometry guards** — `device::geom` frozen contract with the
   pathological battery (zero rejected, 1×1, 1×64, non-divisible,
   blocks>work, page ±1, max geometry); the amdgcn kernels guard-return on
   invalid geometry.

Seal run (release, `--all-features`, clean tree `087da6b`, `git_dirty: false`):

- 18 receipts, committed separately at `550e7cc`; **every receipt
  source-bound** (compiled-from == executed-in-worktree == `087da6b`).
- All pre-existing courts SUPPORTED with frozen hashes unchanged (semantic
  `1791816f4b93…`, authored `f7e103f3a97d…`); the cuda/entropy artifact
  extras now consume the PTX provenance sidecar.
- `court rocm`: `UNSUPPORTED_BY_HARDWARE` with the compile surface
  **satisfied** — ELF machine AMDGPU, entries
  `vole_render_d0`/`vole_entropy_decode`/`vole_upmix_mono_dup` found in
  `.symtab`, sidecar matches the attested tree
  (`vole_audio.amdgcn.elf`, 93 096 bytes, sha256
  `5092e129b92c93f08de144058d470afc36b933c7fb908ac738ba4494473bb38e`,
  byte-deterministic) — and the runtime chain recorded (no AMD GPU; KFD
  absent; HIP/HSA dlopen attempts with reasons).
- PTX artifact unchanged: sha256 `8b23325d03700847b056b29df4f4d4afd1a0c67386458512986c52f4fca7896b`.
- Host tests: 313 total (308 passed, 5 ignored) all-features on the pinned
  nightly (303 total, 298 passed default-features); MSRV 1.89.0 green with
  the same all-features count; clippy `-D warnings` and
  `cargo fmt --check` clean.

## Execution record (implementation summary)

- Entry freeze captured above; artifact baseline
  `scripts/out/vole_audio.amdgcn.elf` sha256 `5092e129…` (gfx906,
  byte-deterministic across isolated builds).

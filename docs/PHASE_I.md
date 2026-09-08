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
  explicit launch geometry.
- **I.2 Build script** — `scripts/build-rocm-device.sh` (was
  `NOT_IMPLEMENTED`): preflight checks (rust-src), `-Z build-std=core`
  rlib, direct-rustc cdylib link with `-C target-cpu=$VOLE_ROCM_GFX`,
  DWARF-free opt profile, artifact + `sha256` + provenance sidecar
  (entries, rustc/LLVM, target/gfx, source-tree hash, dirty flag,
  loadability disclaimer). Deterministic: same tree + toolchain + gfx ⇒
  same artifact bytes.
- **I.3 Backend probe** — `src/backend/rocm/` (`mod.rs`, `probe.rs`):
  filesystem/sysfs presence probe (AMD display GPUs from the sysfs PCI
  walk, `/sys/class/kfd` + `/dev/kfd`, ROCm userspace sonames under the
  standard roots) with a **pure, unit-tested classifier** that never
  manufactures a verdict.
- **I.4 Court + CLI** — `court rocm` (registered in the court table and
  dispatcher) and `probe rocm`: typed evidence receipt (probe rows,
  artifact state + SHA, verdict). No kernel is executed by either path.
- **I.5 Docs** — this charter; `PROJECT_STATE` Phase I section + next-work
  renumbering; `ARCHITECTURE` module map; `README`; `NON_CLAIMS`;
  `SEMANTIC_FACTS` rocm row note.

## Claim boundary (what Phase I does NOT claim)

- No ROCm kernel was executed anywhere; no device observation exists.
- The code object is **compile evidence**: ELF AMDGPU code object for the
  recorded baseline `gfx`; loadability on a real device is **Phase J**
  evidence (needs ROCm runtime + matching hardware) and is never assumed.
- `court rocm` on a full ROCm stack classifies `INCONCLUSIVE` with the
  Phase-J reason — it does not run the differential battery, and Phase I
  never reports device `SUPPORTED`.
- AMDGCN kernels share the scalar semantics exactly by construction (thin
  wrappers over the same no_std core); the scalar == ROCm *proof* is the
  Phase-J differential battery, not this phase's word count.

## Exit criteria (Phase I complete only when ALL hold)

1. `docs/PHASE_I.md` exists; the entry-freeze toolchain facts above are
   measured, not assumed.
2. `device/amdgcn_entry.rs` implements the three kernels as pure wrappers
   over shared semantics (no duplicated math, no per-backend forks).
3. `scripts/build-rocm-device.sh` emits a real AMDGPU code object with
   provenance sidecar + sha file on the pinned nightly from a clean tree
   (`git_dirty: false`), and is byte-deterministic for the recorded inputs.
4. `backend/rocm` probe + classifier compile under `--all-features`,
   classify the four probe states correctly (unit-tested), and report
   typed causes on this host.
5. `court rocm` writes an honest receipt on this host
   (`UNSUPPORTED_BY_HARDWARE` with cause) and never a manufactured result;
   `probe rocm` reports the same classification.
6. All pre-existing A–H / H.2 courts remain SUPPORTED with frozen hashes
   unchanged; the full battery + `court rocm` re-seal clean-tree receipts.
7. Host tests green (all-features), clippy `-D warnings` clean, fmt clean;
   MSRV 1.89.0 green (the additions use no post-1.89 language features).
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
  nightly (282 total, 277 passed default-features; the +5 are the rocm probe
  classifier + court-path tests); MSRV 1.89.0 green with the same
  all-features count; clippy `-D warnings` and `cargo fmt --check` clean.
  (An earlier intermediate failure under `cargo test` was stale incremental
  state from mixed toolchains in one target dir; a clean rebuild resolved it
  and the sealed numbers above are from clean artifacts.)

## Execution record (implementation summary)

- Entry freeze captured above; artifact baseline
  `scripts/out/vole_audio.amdgcn.elf` sha256 `e2ae95d0…` (gfx906).

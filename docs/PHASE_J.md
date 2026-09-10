# PHASE J — ROCm D1 (differential scalar == ROCm battery + D1 endpoint experiment)

> Phase charter and seal ledger. Phase J follows the sealed Phase I and
> precedes Phase K (inverse compiler), exactly as the implementation
> contract sequences them. Phase J scope is the repository's own statement:
> **ROCm D1 — the differential scalar == ROCm battery and the endpoint
> experiment on the AMD surface** — executed to the same evidence standard
> as every earlier phase: no manufactured result, no invented data, negative
> outcomes preserved forever.

## Mission

Deliver the executable Phase-J runtime on top of the frozen Phase-I ABI
contract, and the two courts that exercise it:

```text
scalar oracle (authority)
        ↓
backend::rocm::ffi      HIP bindings over exactly HIP_D0_REQUIRED +
                        HIP_D1_ADDITIONAL (no symbol outside the frozen
                        tables is required)
backend::rocm::runtime  RAII session/module/function/device buffers +
                        host registration (hipHostRegister mapped)
backend::rocm::kernel   vole_render_d0 / vole_entropy_decode /
                        vole_upmix_mono_dup launches (grid-stride geometry
                        from device::geom, passed as kernel parameters)
        ↓
court rocm-d0           differential scalar == ROCm: frozen fixture
                        windows + entropy decode jobs + mono->stereo upmix
court rocm-d1           D1 endpoint: hipHostRegister(hipHostRegisterMapped)
                        over the real ALSA mmap ring, direct device writes,
                        equal-work D0 baseline for the bytes-removed claim
        ↓
typed verdicts          SUPPORTED only on D0/D1-ready ROCm hardware; the
                        typed chain is recorded wherever it breaks
```

The development host has **no AMD GPU and no ROCm userspace**, so Phase J —
like Phase I before it — cannot execute kernels here. What it delivers is
the *complete, correct, clean-building* Phase-J runtime and courts plus
evidence that says exactly why device execution is unavailable, with the
same typed-cause discipline the CUDA courts use.

## Claim boundary (what Phase J does NOT claim)

- No ROCm kernel has been executed anywhere; no device observation exists.
  `court rocm-d0`/`rocm-d1` on this host return `UNSUPPORTED_BY_HARDWARE`
  with the full runtime chain recorded; the positive paths exist in code
  and run when a D0/D1-ready ROCm stack is present.
- The code object remains **compile evidence** (ELF AMDGPU, per-ISA
  `gfx906` baseline, byte-deterministic across isolated builds, bound
  sidecar); loadability on a real device is validated only by an actual
  Phase-J execution on ROCm hardware.
- The HIP ABI surface is exactly the frozen Phase-I tables: D0
  (init/device/module/launch/malloc/memcpy/sync) and D1-additional (host
  registration). A stack that runs the D0 battery but refuses host
  registration classifies "ROCm D0 READY; D1 UNSUPPORTED_BY_API" — never
  "runtime unavailable". The direct-HSA path remains a probed fallback
  contract; native HSA launches are not implemented (the HIP runtime is the
  implemented Phase-J path).
- Per-decode/per-launch host-side allocations exist in the entropy decode
  path (scratch/status buffers per launch, mirrored from the CUDA path);
  this is not an allocation-free production RT path — recorded, never
  claimed.
- The D1 experiment's "bytes removed" claim is measured on equal work
  against the court's own D0 baseline (whole-object device render → one
  DtoH → per-chunk host copies), exactly as the Phase-H D1 courts do.

## Work items (executed in order)

- **J.1 HIP FFI** — `backend::rocm::ffi`: dlopen'd HIP bindings over the
  frozen `HIP_D0_REQUIRED`/`HIP_D1_ADDITIONAL` tables plus optional
  evidence extras (error strings, runtime/driver versions, device name).
  No symbol outside the tables is required; typed failures everywhere.
- **J.2 HIP runtime** — `backend::rocm::runtime`: RAII `Rocm` session
  (init/device select/identity evidence), `Module` (code object from
  bytes), `Function` (entry lookup + launch with the `device::geom`
  contract), `DeviceBuffer` (alloc/upload/download/free), and D1
  `HostRegistration` (`hipHostRegister(hipHostRegisterMapped)` +
  `hipHostGetDevicePointer` + unregister) with typed registration attempts
  and rc classification.
- **J.3 Kernel orchestration** — `backend::rocm::kernel`: `RocmWorld`
  (D0 render/render_direct/render_to/upmix) and `EntropyWorldRocm`
  (decode/decode_into) mirroring the CUDA host side; AMD geometry is
  passed as kernel parameters and equals the launch geometry.
- **J.4 `court rocm-d0`** — differential battery: semantic/authored/mixed
  frozen fixtures over multiple windows, entropy literal + exact-residual
  decode jobs, and the mono->stereo upmix transform — every window
  byte-compared against the scalar oracle. Typed negatives without a
  D0-ready device; compile surface required (fail closed).
- **J.5 `court rocm-d1`** — endpoint experiment: D0 baseline + D1
  stereo-direct + D1 mono-upmix sessions on the registered ALSA mmap ring,
  with the frozen zero-xrun falsification policy; per-session traffic and
  exposure cells (gpu→host, host copy, device block, endpoint observation,
  verification reads, the 2048-byte bounded mono intermediate).
- **J.6 Docs** — this charter; `PROJECT_STATE` Phase J section + next-work
  renumbering; `NON_CLAIMS`; `ARCHITECTURE` runtime note.

## Review amendment 1 (runtime safety, before Seal 2)

The first external review of the Phase-J runtime found real host-runtime
correctness defects (not provenance polish); all are fixed here:

1. **Structural ownership (`Arc<HipApi>`).** `Fns` is `Copy`, so resources
   could previously outlive the `Lib` that owns the resolved pointers
   (e.g. `drop(rocm); drop(buf)` would call a copied `hipFree` through an
   unloaded library), and a `Function` could outlive its `Module`. One
   `HipApi` now owns `Lib + Fns` behind an `Arc`; `Module`/`Function`/
   `DeviceBuffer`/`HostRegistration` each retain it (`Function` additionally
   retains the module owner), so the library and module lifetimes are
   enforced by the type system — no field reordering can invalidate them.
   `EntropyWorldRocm`'s destruction order therefore stops mattering.
2. **Exact shared launch Grid.** `render_args()` previously derived the
   `blocks_x`/`threads_x` kernel parameters from `max_frames` while the
   launch used the window grid (512 stereo frames: launch 4 blocks vs
   parameter 2). One `window_grid(frames, channels)` now feeds BOTH the
   launch and the parameter marshalling; a unit test pins
   `window_grid(512,2) == Grid::new(4,256)` and the equality for every
   court window size.
3. **HIP-specific D0 gate.** `RocmProbe::classify()` accepts any D0-ready
   row (HIP *or* HSA); `court rocm-d0` must not be authorized by an
   HSA-ready/HIP-unavailable system. `phase_j_d0_gate()` gates on a
   `libamdhip64` D0-ready row only and reports `UNSUPPORTED_BY_API` with
   the reason otherwise (unit-tested with an HSA-only probe). The battery
   is also wrapped so an operational failure becomes a typed INCONCLUSIVE
   gate receipt instead of a propagated error that skips evidence.
4. **`hipModuleUnload` joins the frozen D0 surface.** The runtime unloads
   modules in ordinary teardown; the symbol is now in `HIP_D0_REQUIRED` and
   in the `d0_missing` mapping, so “no required symbol outside the frozen
   table” is literally true.
5. **ROCm 7 discovery, single-sourced.** `libamdhip64.so.7` is probed, the
   candidate builder scans every explicit ROCm library dir (`ROCM_LIB_PATH`
   entries + `/opt/rocm*/lib{,64}`) by family prefix, and
   `probe::HIP_SONAMES` is the single source shared by the probe and the
   opener (`ffi`'s duplicate list is gone). A hostile test proves a
   `.so.7`-only install in `ROCM_LIB_PATH` is discovered, including via the
   unversioned-soname row.
6. **D1 verdict semantics aligned with CUDA.** Once an endpoint produced a
   D1 session, only that endpoint's cells are verdict-bearing
   (`aggregate_verdict`); other candidates' open/registration failures stay
   in the receipt as trial evidence. Unit tests: candidate A unsupported +
   candidate B exact D1 success = top-level SUPPORTED with A preserved;
   a completed-but-failed D1 session is the verdict.

Seal 2 re-runs the clean-tree battery with these fixes.

## Seal history

### Seal 1 — Phase J implementation + clean-tree battery (2026-09-09)

Seal run (release, `--all-features`, clean tree `3807428`, version 0.7.0):

- 20 receipts, committed separately at `0e56bd3`; every receipt
  `source_binding: bound` and carries `seal_subject_hash =
  ba5b5a1c7cc1a4ce9a0e6eac2178c08191b3c15cd2cf748102b69645baf55094`;
  `vole-audio seal verify` PASSes on the 10-row matrix (the eight prior
  rows plus `rocm-d0` and `rocm-d1`) in **default mode** at the battery
  tree and (after rebuilding from the release head, which only adds
  receipts/docs) at the head itself.
- All pre-existing courts SUPPORTED with frozen hashes unchanged (semantic
  `1791816f4b93…`, authored `f7e103f3a97d…`).
- `court rocm` / `rocm-d0` / `rocm-d1`: `UNSUPPORTED_BY_HARDWARE` with the
  compile surface satisfied and bound (artifact `5092e129…` == sidecar ==
  both determinism shas) and the typed runtime chain recording no AMD
  compute candidate on this host — the Phase-J differential battery and
  the D1 endpoint experiment do not execute here, and no receipt claims
  they did.
- PTX artifact unchanged: sha256
  `8b23325d03700847b056b29df4f4d4afd1a0c67386458512986c52f4fca7896b`.
- Host tests: 340 total (335 passed, 5 ignored) all-features on the pinned
  nightly (330 total, 325 passed default-features; +8 over Seal 8: the
  HIP ffi/runtime/kernel unit battery — marshalling layout, frozen-surface
  coverage, launch-geometry contract, entropy descriptor layout,
  registration classifier); clippy `-D warnings` and
  `cargo fmt --check` clean.

(Seal runs are appended here as they are produced; Seal 1 records the
clean-tree battery on this host, where the AMD surface is compile evidence
and the runtime chain is typed to the missing device. The positive paths
of `rocm-d0`/`rocm-d1` execute when a D0/D1-ready ROCm stack + AMD device
are present.)

### Seal 2 — review-1 closure: runtime safety fixes (2026-09-09)

Seal run (release, `--all-features`, clean tree `dd87c39`, version 0.7.1):

The six review-1 findings (structural `Arc<HipApi>` ownership, exact shared
launch grid, HIP-specific D0 gate, `hipModuleUnload` in the frozen D0
surface, ROCm 7 / explicit-dir soname discovery, CUDA-aligned D1 verdict
aggregation) are fixed as recorded in “Review amendment 1” above; each has
host-only hostile tests (no device required).

- 20 receipts, committed separately at `0ce17fe`; every receipt
  `source_binding: bound` and carries `seal_subject_hash =
  28651951d51d4dcba64b01be99664e03939eb7b438354337773009069c3e5eb9`;
  `vole-audio seal verify` PASSes on the 10-row matrix in **default mode**
  at the battery tree and (after rebuilding from the release head) at the
  head itself.
- All pre-existing courts SUPPORTED with frozen hashes unchanged (semantic
  `1791816f4b93…`, authored `f7e103f3a97d…`).
- `court rocm` / `rocm-d0` / `rocm-d1`: `UNSUPPORTED_BY_HARDWARE` with the
  compile surface satisfied and bound (artifact `5092e129…` == sidecar ==
  both determinism shas) and the typed runtime chain (no AMD compute
  candidate on this host).
- PTX artifact unchanged: sha256
  `8b23325d03700847b056b29df4f4d4afd1a0c67386458512986c52f4fca7896b`.
- Host tests: 349 total (344 passed, 5 ignored) all-features on the pinned
  nightly (339 total, 334 passed default-features; +9 over Seal 1: the
  Arc-lifetime graph, exact launch-grid equality for every court window,
  HSA-only gate rejection, frozen-D0 `hipModuleUnload`, `.so.7`-only
  discovery, and the D1 aggregation pair); clippy `-D warnings` and
  `cargo fmt --check` clean.

Procedural note (evidence hygiene): the first attempt at Seal 2 ran with an
untracked research PDF (`vole_audio_prior_art.pdf`) present at the repo
root, which made the worktree (and the binaries/artifacts built in it)
dirty; the gate correctly refused the resulting receipts (they were never
committed and were discarded). The PDF was moved under the ignored
`research/` tree — its designated home; the seal subject is unchanged —
and Seal 2 was re-run clean.

## Execution record (implementation summary)

- HIP/runtime/kernel and both courts compile clean on the pinned nightly
  (host `std`); the backend unit battery (marshalling layout, frozen
  surface coverage, launch-geometry contract, entropy descriptor layout,
  registration classifier) runs without a GPU.
- Artifact baseline: `scripts/out/vole_audio.amdgcn.elf` sha256
  `5092e129…` (gfx906, byte-deterministic across isolated builds);
  `scripts/out/vole_audio.ptx` sha256 `8b23325d…`.

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

## Execution record (implementation summary)

- HIP/runtime/kernel and both courts compile clean on the pinned nightly
  (host `std`); the backend unit battery (marshalling layout, frozen
  surface coverage, launch-geometry contract, entropy descriptor layout,
  registration classifier) runs without a GPU.
- Artifact baseline: `scripts/out/vole_audio.amdgcn.elf` sha256
  `5092e129…` (gfx906, byte-deterministic across isolated builds);
  `scripts/out/vole_audio.ptx` sha256 `8b23325d…`.

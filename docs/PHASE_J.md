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

(Seal runs are appended here as they are produced; Seal 1 records the
clean-tree battery on this host, where the AMD surface is compile evidence
and the runtime chain is typed to the missing device.)

## Execution record (implementation summary)

- HIP/runtime/kernel and both courts compile clean on the pinned nightly
  (host `std`); the backend unit battery (marshalling layout, frozen
  surface coverage, launch-geometry contract, entropy descriptor layout,
  registration classifier) runs without a GPU.
- Artifact baseline: `scripts/out/vole_audio.amdgcn.elf` sha256
  `5092e129…` (gfx906, byte-deterministic across isolated builds);
  `scripts/out/vole_audio.ptx` sha256 `8b23325d…`.

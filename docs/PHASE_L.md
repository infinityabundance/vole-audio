# PHASE L — GPU inverse search

## Mission

Place the inverse compiler's dominant search cost on multiple surfaces and
prove that placement is a **performance** question only. Deliverables
(implementation contract §34):

* parallel candidate sweeps;
* CPU vs CUDA vs ROCm comparison.

The contract's own rules for this phase:

> "But do not make GPU search necessary to decode. Search output is only a
> proposal. Every accepted candidate must be reverified by the normative exact
> evaluator. Measure CUDA/ROCm against CPU. If GPU does not win for a search
> family, keep the faster implementation."

## What is actually parallelized

The compiler's bounded proposal search spends nearly all its time in the
**bounded period scan**: for every candidate period `p`, how many frames
disagree with the periodic hypothesis `cycle = X[0..p]`? For mono content that
count is exactly the periodic residual record count
(`ResidualModel::Periodic` is mono, so channel 0's delta is
`X[f] - cycle[f % p]`, zero iff the frames agree), and the count that cannot
close is exactly the set `Residual::closing_residual` rejects.

So the scan is one independent computation per period — embarrassingly
parallel — and it is the *only* thing Phase L places on a device. Candidate
construction, exact acceptance, pricing and the Pareto frontier all stay on
the host.

## Surfaces

| surface | implementation | notes |
| ------- | -------------- | ----- |
| `scalar` | `inverse::search::scan_scalar` | sequential reference |
| `parallel` | `inverse::search::scan_parallel` | host threads over disjoint period ranges |
| `cuda` | `backend::cuda::SearchWorld` → `vole_period_scan` | one device thread per period |
| `rocm` | `backend::rocm::SearchWorldRocm` → same entry in the AMDGPU code object | frozen `device::geom` launch contract |

All of them call the same shared, `no_std`
`device::search_shared::period_records` function, so their agreement is
structural rather than a coincidence — and `court inverse-search` re-verifies
it for every surface that executes on the host. (`scalar` and `parallel` are
host surfaces and always execute; `cuda` executes when the PTX artifact and a
device are present; `rocm` executes only when an AMD compute candidate passes
the Phase-J HIP D0 gate, and is otherwise a compile-only, hardware-pending
surface. The receipt states the executed-surface count explicitly.)

### Placement policy

The default is `SearchPlacement::Auto`: a scan whose exact comparison count
(`P*F - P*(P+1)/2`) is below `PARALLEL_SCAN_THRESHOLD` runs on the scalar
reference surface, and a larger scan runs on the host-parallel surface, whose
placement has been measured to win. `SearchPlacement::Scalar` remains
available as the reference placement, and the device surfaces are explicit
court surfaces — never selected implicitly — because they lost the measured
family on this host.

## CUDA resource identity: lifetime **and** affinity

Phase J gave HIP a structural resource graph (every resource retains its
owning API handle). Phase L's review showed the CUDA port had only half of the
same property, and that the second half is a *different* axis:

```text
owning/session handle dropped while a buffer lives   →  lifetime (fixed)
buffer used while another context is current        →  affinity  (this fix)
buffer used on a thread that never entered it       →  affinity  (this fix)
```

CUDA's driver API operates on the calling thread's **current** context, so
lifetime alone is not enough. `CudaContext` now owns

```text
Enter  = cuCtxGetCurrent  → (no-op if already current | cuCtxSetCurrent)
Guard  = current value saved, restored on drop
```

and every context-dependent operation — allocation, copies, launches,
stream/event/graph work, module load and function lookup, synchronize, D1 host
registration — begins with that guard. Resource identity is therefore
`driver lifetime + context identity + current-thread affinity`.

Two details are deliberate, and both are empirical rather than assumed:

* the guard uses `cuCtxSetCurrent` and restores the saved value, **not**
  `cuCtxPushCurrent`/`cuCtxPopCurrent`. A context created by `cuCtxCreate` is
  already on the calling thread's context stack, and pushing it again returns
  `CUDA_ERROR_INVALID_CONTEXT` (rc 201, reproduced on driver 610.57.04 against a
  minimal C driver-API program) — so push/pop would fail for exactly the
  contexts this program creates;
* every API that used to take a raw `CUstream` now takes a `&Stream` whose
  context identity is checked (`Arc::ptr_eq`), so a stream from context B
  cannot be handed to a resource owned by context A. The raw handle is private
  to the driver module.

Entering costs one `cuCtxGetCurrent` in the single-context single-thread case
(the whole forward/entropy/D0/D1 path), which is why the ordinary courts are
unaffected — their counts and frozen hashes are unchanged.

## The two invariants the court asserts

1. **Placement has no semantics.** Every surface must produce identical
   per-period counts, hence an identical ranked period list.
2. **Search output is only a proposal.** Feeding an externally ranked period
   list into the compiler (`inverse::compile_with`) must produce exactly the
   same accepted candidate set as the sequential scan, and every accepted
   candidate must pass the full exactness battery (intrinsic closure + scalar
   oracle observation + bounded seek). A ranking can change *which* periodic
   hypotheses are tried; it can never manufacture an acceptance.

The court additionally records the measured wall time of each surface and the
implied ratio. It makes **no** claim that the GPU wins: work per period is
`O(frames - p)` (a period `p` compares frames `p..frames`), so a scan of periods
`1..=P` does `P*F - P*(P+1)/2` comparisons; the balance therefore depends on the
period bound and the window length. The host-parallel surface is **consistently
faster** than the sequential reference (measured ~0.22–0.25× across the sealed
runs). The CUDA ratio is **not stable on this host**: across release runs it
ranged from ≈0.21× to ≈1.78×, dominated by GPU clock state (the device idles at
~720 MHz of a 3105 MHz ceiling while a scan is only ~2 M comparisons per
fixture), so the court claims no stable device comparison. The default placement
therefore selects the host surface and the device path is kept as a
verified-equal surface.

## Seal history

### Seal 1 — Phase L implementation + clean-tree battery (2026-09-10)

> **Superseded in part by Seal 2.** The complexity explanation below
> (`O(frames / p)`) is wrong and is corrected to `O(frames - p)`; the GPU/CUDA
> ratio recorded here is one clock-state-dependent sample of a noisy
> measurement (Seal 2 reports the observed range); and the default placement is
> no longer fixed at `Scalar` but `Auto` (still the host surface, chosen by
> measurement). The Seal 1 numbers are kept as the measurement record, not as a
> current claim.

Seal run (release, `--all-features`, clean tree `3ddf66f`, version 0.9.0):

- `court inverse-search` SUPPORTED: 14 fixtures × 512 candidate periods on
  scalar / 16-thread parallel / CUDA, with the CUDA counts **exactly equal**
  to the scalar counts on all 14 fixtures and 80 accepted candidates
  re-verified by the exact evaluator, and the device-ranked period list
  producing exactly the sequential accepted set.
- **Measured on the release build** (this is the number that governs the
  policy): parallel **0.228×** and CUDA **1.779×** the sequential scan wall
  time — i.e. the 16-thread host scan is ~4.4× faster than sequential, and the
  CUDA scan is **~1.8× slower** than the sequential host scan (the per-launch
  and host↔device transfer overhead dominates a scan this small: ~2 M frame
  comparisons per fixture). An earlier debug-build run showed the opposite
  ordering; measuring on the release build is what makes the result usable.
  Per the contract ("if GPU does not win for a search family, keep the faster
  implementation") the default placement stays on the host (`inverse::compile`
  is sequential; `scan_parallel` is available), and the device path is kept as
  a verified-equal surface rather than adopted. No claim is made that the GPU
  wins at larger period bounds — that is untested here.
- `court rocm`: the ROCm search entry (`vole_period_scan`) is part of the
  AMDGPU compile surface and the artifact is rebuilt with four entries
  (`vole_render_d0`, `vole_entropy_decode`, `vole_upmix_mono_dup`,
  `vole_period_scan`); the ROCm execution row is a typed
  `UNSUPPORTED_BY_HARDWARE` (no AMD compute candidate on this host), so no
  AMD execution is pretended.
- Artifacts change because the kernel surface grew: PTX `8b23325d…` →
  `d13d22c3…`, AMDGPU `5092e129…` → `5c30a4bc…` (still byte-deterministic
  across isolated builds), both rebuilt from the clean tree.
- All pre-existing courts SUPPORTED with frozen hashes unchanged (semantic
  `1791816f4b93…`, authored `f7e103f3a97d…`, inverse `217b09a7…`).
- Host tests: **387 passed, 5 ignored** all-features (377 passed, 5 ignored
  default-features; +15 over Phase K Seal 3: the shared period-scan semantics,
  the scalar/parallel scan equality battery, the ranking unit battery, and the
  proposal-source invariance tests); clippy `-D warnings` and
  `cargo fmt --check` clean.
- 23 receipts, every one `source_binding: bound` and carrying
  `seal_subject_hash =
  270df703f42aa049b53b8ce4ca2482a5e15ffcb6ed9493bb23c31d29ec6fedc7`;
  `vole-audio seal verify` PASSes the 13-row matrix in **default mode** at the
  battery tree and at the release head.

### Seal 2 — review-1 closure (2026-09-10)

Review-1 findings closed in implementation (no redesign): CUDA structural
ownership, the `O(frames - p)` complexity model, the full-corpus ROCm battery,
rank-preserving external period lists, a placement policy that actually uses the
faster host surface, and truthful executed-surface wording. See the Phase-L
commit for the detail.

Seal run (release, `--all-features`, clean tree `3cb4a6d`, version 0.10.0):

- `court inverse-search` SUPPORTED: 14 fixtures × 512 candidate periods on
  **3 execution surfaces** (scalar, 16-thread parallel, CUDA) **+ 1 compile-only
  hardware-pending ROCm surface**, with the CUDA counts exactly equal to the
  scalar counts on all 14 fixtures, 80 accepted candidates re-verified by the
  exact evaluator, and the device-ranked period list producing exactly the
  sequential accepted set. The receipt reports the executed-surface count from
  actual execution and the frozen `d966d98e…` result hash is unchanged.
- **Measured ratios are honest but noisy.** In this sealed run: scalar 27.24 ms,
  parallel 5.99 ms (**0.220×**), CUDA 8.51 ms (**0.312×**). The host-parallel
  surface is consistently faster than sequential; the CUDA ratio is not stable
  — four consecutive release runs measured 0.90×, 0.41×, 0.30×, 0.31× (and Seal
  1 measured 1.78×), dominated by GPU clock state (720 MHz idle of 3105 MHz max
  for a ~2 M-comparison scan). No stable GPU comparison is claimed, and the
  default placement keeps the host surface.
- `court inverse` still reports **6 non-literal explanations** with the frozen
  static result `217b09a7…`; the accepted set is unchanged by the placement
  policy (verified).
- Device artifacts are **byte-identical to Seal 1** (the change was host-side
  only): PTX `d13d22c3…`, AMDGPU `5c30a4bc…`, both rebuilt from the clean
  `3cb4a6d` tree and still byte-deterministic across isolated builds
  (`source_tree_sha = ea614998…`, `source_dirty = false`).
- ROCm rows remain typed `UNSUPPORTED_BY_HARDWARE` (no AMD compute candidate on
  this host); `court rocm` compile surface satisfied, runtime a typed negative.
- All pre-existing courts SUPPORTED with frozen hashes unchanged (semantic
  `1791816f4b93…`, authored `f7e103f3a97d…`, inverse `217b09a7…`).
- Host tests: **391 passed, 7 ignored** all-features (381 passed, 7 ignored
  default-features; +4 over Seal 1: the rank-order normalization test, the
  closed-form comparison-count test, the Auto placement-policy test, and the
  every-placement-agrees test; +2 ignored CUDA ownership regressions gated on a
  device); clippy `-D warnings` and `cargo fmt --check` clean.
- 23 receipts, every one `source_binding: bound` and carrying
  `seal_subject_hash =
  02c8157e41e0be6381c9a4a04bdf6c61bea8d23c73889c02b2fabf65da7811a4`;
  `vole-audio seal verify` PASSes the 13-row matrix in **default mode** at the
  battery tree and at the release head.

## Execution record (implementation summary)

- The device surface grows by exactly one entry and no per-backend semantics:
  `device::search_shared` is the single implementation, and the NVPTX/AMDGCN
  entries are thin ABI wrappers.
- Search placement has zero decoder authority and cannot affect the archive:
  `inverse::compile` ranks its periodic candidates through
  `SearchBudget::placement` (`Auto` by default, which selects the measured-faster
  host surface), and `compile_with` differs only in which periodic hypotheses
  are proposed.
- ROCm compute is still unexecuted on this host; the court reports that as a
  typed negative and the compile surface carries the new entry. When AMD
  hardware is present the ROCm row runs the **same 14-fixture battery** as CUDA
  (a single-fixture smoke test would let an all-zero kernel pass, because the
  first fixture is silence).
- **CUDA resource identity is now lifetime *and* affinity.** Every GPU
  resource retains an `Arc<CudaContext>` (driver + context + device) so a
  context cannot be destroyed under a live resource, and every context-
  dependent operation makes its owner the calling thread's current context
  (`cuCtxSetCurrent`, restoring the previous value) so a resource cannot be
  used while another context is current or from a thread that never entered it.
  Stream-taking APIs take `&Stream` and check context identity instead of a raw
  stream handle. Gated regressions on the RTX 4080 prove: a buffer from A is
  usable while B is current and B is restored afterwards; a foreign thread can
  use a buffer it did not open; cross-context pairings are rejected.

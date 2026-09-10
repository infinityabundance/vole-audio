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

All four call the same shared, `no_std` `device::search_shared::period_records`
function, so their agreement is structural rather than a coincidence — and
`court inverse-search` re-verifies it anyway.

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
`O(frames / p)`, so the balance depends on the period bound and the window
length, and the receipt carries the numbers as they came out.

## Seal history

### Seal 1 — Phase L implementation + clean-tree battery (2026-09-10)

Seal run (release, `--all-features`, clean tree at the implementation commit,
version 0.9.0):

- `court inverse-search` SUPPORTED: 14 fixtures × 512 candidate periods on
  scalar / 16-thread parallel / CUDA, with the CUDA counts **exactly equal**
  to the scalar counts on all 14 fixtures and 80 accepted candidates
  re-verified by the exact evaluator. Measured on this host: parallel 0.198×
  and CUDA 0.406× the sequential scan wall time (CUDA includes host↔device
  transfers) — the GPU wins here, and the number is reported rather than
  claimed as architecture.
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
  the scalar/parallel scan equality battery, the ranking unit battery, and
  the proposal-source invariance tests); clippy `-D warnings` and
  `cargo fmt --check` clean.
- `vole-audio seal verify` PASSes the 13-row matrix in **default mode** at the
  battery tree and at the release head.

## Execution record (implementation summary)

- The device surface grows by exactly one entry and no per-backend semantics:
  `device::search_shared` is the single implementation, and the NVPTX/AMDGCN
  entries are thin ABI wrappers.
- Search placement has zero decoder authority and cannot affect the archive:
  `inverse::compile` (sequential) remains the default path, and `compile_with`
  differs only in which periodic hypotheses are proposed.
- ROCm compute is still unexecuted on this host; the court reports that as a
  typed negative and the compile surface carries the new entry.

# Performance

This file contains actual measurements only. Every row names its method,
hardware, backend, and workload; receipts under `receipts/` bind each run to
its environment. Flagship measurements (the frozen-corpus ladder, real-time
deadlines, interference) arrive with Phase M; rows below are **fixture-level**
evidence from the Phase F SIMD courts and the Phase G CUDA D0 court and are
labeled as such — they are indicative of the backend shape, not a claim about
the flagship corpus.

Terms that will not appear here: *revolutionary*, *10x*, *near-zero latency*,
*orders of magnitude*, *GPU is faster* (before the GPU courts run).

## Method

- Host: AMD Ryzen 7 9800X3D, `performance` governor, single thread, warm cache.
- Release profile (`opt-level 3`, `lto thin`), toolchain pinned in
  `rust-toolchain.toml`.
- Timing: host `CLOCK_MONOTONIC`-class wall time (`std::time::Instant`),
  whole-observation end-to-end, 7 runs per row (warmup run discarded), median
  reported. Not a real-time deadline measurement (Phase M owns those).
- Workloads: deterministic worlds built from frozen fixtures — see
  `src/courts/simd.rs` (`mixed_world`) and the semantic/authored court
  fixtures. Hash of every observation reproduced exactly by every backend.
- Execution surfaces (all bit-identical, enforced by `court simd` and
  `court cuda`):
  - `scalar` — the planned scalar floor of the SIMD engine (semantic
    authority; itself 2–3× faster than the naive per-frame oracle loop);
  - `avx2` — 4×i64-lane kernels (64-bit ops emulated);
  - `avx512` — 8×i64-lane kernels (`vpmullq`, AVX-512F/DQ/VL).
  - `cuda-d0` — the Phase G D0 buffered-diagnostic kernel (RTX 4080 SUPER;
    full path = state upload + launch + sync + device→host copy-back;
    kernel-only time via driver events).
  - Partial-bank and residual-governed voices render through the exact shared
    scalar path on every floor (documented, not hidden): those classes are
    compute-bound on the *per-partial* loop, which the kernels do not yet
    vectorize.

## Phase F fixture-level throughput (2026-09, CachyOS, Zen5)

Whole-window single-threaded wall time; speedup relative to the planned
scalar floor of the same window.

| Workload | frames | scalar | avx2 | avx512 | avx512 speedup |
|---|---|---|---|---|---|
| 256 literal voices, stereo, loop+linear+env (P1) | 65 536 | 225.6 ms | 132.9 ms | 77.4 ms | 2.9× |
| 1024 oscillator voices, envelope churn (P3) | 8 192 | 42.6 ms | 39.1 ms | 20.9 ms | 2.0× |
| 512 noise voices (P4) | 65 536 | 308.0 ms | 110.7 ms | 57.7 ms | 5.3× |
| 256 wavetable-cycle voices, release envelopes (P5) | 20 000 | 38.9 ms | 28.1 ms | 15.2 ms | 2.6× |
| 64 × 256-partial banks (P2) | 65 536 | 1.770 s | 1.761 s | 1.759 s | 1.0× |

Workload definitions and the verifying hashes are in the Phase F probe
(removed after measurement; worlds reproducible from `court simd` fixtures +
the parameter tables above). P2 is the partial-bank class: every floor calls
the exact shared per-partial semantics, so 1.0× is the honest, expected row —
it documents the CPU baseline the GPU partial path will be compared against.

## Where the CPU baseline stands

- Exact scalar floor: ~3–7 M voice-frame·channel observations per second at
  256+ voices on this host (P1/P3/P5 shapes), before vector kernels.
- AVX-512 kernels multiply that by 2–5× on the vectorized classes; the
  crossover vs an optimized AVX-512 CPU sampler and the GPU lies with
  Phase M's frozen corpus.

## Phase G fixture-level throughput (2026-09, same host + RTX 4080 SUPER)

All rows below come from `court cuda` receipts under `receipts/cuda/`;
frames = 8192 per window unless noted; **every cell was verified bit-exact
(scalar == CUDA; cuda == scalar on the same world) before timing**; CUDA
`wall` = full D0 path (state upload + launch + stream sync + DtoH copy-back),
`kernel` = driver-event kernel time. Fixture-level only — not flagship
numbers (Phase M owns the frozen corpus).

| fixture (mono) | scalar | avx2 | avx512 | cuda-d0 wall | cuda kernel |
|---|---:|---:|---:|---:|---:|
| literal 256v | 40.0 ms | 11.4 ms | 6.1 ms | 2.1 ms | 2.0 ms |
| noise 512v | 96.6 ms | 13.4 ms | 7.1 ms | 2.7 ms | 2.8 ms |
| wavetable 256v | 38.3 ms | 11.4 ms | 6.1 ms | 2.6 ms | 2.5 ms |
| oscillator 1024v | 143.3 ms | 39.8 ms | 20.9 ms | 11.0 ms | 10.3 ms |
| partials 32v × 256 | 111.7 ms | 107.2 ms | 106.6 ms | 16.9 ms | 13.4 ms |

Voices × quantum crossover cells (mean wall ms):

| fixture | frames | scalar | avx512 | cuda-d0 wall |
|---|---:|---:|---:|---:|
| noise 64v | 512 | 0.747 | 0.060 | 0.193 |
| noise 64v | 8192 | 11.9 | 0.88 | 0.196 |
| noise 1024v | 512 | 12.0 | 0.95 | 3.03 |
| noise 1024v | 8192 | 191.1 | 14.1 | 3.10 |
| oscillator 64v | 8192 | 8.9 | 1.3 | 0.37 |
| partials 64v × 256 | 512 | 13.8 | 13.4 | 19.7 |
| partials 32v × 256 | 8192 | 111.7 | 106.6 | 16.9 |

Read honestly: at a 512-frame quantum the CPU (AVX-512) wins every cell —
launch + copy-back overhead dominates the small D0 window. At 8192 frames the
CUDA D0 wall beats AVX-512 by ~2–6× on these fixtures, including the
partial-bank class where the CPU SIMD floors gain nothing (106.6 ms CPU vs
16.9 ms CUDA wall). This is the shape of the crossover surface the Phase M
corpus will measure properly — the numbers above are fixture-level only, not
a claim about real-time deadlines or endpoint directness (no D1 path exists
until Phase H).

## Real-time and deadline evidence

None yet (Phase M). This file says so rather than guessing.

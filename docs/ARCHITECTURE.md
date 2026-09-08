# Architecture

The current VOLE-Audio architecture paper is v1.1 (DOI
10.5281/zenodo.22666746), which makes the entropy-native layer explicit;
the original v1.0 broad prior-art disclosure (DOI
10.5281/zenodo.22649073) remains the earlier statement. This file is the
implementation's own architecture map; it exists to make the code
navigable, not to re-litigate the paper.

## 1. One semantic core, many execution surfaces

The arithmetic-heavy semantics live in modules that compile unchanged for:

| execution surface        | target                         | build path                                    |
| ------------------------ | ------------------------------ | --------------------------------------------- |
| scalar reference (host)  | `x86_64-unknown-linux-gnu`     | `cargo build` (default `std`)                 |
| SIMD (host)              | same, runtime-dispatched       | `cargo build --features std`                  |
| NVIDIA device            | `nvptx64-nvidia-cuda`          | `scripts/build-cuda-device.sh` (`no_std`)     |
| AMD device               | `amdgcn-amd-amdhsa`            | `scripts/build-rocm-device.sh` (`no_std`, build-std=core)     |

Rule: **semantic duplication is forbidden; only tiny target entry points
differ.** The scalar evaluator owns semantic authority. Every other surface is
differentially tested against it (scalar == SIMD == CUDA == ROCm) for exact
supported features, and every backend's observation is compared by hash.

## 2. Module map (what lives where)

- `status.rs` — the nine evidence verdicts + `NOT_IMPLEMENTED`, shared between
  host receipts and device status words (single `u8` vocabulary).
- `limits.rs` — hostile-input ceilings; compiled into device builds unchanged.
- `hash/` — in-repo SHA-256; content identity + integrity + trace hashing.
- `universe/` — `vole.audio.u1` frozen semantics: time, sample, phase,
  arithmetic, PRNG, event ordering, clock, layout, observation.
- `object/` — the SampleObject model: descriptor, literal, wavetable,
  oscillator, partials, repeated, reference, residual, graph, checkpoint.
- `sampler/` — voices, world, scheduler, envelope, gain/pan/rate,
  interpolation, resampler, mix, filter.
- `eval/` — evaluator surfaces: `common`, `scalar`, `simd`. These call the
  same per-contribution functions; only traversal/reduction differs (safe
  because mixing is order-independent i64 accumulation with one final
  saturation).
- `device/` — GPU ABI: `kernel_shared` (descriptors + layout shared with
  host), `nvptx_entry`, `amdgcn_entry` (thin kernels only; AMDGCN entry
  lands in Phase I, geometry passed as kernel parameters).
- `backend/` — host runtimes: `cuda/` (Driver API via audited FFI + dlopen),
  `rocm/` (Phase I: filesystem/sysfs presence probe + artifact contract;
  the HIP/HSA launch runtime is Phase J, where ROCm hardware validates it).
  Probes, memory, streams, graphs, direct paths.
- `audio/` — ALSA endpoint (mmap discipline), endpoint clock, directness
  (D0..D3), topology. `directness.rs` and `topology.rs` are pure vocabulary;
  the endpoint implementation is deliberately separate.
- `format/` — canonical binary `.voleaudio` archive (explicit encoding, no
  bincode/serde-normative), WAV ingest, manifest.
- `inverse/` — bounded proposal search + residual closure + Pareto frontier.
- `transport/` — deterministic framing of OBJECT/EVENT/STATE/CHECKPOINT/...;
  integrity; recovery.
- `evidence/` — `vole.audio.evidence.v1` receipts, counters, timing, energy,
  environment, hardware, trace.
- `courts/` — the executable courts (`court semantic`, `court cuda`, ...).
- `main.rs` — CLI.

## 3. The observation model

```
persistent deterministic state (SampleObjects + world + event timeline)
        │  deterministic observation function (universe u1)
        ▼
bounded endpoint observation  ──► D0 (buffered, diagnostic)
                                ├─► D1 (GPU writes endpoint-mapped region)
                                ├─► D2 (peer/device-DMA)  [topology-gated]
                                └─► D3 (endpoint-native)  [FUTURE CONCEPTUAL]
```

Closure precedes observation. For sampled-origin content the *intrinsic
closure* `C_rho(Ua, H, R) = X_O` reconstructs the canonical SampleObject
domain first; sampler transformations `P_q = T_q(Ua, X_O, A, q)` happen only
after closure. Residuals are never summed with transforms under an unproven
commutation.

## 4. GPU design posture

- Heavy sampler state is **VRAM-resident**: SampleObjects, voice state,
  generator tables, residual indexes, filter/checkpoint state. Host-mapped
  memory is reserved for compact control traffic and **final endpoint
  observation writes** where it materially helps — never as the default home
  of ordinary state.
- The preferred kernel decomposition fuses object evaluation, rate/position,
  interpolation, envelope, gain/pan, mix accumulation, and endpoint packing
  into the bounded final observation: one CTA owns a frame tile, voices stream
  through shared memory in batches, threads accumulate exact i64 mixes, and
  the endpoint code is written once at the end. A conventional
  voice×frame→second-kernel design is explicitly labeled
  `GpuBufferedDiagnostic` and nothing more.
- Submission strategies (plain stream, high-priority stream, graphs,
  persistent kernel) are benchmarked, not canonized.

## 5. Evidence architecture

Every court writes one immutable JSON receipt (`vole.audio.evidence.v1`)
binding verdict, environment, hardware, method, counters, timing, provenance
hashes, and limitations, with a self-hash over canonical JSON. Receipts are
never rewritten. Negative and inconclusive verdicts are first-class.

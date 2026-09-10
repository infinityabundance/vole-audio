# EVIDENCE — `vole.audio.evidence.v1`

How this repository proves things, and how it refuses to fake them.

## Result vocabulary

Every result is exactly one of (see `src/status.rs`):

| Verdict                | Meaning                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| `SUPPORTED`            | path worked; correctness/deadline criteria satisfied               |
| `UNSUPPORTED_BY_API`   | the API surface does not expose the operation                      |
| `UNSUPPORTED_BY_HARDWARE` | the hardware/driver/firmware combination cannot do it           |
| `UNSUPPORTED_BY_TOPOLOGY` | the PCI/fabric topology forbids it                              |
| `FAILED_CORRECTNESS`   | ran but produced wrong results                                     |
| `FAILED_DEADLINE`      | ran correctly but missed the deadline                              |
| `FELL_BACK_TO_D0`      | D1/D2 attempt not possible; run legitimately used D0 — stated      |
| `INCONCLUSIVE`         | evidence insufficient; reason recorded                             |
| `NOT_APPLICABLE`       | class of operation does not apply                                  |
| `NOT_IMPLEMENTED`      | declared future surface (D3); never a real result                  |

Device kernels emit the same vocabulary as `u8` status words; the host maps
them back. Codes cannot drift because both sides compile `src/status.rs`.

## Forbidden substitutions (evidence level)

- D1 attempt → D0 playback reported as success.
- "GPU rendered sound" → "direct endpoint materialization" without the memory
  path proven.
- compact model parameters → "compression ratio" without counting residuals,
  dependencies, tables, dictionaries, checkpoints, indexes, metadata,
  integrity, decoder/universe dependencies, shared-resource amortization.
- "output sounded correct" → "exact conformance" without byte/hash equality.
- "mean latency is low" → "real time" without deadline and tail evidence.

## Receipt anatomy

A receipt is a versioned JSON document: schema `vole.audio.evidence.v1`,
version 1. It carries (as applicable) git identity/dirtiness, universe and
profile, backend, GPU artifact hash, CPU, GPU, audio device, PCI topology,
memory config, firmware/BIOS, NUMA, OS, kernel, GPU driver, CUDA/ROCm
version, audio driver, rustc, scheduler, IRQ affinity, directness class,
topology class, host PCM staging/copy bytes, GPU→host PCM bytes, endpoint
observation depth, endpoint memory provenance, fence/sync evidence, coherency
assumptions, hidden-staging investigation, corpus hash, sample rate,
channels, voice distribution, quantum, benchmark order, reference/backend
hash, CPU governor, GPU clocks/power, thermal/display/concurrent-load/cache
state, endpoint clock, test duration, observation/warmup/run counts,
deadline, raw samples, raw trace hash, tail estimator, confidence method,
xruns, energy method, result, limitations.

**Immutability.** Receipts are written with `create_new` semantics under
`receipts/<court>/`; an existing receipt is never rewritten after code
changes. Each receipt self-verifies (`receipt show <file>` recomputes the
self-hash).

**Canonicalization (what the self-hash covers).** `receipt_sha256` is the
SHA-256 of the compact JSON encoding of the `receipt` body *exactly as the
file carries it* (object key order preserved; `serde_json`'s
`preserve_order`). Verification therefore re-derives the hash purely from the
stored bytes, which is what makes the receipt hash stable under **additive**
schema growth: a receipt written before a new field existed still hashes its
own field set, whether the newer code omits absent fields or writes them as
`null`. (Hashing the typed struct instead would silently invalidate every
older receipt the moment a field is added — the failure mode this rule
exists to prevent.)

Float-valued fields are encoded exactly: the receipt parser uses
`serde_json`'s `float_roundtrip` feature so that a shortest-decimal `f64`
re-parses to the same bits it was written from. Without it, a receipt
containing a float such as `0.9898596181369713` parses 1 ULP low and cannot
reproduce its own self-hash. Both features are load-bearing for evidence
integrity, not optimizations.

**Source anchors.** Each receipt records the git commit SHA, the committed
source-tree hash (`git rev-parse HEAD^{tree}` — a content anchor that stays
meaningful even when the work tree is dirty), and a source dirty flag. The
dirty computation excludes `receipts/` by pathspec: writing evidence must
never, by itself, mark the very tree it attests as dirty. Uncommitted source
changes still do. New fields may be added over time (the schema is versioned
and additive); archived receipts keep re-verifying because the self-hash
follows the stored shape, not the current struct (see Canonicalization
below).

Each receipt also records the **seal subject** (`environment.seal_subject_hash`,
see `evidence::subject`): a SHA-256 over every tracked source file except the
evidence/governance trees (`receipts/`, `target/`, `scripts/out/`, `docs/`,
`.git/`). This is the identity a seal compares — committing receipts/docs
never invalidates a seal, and only a code change requires a new one. The git
commit/tree stay in the receipt as the exact historical provenance of the
battery tree (`vole-audio seal subject` prints the current subject).

## Measurement boundary (PCM/sample-domain exposure)

Counted as sample-domain exposure (independently addressable storage):

- `host_pcm_resident_peak` / `host_pcm_resident_integral` (byte·s)
- `host_pcm_staging_bytes`, `host_pcm_copy_bytes`, `gpu_to_host_pcm_bytes`
- `endpoint_observation_bytes`, `endpoint_depth_frames` (current/min/max)
- device sample block bytes (D0 diagnostic block)

Counted separately as transient compute state:

- `transient_compute_words` — register/shared-memory reduction accumulators.

Aliases of one physical shared region are not double-counted unless reporting
virtual exposure. Each receipt names its measurement boundary.

## Timing

- Host/end-to-end: `CLOCK_MONOTONIC_RAW`.
- GPU kernel: CUDA/HIP events — never substituted for end-to-end deadline.
- Raw samples are stored. Percentile reporting policy:
  `n >= ceil(20 / (1 - p))` (p50 ≥ 40, p90 ≥ 200, p99 ≥ 2 000,
  p99.9 ≥ 20 000). Below that: `max_observed` + the deepest defensible
  percentile + why deeper tails are unreported.

## Energy

Only real measurement sources: NVML, ROCm SMI/AMDSMI, hwmon/sysfs, external
meter. Method must record source, resolution, interval, idle subtraction,
uncertainty. Energy is **never** invented from TDP.

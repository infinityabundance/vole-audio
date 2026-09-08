# ENTROPY ACCOUNTING — complete-cost rules, counters, and methodology

> Owner of every "how many bytes / how fast / how resident" rule. The
> companion files `docs/RANS.md` and `docs/ENTROPY_NATIVE.md` own the codec
> and the architecture; this file owns the numbers.

## Complete representation cost (H.2.11)

For every candidate representation:

```
L_complete =
    hypothesis bytes            (procedural parameters/state)
  + parameters/state
  + entropy model bytes         (inline models, or attributed share of shared)
  + entropy payload bytes       (rANS/RAW bodies)
  + entropy page index bytes
  + checkpoints
  + dependencies
  + metadata
  + integrity
```

Never report only the rANS body as "the representation size". A 100-byte
payload backed by a 512-byte model is not a 100-byte representation. Reported
separately, always:

- raw sample bytes (canonical i32 LE codes),
- canonical U1 literal bytes (descriptor header + sample count + codes),
- source WAV bytes where applicable,
- native rANS literal bytes,
- model + rANS residual bytes,
- **complete** representation bytes.

## RAW fallback (H.2.5)

rANS is never mandatory. For every entropy block, choose RANS only when

```
complete_rans_bytes < complete_raw_bytes
```

where `complete_rans_bytes` includes block header, model bytes or model
reference, rANS state, encoded symbols, indexes, alignment/padding, and
integrity where applicable. Uniform / incompressible data converges toward
RAW or another stronger literal fallback — that convergence is a **success
condition**, and high-entropy negative controls (white noise,
random/encrypted-like data, structurally hostile signals) must be seen to
fall back honestly (H.2.31).

## Storage accounting: declared / unique / physical (H.2.24)

Always report three distinct quantities:

- **DECLARED** — complete bytes attributed to each logical audio object if
  it were stored standalone;
- **UNIQUE** — content-unique canonical payload bytes across the store
  (deduplication applied);
- **PHYSICAL** — actual backing bytes of the store engine, including its own
  metadata/framing.

A shared model is never reported as zero bytes. For shared dependencies,
report

- **standalone** cost (attributed in full to one object), and
- **marginal** cost given the dependency already resident,
separately. Model identity is content-addressable (its own canonical bytes
define its id).

## Receipt fields (H.2.35, additive to `vole.audio.evidence.v1`)

Entropy receipts include (added as optional extras; old receipts remain
parseable and self-verifying):

- representation kind; source sample bytes; canonical U1 bytes;
- hypothesis bytes; model bytes; entropy payload bytes; entropy index bytes;
  dependency bytes; integrity bytes; complete bytes;
- page size; page count; model-sharing mode; symbolization;
- compression ratio vs each baseline;
- exact reconstruction hash;
- symbols decoded; pages touched; random-access halo;
- scalar decode time; SIMD decode time; GPU decode time;
- sample-domain residency (per-surface peak + byte·time);
- DtoH sample bytes; host sample-copy bytes; endpoint observation bytes;
- search strategy; candidates evaluated; search work; DSFB diagnostics;
- EntropyFS declared/unique/physical bytes;
- environment, source commit, source-tree hash, artifact hashes.

No manually transcribed flagship numbers: Markdown tables are generated from
receipts (existing `receipt perf` pattern extended).

## Performance methodology (H.2.36)

"rANS is faster" is never claimed from encode throughput. Separate:

- encode speed; sequential decode speed; partial decode speed; seek latency;
  GPU decode speed; fused observation latency; endpoint deadline margin;
  storage ratio; CPU utilization; GPU utilization; energy.

Measurements use cold/warm/repeated runs and randomized benchmark order where
appropriate; raw samples are retained for Phase-M statistical treatment.
Report **crossover surfaces** (voices, page size, frames/quantum,
representation class, residual density, entropy density) rather than one
favorable speedup number as universal truth (H.2.37).

Expected possible (all valid, all to be reported as measured):

- CPU wins small entropy pages;
- AVX-512 wins low voice counts;
- GPU wins only with enough simultaneous voices/pages;
- fused D1 reduces traffic while being slower;
- rANS literal trails FLAC on natural music while procedural + rANS wins
  dramatically on generated/structured audio;
- model overhead makes tiny pages poor;
- shared models improve libraries but not standalone files;
- random audio falls back to literal storage.

The failure mode is not losing a benchmark; it is designing a court in which
the preferred architecture cannot lose (H.2.54).

## Memory-path honesty (H.2.38)

Distinguish entropy state, procedural state, residual symbols, transient
decoded page, global decoded waveform, D0 output block, host materialization
buffer, endpoint ring, verification buffer (see the surface table in
`docs/ENTROPY_NATIVE.md`). A transient decoded page is not a persistent
full-object waveform, but it is still sample-domain state and is counted
under the appropriate surface.

## Exposure counters (H.2.15)

Evidence counters extend additively; separate surfaces tracked:

- persistent sample-domain bytes;
- entropy state bytes; entropy model bytes; entropy payload bytes;
  entropy index bytes;
- host materialization sample peak + integral;
- host verification sample peak + integral;
- GPU global sample intermediate bytes; GPU transient sample words;
- GPU→host sample bytes; host sample-copy bytes;
- endpoint observation bytes.

Materialization state is never conflated with verification instrumentation.
Old receipts remain parseable; prefer additive optional fields over rewriting
evidence schema history.

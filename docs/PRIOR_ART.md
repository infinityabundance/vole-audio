# Prior art

VOLE-Audio does not claim its ingredients are individually new. This file
indexes the prior-art landscape acknowledged in the paper and in this
repository's evidence posture. Research materials (snapshots, notes, source
archives) live in the gitignored `research/` directory; this document is the
maintainable index for the implementation.

## Acknowledged foundations

- **Programmable sound synthesis**: MUSIC-N (Music V), Csound, SuperCollider —
  sound as a computed process; decades of established synthesis systems.
- **Structured audio**: MPEG-4 Structured Audio (SAOL/score/sample-bank
  model, transmitted synthesis programs, synthetic/natural hybrid).
- **Procedural audio**: "sound as process" literature and systems; parametric
  and generative audio in games and tools.
- **Predictive/residual coding**: predictive lossless codecs — model plus
  residual reconstruction (the general ancestor of residual-governed
  SampleObjects).
- **Object audio / spatial rendering**: scene- and listener-dependent
  rendering (MPEG-H, Dolby Atmos-class systems; prior art for object/scene
  observation views).
- **GPU audio and DSP**: GPU-accelerated synthesis, convolution, and signal
  processing research and products.
- **Direct memory and peer DMA**: GPUDirect-class mechanisms, DMA-BUF,
  peer-to-peer DMA; endpoint DMA regions (ALSA/HDA/USB); zero-copy audio
  server paths.
- **Lossless formats**: FLAC-class predictive+residual codecs (benchmark
  baselines B1/B4, not normative dependencies).
- **EntropyFS/DSFB**: separate research threads in the same author's program;
  the normative decoder/evaluator must not depend on them. They are baselines
  and comparators only, and only where the paper says so.

## What the paper positions as its disclosed research architecture

Not the ingredients, but the **systematic representation-boundary inversion**:
treating sample-domain audio as an observation API over persistent
deterministic state, coupling sampled-origin inverse proceduralization to
explicit residual closure and literal fallback, preserving that state through
sampler operations and transport, and pushing the observation boundary toward
the physical endpoint via GPU-resident direct materialization — with an
empirical program that keeps negative results first-class.

## Evidence posture for baselines

Conventional baselines (B0 literal PCM; B1 lossless codec; B2 PCM-resident
sampler; B3 disk-streaming sampler; B4 compressed-file decode + playback) are
built and measured inside this repository with the same receipt machinery as
the VOLE paths. A missing/unsupported D1/D2 row remains visible as
`UNSUPPORTED` / `INCONCLUSIVE` / `NOT_APPLICABLE` — rows are never deleted to
make a chart friendlier.

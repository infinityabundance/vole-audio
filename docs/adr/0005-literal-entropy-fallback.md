# ADR 0005 — Literal entropy fallback is mandatory and complete-cost based

- Status: accepted (Phase H.2)
- Owner: `docs/ENTROPY_ACCOUNTING.md`, `docs/RANS.md`

## Context

Entropy coding can always be made to look good by hiding costs (models,
indexes, headers) or by comparing against artificially weak baselines (raw
32-bit PCM against 16-bit sources). VOLE-Audio must be impossible to fool:
incompressible content must honestly fall back.

## Decision

For every entropy block/page the encoder must compare **complete costs** and
choose rANS only when it wins:

- `complete_rans_bytes` includes block header + model bytes or model
  reference + rANS state + encoded symbols + indexes + alignment/integrity;
- `complete_raw_bytes` is the literal canonical sample bytes (RAW page).

Additional rules:

- RAW is a first-class literal representation (`LiteralRaw`); uniform /
  incompressible data converges toward RAW or another stronger literal
  fallback — a success condition, never hidden.
- High-entropy negative controls (white noise, random/encrypted-like,
  structurally hostile content) must approach or lose to RAW / literal /
  conventional baselines in the courts; the corpus is frozen with these
  controls and no cherry-picking.
- Storage comparison is fair on source bit depth; raw i32 expansion is never
  used to make VOLE look good against 16-bit WAV content.
- Conventional baselines (FLAC via a pinned external executable when present,
  else `NOT_AVAILABLE`) are recorded with exact command/version/hash.

## Consequences

The literal entropy floor (`court entropy-literal`), the CUDA RAW-fallback
jobs (`court entropy-cuda`), and the D1 high-entropy control (`court
entropy-d1`) all exercise RAW pages; negative controls stay visible forever.

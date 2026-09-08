# ADR 0002 — EntropyFS is optional persistence, never required playback

- Status: accepted (Phase H.2)
- Owner: `docs/ENTROPYFS.md`

## Context

VOLE Video and EntropyFS established content-addressed, exactly-sharing,
GC-able physical persistence. VOLE-Audio needs the same capability for its
canonical entropy records without making playback depend on a mount, an
external engine, or a particular store.

## Decision

Persistence is an **optional adapter behind an in-crate abstraction**
(`ObjectStore`: `put`/`get`/`contains`/`sync`/physical metrics). Two
backends exist:

- `EmbeddedStore` — always available, content-addressed canonical payloads in
  one directory, used by courts and standalone materialization;
- `EntropyFsStore` — behind the default-off `entropyfs-store` feature, backed
  by the real published `entropyfs` 0.7.17 embeddable engine.

Rules:

- EntropyFS must never be required to materialize or play a VOLE-Audio
  object; `EmbeddedStore` provides identical canonical bytes.
- Store identity maps VOLE semantic/content identity to EntropyFS `BlobId`
  explicitly and re-verifies payload hashes on retrieval; identities are
  never silently replaced.
- Accounting always reports `declared` (standalone bytes per logical object),
  `unique` (content-unique canonical bytes), and `physical` (real backing
  bytes) separately; shared models are never reported as zero bytes
  (`docs/ENTROPY_ACCOUNTING.md`).
- Phase N's archive grammar is not bound to EntropyFS; canonical entropy
  records stay self-delimiting and versioned so Phase N can embed them.

## Consequences

CPU-only, no-feature builds run every court except the EntropyFS adapter
court, which reports `INCONCLUSIVE` plus a limitation when the feature is
off — never a fabricated result.

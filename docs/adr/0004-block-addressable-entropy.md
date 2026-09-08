# ADR 0004 — Block-addressable entropy and the random-access requirement

- Status: accepted (Phase H.2)
- Owner: `docs/ENTROPY_NATIVE.md` (architecture), `docs/RANS.md` (block/page layout)

## Context

A monolithic rANS stream over an object is incompatible with VOLE-Audio's
observation model: loops, reverse playback, non-unit rate, random seeks, and
bounded endpoint deadlines must never require decoding unrelated content, and
corruption must stay local. A single giant stream also serializes decode and
makes GPU/SIMD page-parallelism impossible.

## Decision

Entropy payloads are organized as **independently decodable pages**
(`vole.entropy.p1` pages; a page index maps frame ranges to pages):

- A page is self-describing: start frame, frame count, channel scope,
  symbolization id, model reference (inline or shared pool id), decoded
  symbol count, rANS state, encoded length/bytes, integrity.
- Observation of a region decodes only the pages intersecting it (plus a
  recorded, bounded halo where the sampler needs neighbors); the D1 court
  proves per-window decode of exactly the intersecting pages.
- Every page is independently bounded; decode complexity limits bound the
  worst case (maximum symbols/pages/models/dependency depth; hostile inputs
  fail typed, never with unbounded work).
- Page size is representation metadata (the Pareto court measures 64–4096
  frames); no universal optimum is assumed.

## Consequences

Reverse playback decodes the relevant pages into bounded transient state and
reads them backward — it never decodes the whole object. The CUDA decoder
parallelizes over pages (one thread per page); the entropy→D1 path decodes
only the pages of the current window.

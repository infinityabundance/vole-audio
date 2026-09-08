# ADR 0001 — Entropy coding is representation, never semantic authority

- Status: accepted (Phase H.2)
- Owner: `docs/ENTROPY_NATIVE.md` (representation contract), `docs/RANS.md` (codec freeze)

## Context

The normative object in VOLE-Audio is the `SampleObject` under the frozen
`vole.audio.u1` semantics. Once entropy-coded payloads exist, the tempting
shortcut is to treat "the rANS file" as an audio format with its own
semantics, or to let a convenient coder change sample semantics.

## Decision

Entropy coding is a **physical/canonical representation** of a `SampleObject`,
orthogonal to the hypothesis family (literal, procedural, procedural +
residual, referenced, …). Entropy payloads (`vole.entropy.p1`) never carry
semantic authority:

- decoding any representation must reproduce the canonical U1 sample codes
  (literal) or the exact Phase-E residual closure (residual-governed)
  byte-for-byte;
- U1 arithmetic, residual closure algebra, clocks/events, and observation
  semantics are never altered to make coding convenient;
- rANS probabilities are integer model tables (`MODEL_TOTAL = 16384`,
  `scale_bits = 14`, deterministic largest-remainder normalization); floating
  point never enters normative decode;
- the scalar decoder is the semantic reference; SIMD/GPU paths must prove
  byte equality in courts (they hold no authority of their own).

## Consequences

- Phase K's inverse compiler and Phase O's learned prediction must judge
  candidates by complete entropy-coded exact-residual cost (hypothesis +
  model + payload + index + dependencies), never by raw residual counts or
  MSE alone — the API is provided by H.2 (`docs/ENTROPY_ACCOUNTING.md`).
- "rANS" is never advertised as a generator; a coded payload always denotes
  the same semantic object its decoded samples denote.

# Corpus

This directory holds the canonical flagship corpus manifest (Phase M).

- `manifest.json` — the frozen, identity-bearing corpus manifest
  (`vole.audio.corpus.v1`).
- `generated/` — reserved for admitted **external** objects. It is empty by
  default: the flagship corpus objects are *generated*, never stored, so the
  manifest carries their generator specification and canonical content hash
  instead of their samples.

Current state: **FROZEN** — 115 objects, ~3.6 minutes of material at
44.1/48/96/192 kHz, stratified across source structure, amplitude occupancy,
channel structure, temporal structure and entropy character.

`source_structure_class` names the material an object was **designed as**
(frozen before any result exists). It is not the representation the inverse
compiler later selects; measured results record `selected_representation`
separately, and a source class of `oscillator` may legitimately compile to
`exact_repeat`.

## What is frozen

Membership and class assignments live in `src/corpus/specs.rs` (code, in the
seal subject). The manifest is written **once** from that membership:

```sh
vole-audio corpus freeze --out corpus/manifest.json
```

and is never tuned afterwards. A re-run refuses to overwrite an existing
`FROZEN` manifest unless a new corpus identity is declared deliberately:

```sh
vole-audio corpus freeze --out corpus/manifest.json \
    --amend-frozen "reason recorded in the seal ledger"
```

Everything that can change a result is part of the object identity:

```text
id ∥ class ∥ source structure ∥ amplitude ∥ channel structure ∥ temporal
   ∥ entropy ∥ sample_rate ∥ channels ∥ frames ∥ semantics ∥ repeat period
   ∥ generator kind ∥ generator parameters ∥ source ∥ conversion
   ∥ b1 comparable ∥ canonical i32 hash
```

and `corpus_sha256` covers every object's identity bytes **in order**.
`duration_ms` is derived and recomputed+verified rather than hashed, and
`expected_inclusion_surfaces` is derived policy.

## Verification

```sh
vole-audio corpus list      # the frozen population and its classes
vole-audio corpus verify    # regenerate everything and prove the manifest
```

`corpus verify` exits nonzero on **any** of: a manifest that does not parse or
carries the wrong schema, universe, profile or state; a `corpus_sha256` that
does not cover its objects; a duplicate object id; an object the frozen
membership defines but the manifest lacks (or the reverse); a manifest order
that differs from the frozen membership (benchmark order is experimental state);
a mutated class/rate/size/channel; any identity field (including `class` and
`conversion`) that differs from the canonical regenerated object; a generator
whose regenerated samples no longer match the frozen canonical hash; population
counts that disagree with the **derived** format-domain counts.

## Populations and the B1 format domain

FLAC encodes at most 8 channels; VOLE supports up to 32. That asymmetry is made
explicit rather than allowed to become a hidden denominator change:

```text
flagship/B1-comparable        1..=8 channels
flagship/high-channel stress  >8 channels: B1 = NOT_APPLICABLE_BY_FORMAT_DOMAIN
```

Objects in the second population can still be measured by every other surface,
but their bytes must never enter a B1-vs-VOLE aggregate. Eligibility is derived
from each object's channel count by the verifier and the courts — never read
from the manifest's `b1_comparable` audit field, which must merely agree with
the derivation.

## The real-recording stratum

The frozen population is deterministic generated material plus hostile
adversarial controls. A **license-clean real-recording stratum is declared and
currently vacant**: no production claim from this corpus rests on real
recordings yet, and `court corpus` records that as `real_audio_stratum:
VACANT_DECLARED` rather than staying silent.

Admission of an external object requires, before it may enter `generated/`:

```text
source
license
original-content hash
explicit ingest/conversion procedure (and the conversion path recorded)
canonical i32 hash
```

No mystery files. Courts report external entries as `NOT_AVAILABLE` rather than
inventing data.

## Rules

1. The corpus is frozen before the flagship performance court runs; it is never
   tuned after results exist, and membership is not changed because a result is
   unfavourable (contract §48).
2. Every object is bound by canonical content identity (SHA-256).
3. Regeneration must reproduce identical hashes on any host (integer-only
   generation: the frozen Q30 sine table and splitmix64).

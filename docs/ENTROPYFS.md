# ENTROPYFS — optional persistence (H.2.21–25)

> Owner of the optional persistence boundary. Companion:
> `docs/ENTROPY_ACCOUNTING.md` owns the declared/unique/physical rules;
> `docs/ENTROPY_NATIVE.md` owns the representation contract. Prior art:
> VOLE Video Phase P (`infinityabundance/vole` docs/phase-p.md) and the
> EntropyFS engine (`infinityabundance/entropyfs`), whose complete information
> accounting is reused rather than reinvented.

## Boundary

EntropyFS owns **optional** physical persistence, content-addressed storage,
exact object sharing, GC/reachability, cross-object deduplication, and
physical storage accounting. EntropyFS **must not** become required for
standalone VOLE-Audio materialization. VOLE-Audio stays materializable with
no EntropyFS present: the same canonical object through an embedded store
must equal the same canonical object fetched through the EntropyFS adapter.

Required invariant (H.2.25):

```
embedded canonical object
  == canonical object via EmbeddedStore
  == canonical object via EntropyFsStore
```

## Store abstraction

In-crate `ObjectStore`-level abstraction (`src/entropy/store.rs`) with
bounded operations:

- `put(canonical_bytes) -> id` (content-addressed; deduplicates),
- `get(id, max_bytes)`,
- `contains(id)`,
- `sync()`,
- physical/accounting metrics (declared/unique/physical bytes).

Two implementations:

- **EmbeddedStore** (always available): canonical payloads in a bounded local
  container, content-identity keyed.
- **EntropyFsStore** (Cargo feature `entropyfs`, default-off): adapter around
  the real published EntropyFS engine. Do **not** reimplement EntropyFS
  inside vole-audio.

The adapter keeps an explicit mapping

```
VOLE semantic/content identity <-> EntropyFS BlobId
```

and re-hashes/verifies canonical payloads on retrieval. If EntropyFS content
identity differs from VOLE-Audio's object identity, VOLE identity is never
silently replaced.

## Persisted object types (H.2.23)

Independently addressable canonical payloads:

- SampleObject descriptors;
- literal entropy pages;
- residual entropy pages;
- rANS models (content-addressable);
- shared symbolization tables;
- wavetable payloads;
- procedural parameter objects;
- checkpoints;
- shared dictionaries (when later admitted by evidence).

Cross-object sharing is never *claimed* until a court finds identical
canonical objects.

## Claim boundary (H.2.40)

A successful EntropyFS court permits stating:

- canonical VOLE-Audio entropy/procedural objects persist through EntropyFS
  and materialize identically;
- exact shared objects occupy unique physical storage once where the engine
  does so;
- declared, unique, and physical bytes are measured and reported separately.

It does not permit claiming: EntropyFS creates missing information; every
audio library deduplicates substantially; physical bytes equal logical
representation bytes; shared dependencies are free.

## Feature wiring

- Cargo feature `entropyfs` (default-off) pulls the real engine.
- Default CPU-only builds work without CUDA/ROCm/ALSA/EntropyFS/DSFB/FLAC
  (H.2.49).
- Phase N embeds canonical records without depending on EntropyFS; the
  archive format is not frozen early.

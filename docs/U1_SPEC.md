# U1_SPEC — vole.audio.u1

**Status: DRAFT — being frozen in Phase B in lockstep with `src/universe/`.**
This file will become the normative specification of the first semantic
universe. Until Phase B completes, nothing below is normative; the file
exists so the document is present from Phase A and the freeze lands in one
place.

## Scope of the freeze (Phase B will specify and the code will match)

- canonical sample code domain and integer-PCM ingest mapping;
- fixed-point conventions: positions/rates (Q24), gain/pan/envelope (Q16);
- oscillator phase (u64 mod 2^64), phase→table addressing, interpolation;
- overflow/rounding/saturation policy and the mixing bound proof
  (i64 accumulation, one final saturation, |mix| < 2^43 « 2^63);
- time model: logical media frames, epoch ids, object and endpoint frame
  coordinates, nominal sample rate, physical clock separation;
- event total order (timestamp, class priority, sequence);
- PRNG algorithm id/state/stream partitioning (frozen; no `thread_rng`);
- resampler table policy: measured then frozen, hashed, committed, treated as
  a universe dependency (generation is non-normative after freeze);
- interpolation rules (nearest/linear integer-exact), endpoint packing;
- canonical serialization and content identity (SHA-256);
- checkpoint encoding and residual rules.

## Reference vectors

Phase B freezes small objects and their expected SHA-256 observation hashes
here; `cargo test` enforces them.

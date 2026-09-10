# LEARNED OBJECT FORMAT — canonical bytes

Explicit little-endian binary, written and read by hand. serde/bincode layout,
Rust enum memory layout and any implementation-language object model are
**never** normative media representation.

```text
LearnedObject :=
    MAGIC          12   b"vole.learned"
    VERSION        u8   = 1
    PROFILE_LEN    u8
    PROFILE        bytes  b"vole.audio.u1/vole.audio.learned.exp1"
    CHANNELS       u8
    FRAMES         u64
    SAMPLE_RATE    u32
    MODEL_LEN      u64
    MODEL          bytes  (kind tag first)
    RESIDUAL_CODEC u8
    RESIDUAL_LEN   u64
    RESIDUAL       bytes  (payload only, no codec id)
    DEP_COUNT      u32
    DEP            DEP_COUNT × 32-byte content id
    DIGEST         32     SHA-256 over everything before it
```

Model bytes:

| tag | family | payload |
| --- | ------ | ------- |
| 0 | linear finite-field / block-local | kind, channels, taps(u16), block(u32), weights `K·C·C` × i16, bias `C` × i32 |
| 1 | nonlinear finite-field | kind, channels, taps(u16), layer count(u32), block(u32), then per layer: `in(u32) out(u32) weights(out×in × i16) bias(out × i32) activation` |
| 2 | stateful | kind, channels, state_dim(u16), checkpoint interval(u32), checkpoint count(u32), out_w, out_b, rec_w, rec_b, in_w, activation, checkpoints (frame u64 + state) |
| 3 | transfer operator | kind, channels, source_channels, taps(u16), delay(i64), weights `K·C·Cs` × i16, bias `C` × i32 |

Identity is `SHA-256` of the canonical bytes (including the trailing digest).
Decoding validates: magic, version, profile, bounded geometry, bounded model
length, known model kind, bounded residual length, bounded dependency count, the
trailing digest, full model validation and **residual decodability against the
declared geometry**. Trailing or missing bytes are rejected.

## Dependency closure

Every learned object exposes its complete transitive dependency closure: the
learned model, any shared model, activation tables, a source `SampleObject`, a
dictionary, latent blocks, checkpoints, the frozen lookup tables, the profile and
the residual codec. No dependency is free because it is resident in VRAM, shared
in process memory, installed with the application, or part of the trainer.

## Hostile-input bounds

Hard ceilings exist for graph nodes, graph depth, tensor rank and elements,
weight/bias/activation/state/latent/checkpoint bytes, receptive field, tap count,
dependencies, checkpoint count, operations per sample/block and total decode
operations, canonical object bytes and residual bytes. Cyclic graphs, mismatched
dimensions, integer-overflowing dimensions, malformed quantization, invalid
scales, impossible checkpoint references, unknown opcodes and malformed residual
streams are rejected.

## Hostile-input guarantees

There is no arbitrary native code, no arbitrary GPU code, and no embedded
generic scripting runtime. The only thing a learned object can do is evaluate a
bounded integer graph and read its bounded residual.

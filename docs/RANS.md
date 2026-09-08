# RANS — native deterministic rANS codec (normative freeze)

> Owner of the frozen rANS semantics. Codec core: `src/entropy/rans.rs`
> (primitive state machine, `no_std`-clean, scalar authority),
> `src/entropy/model.rs` (canonical normalization), `src/entropy/block.rs`
> (canonical self-describing blocks/pages, host). Container grammar for whole
> objects lives with `docs/ENTROPY_NATIVE.md`; complete-cost rules live with
> `docs/ENTROPY_ACCOUNTING.md`.

## Provenance

The core parameters and state machine are reused from the VOLE Video native
rANS floor (`infinityabundance/vole`), which was itself validated against an
independent byte-parity oracle (ryg-rans-rs) and hostile-input courts. Reuse
is deliberate: cross-VOLE semantic consistency is preferred where it does not
hurt the audio representation. The rANS *primitive* semantics here are
byte-compatible with ryg's `rans_byte.h` conventions and with
`ryg-rans-rs-core` 0.5.1 (independent oracle, **dev-dependency only** — never
linked into normative materialization).

Deliberate divergence (documented, none affects the primitive): the block/
page container bytes are audio-specific (`vole.entropy.p1`, see below) and
are **not** byte-compatible with VOLE Video's container; wherever model and
layout semantics match (symbol sequence, `(start, freq)` model values, byte
order), tests assert **byte parity** with the oracle; where audio layout
deliberately differs, tests assert independent symbol-sequence equivalence.

## Frozen parameters

| Name | Value | Meaning |
| --- | --- | --- |
| `RANS_SCALE_BITS` | `14` | log2 of the probability scale |
| `RANS_MODEL_TOTAL` | `16384` (`1 << 14`) | total normalized frequency per model |
| `RANS_STATE_L` | `2^23` (`1 << 23`) | lower bound of the normalized state interval |
| state width | 32 bits | `u32` state, checked arithmetic |
| byte order | little-endian for the flushed state; renorm bytes LSB-first written backward | byte-compatible with ryg |
| encoder renorm | `x_max = ((RANS_STATE_L >> scale_bits) << 8) * freq`; emit `x & 0xff`, `x >>= 8` while `x >= x_max` | division-based reference path |
| encoder transition | `x = ((x / freq) << scale_bits) + (x % freq) + start` | exact division; no reciprocal approximation in the normative path |
| decoder transition | `x = freq * (x >> scale_bits) + (slot - start)` | where `slot = x & (RANS_MODEL_TOTAL - 1)` |
| decoder renorm | while `x < RANS_STATE_L`: `x = (x << 8) | byte()` | forward read of the stream |
| state flush | `u32` little-endian at the front of the stream | 4 bytes |
| symbol alphabet | ≤ 256 present symbols, canonical order = ascending symbol value | models over `u8`-valued symbols |

Freezing policy: changing any parameter above is a representation/profile
change requiring a new container version id and fresh reference vectors —
never a silent edit. The four rANS parameters above are additionally pinned
by cross-VOLE references; a future audio-only profile with different
scale/width must be a **separate representation**, never a mutation of this
one.

## State machine (normative)

Encode of a symbol sequence `s_0..s_{n-1}` with model `M`:

1. `x = RANS_STATE_L`.
2. For each symbol in order: let `(start, freq)` be the model interval of the
   symbol. Renormalize (emit bytes while `x >= ((RANS_STATE_L >>
   scale_bits) << 8) * freq`), then
   `x = ((x / freq) << scale_bits) + (x % freq) + start`.
   Invariant after every transition: `x in [RANS_STATE_L, 2^31 + 2^14)`.
3. Flush `x` as 4 bytes little-endian at the front of the output.

Decode of `n` symbols (the count is carried by the container):

1. `x = read_u32_le()`. (`x >= RANS_STATE_L` always holds for valid streams.)
2. For `i` in `(0..n).rev()`: `slot = x & (RANS_MODEL_TOTAL - 1)`;
   symbol = the model symbol whose interval `[start, start + freq)` contains
   `slot`; `x = freq * (x >> scale_bits) + (slot - start)`; then renormalize
   (read bytes while `x < RANS_STATE_L`). Output symbol for position `i`.
3. A valid stream ends with `x == RANS_STATE_L` after the last transition and
   with the byte cursor exactly at the end of the encoded payload.

Decode order is the reverse of encode order; the decoder emits symbols
back-to-front. Malformed conditions (byte exhaustion during renorm, model
interval arithmetic that cannot satisfy the invariant, wrong symbol count)
are **typed errors** — never panic, never UB, never silent wrong output.

Worst-case encoded size: ≤ 4 bytes renorm per symbol + 4 state bytes. The
host encoder allocates `4 * n + 8` and the container records the exact
encoded length.

## Canonical model normalization (frozen; H.2.3)

Input: raw counts `c[0..A)` over the `A` **present** symbols (each `>= 1`),
ordered ascending by symbol value; `1 <= A <= 256`; `S = sum(c)` (`u64`).
Output: integer frequencies `f[0..A)`, `sum(f) == RANS_MODEL_TOTAL`,
`f[i] >= 1` for every present symbol.

1. `q[i] = floor(c[i] * T / S)` computed in `u128` (`T = RANS_MODEL_TOTAL`).
2. `rem[i] = c[i] * T - q[i] * S` (`0 <= rem[i] < S`).
3. `extra = T - sum(q)` (nonnegative; `sum(q) <= T` always).
4. Assign one unit to each of the `extra` symbols with the **largest**
   `rem[i]`; ties broken by **ascending symbol index** (smallest index
   first).
5. **Guaranteed-minimum pass**: while a present symbol has `f[i] == 0`,
   set `f[i] = 1` for the smallest-index zero-frequency symbol and subtract 1
   from the largest-index symbol with `f[j] >= 2`. Each iteration strictly
   reduces the number of zeros; the pass terminates in at most `A`
   iterations. If it cannot (impossible by construction when `A <= T`), model
   construction fails and the caller falls back to RAW.

The zero-frequency rule is: **absent symbols have frequency 0 and never
appear in any encoded stream of that model; present symbols always receive
`>= 1`**. Model bytes are stored information and are always counted (see
`docs/ENTROPY_ACCOUNTING.md`). No floating point appears anywhere in
frequency construction, normalization, or coding.

## Canonical block container (`vole.entropy.p1`)

A self-describing coded block. Versioned prefix bytes are part of every
block's complete byte cost.

```
vole.entropy.p1           12-byte profile/format tag (ASCII)
version u8                1 byte (1)
payload_kind u8           0 = rANS, 1 = RAW
symbolization_id u8       versioned reversible symbolization (see ENTROPY_NATIVE)
channel_scope u8          1..=MAX_CHANNELS
model_mode u8             0 = inline model, 1 = shared model reference
model_payload             inline: model bytes; shared: content-id prefix (u8 len + bytes)
symbol_count u64 LE       number of coded symbols (0 for RAW)
encoded_len u32 LE        byte length of the payload that follows
payload                   rANS stream or RAW bytes (exact length above)
integrity u8 + 32 bytes   0 = none, 1 = SHA-256 of everything before the tag
```

RAW payload kind carries the raw symbol bytes (or raw sample bytes for
symbolization `identity`) with `symbol_count` decoded implicitly from the
container extent. Every page/block must be decodable independently (see
H.2.6 page rules in `docs/ENTROPY_NATIVE.md`).

## Malformed-stream behavior

Every malformed input produces a typed error:

- truncated models, truncated state, truncated renorm bytes;
- frequencies that do not sum to `RANS_MODEL_TOTAL`, or `f[i] == 0` for a
  present symbol;
- impossible output lengths, page-count/model-count/symbol-count bombs,
  overlapping offsets, integer overflow, out-of-range page indexes,
  missing model references, recursive dependency cycles, corrupt content
  ids, absurd declared decode work.

Never: panic, undefined behavior, unbounded allocation, unbounded CPU, or
silent incorrect sample output. Enforced by unit tests, the hostile-input
court, and fuzz targets (H.2.33/H.2.34/H.2.47).

## Independent oracle

`ryg-rans-rs` (0.5.1) is a **dev-dependency only** oracle. Tests:

- byte parity: our encode == oracle encode for identical symbol/model
  semantics; our encode -> oracle decode; oracle encode -> our decode;
- independent symbol-sequence equivalence where audio layout deliberately
  differs from the oracle's container assumptions.

The normative runtime never depends on the oracle.

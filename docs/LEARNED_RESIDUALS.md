# LEARNED RESIDUALS — exact codec family

A learned predictor may leave *small nonzero* residuals almost everywhere rather
than *sparse large exceptions*. Judging learned predictors with a sparse-only
residual form would bias the experiment against them, so Phase O establishes a
deterministic, canonical, exact residual codec family **before** any learned
predictor is compared, and judges every predictor against the best member.

All six codecs are exact inverses on the canonical dense residual domain
(`Vec<i32>`, frame-major, channel-minor, `0` = no residual). Every decoder is
length-checked, allocation-bounded and hostile-safe.

| id | codec | form | notes |
| -- | ----- | ---- | ----- |
| 0 | `DenseI32` | fixed 4 bytes per value | length-binding by exact size |
| 1 | `SparseDelta` | `u64` total, `u64` count, then varint (index-delta, zigzag value) | index 0 representable; strictly ascending |
| 2 | `ZigZagVarint` | zigzag varint per value | self-delimiting, no trailing bytes |
| 3 | `BlockRice` | signed-mapped values, 32-value blocks, per-block `k` | `k` chosen by minimum bits with a decodability floor |
| 4 | `PredictiveRice` | first differences, then `BlockRice` | exact `i64` reconstruction checked against `i32` |
| 5 | `LiteralResidual` | `u64` total, `u64` count, then fixed `(index u64, value i32)` records | no varints |

Selection is `min_j (codec_id_byte + payload_bytes)` with ascending-id tie break,
so the selected codec is a pure function of the residual.

## Cost

```text
residual_complete_bytes = 1 (codec id) + payload.len()
```

The codec is part of representation accounting: it is never chosen to make a
preferred predictor win, and the same family prices every baseline.

## Decoding safety

* lengths are converted with `try_from`, never cast;
* declared counts are bounded by the declared value count;
* a huge sparse count is rejected before allocating;
* Rice unary runs are bounded (`MAX_RICE_UNARY`), and the encoder raises `k`
  whenever a block would exceed it, so every encoded stream is decodable;
* `i64` overflow in predictive reconstruction is impossible by construction and
  checked anyway;
* trailing bytes are rejected where the codec is self-delimiting.

## Residual shape (O.11)

`ResidualShape` records zero fraction, nonzero density, mean nonzero magnitude,
maximum magnitude, longest zero run and a first-order entropy estimate over the
first varint byte, plus a cheap `estimated_bits` proxy. These steer training only
as *proposals*: `learned-residual` records the shape and the selected codec, and
the final acceptance decision uses the real encoded bytes.

Two residual shapes are deliberately distinguished in the courts:

```text
tiny nonzero almost everywhere   -> dense codecs win
exact zero almost everywhere     -> sparse codecs win
```

and neither is privileged.

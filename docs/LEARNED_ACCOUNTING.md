# LEARNED ACCOUNTING — complete representation cost

The fundamental quantity is the **complete representation cost**, never the
prediction error. For a learned object:

```text
complete_bytes = metadata
               + model
               + residual
               + dependencies
               + integrity
```

where the model decomposes into:

```text
model_bytes = canonical_weight_bytes
            + bias_bytes
            + scale_bytes
            + activation_table_bytes
            + graph_metadata_bytes
            + tensor_dimension_bytes
            + state_definition_bytes
            + checkpoint_definition_bytes
```

Every component is computed from the canonical serialization, and the
decomposition is required to sum **exactly**:

```text
metadata + model + residual + dependency + integrity == canonical_bytes().len()
model sub-components                                  == model_bytes
```

`learned-accounting` in `court learned-capacity` asserts both; an accounted
number that is not the stored number is a bug, not a rounding difference.

## Raw vs canonical weight bytes

`raw_weight_bytes` and `canonical_weight_bytes` are reported separately. The
canonical container stores the quantized weights; no outer generic compressor is
applied. If a generic compressor is ever used as a baseline, its contribution
must remain visible and must not be hidden inside the model bytes.

## Explicitly not in the sum

| field | meaning |
| ----- | ------- |
| `raw_sample_bytes` | `4 · frames · channels` of the unencoded intrinsic |
| `canonical_literal_bytes` | the canonical U1 literal byte count for the same window |
| `persistent_sample_domain_bytes` | 0 for a learned representation |
| `persistent_state_bytes` | persistent learned state (stateful family) |
| `decoded_window_state_bytes` | window materialized to evaluate one range |
| `ops_per_sample` | abstract operations per output sample |
| `worst_case_replay_frames` | frames replayed to serve a mid-extent seek |

## Transfer regimes (O.5)

If target `X` depends on source `S`:

```text
L_standalone(X via S) = L(S) + L(Theta) + L(R) + L(metadata) + L(integrity) + L(index)
L_marginal(X | S)     =      L(Theta) + L(R) + L(metadata) + L(integrity) + incremental index
```

Both are reported where meaningful, and **no ratio is ever formed across the two
regimes**. A learned marginal byte count is never compared against a
conventional standalone byte count without labelling the asymmetry.

## Shared models (O.15)

```text
shared_model_bytes               counted once
per_object_incremental_bytes     residual + framing + dependencies + integrity
whole_corpus_bytes(n)            shared_model_bytes + n · per_object_incremental_bytes
amortization_crossover_objects   N* = smallest n with corpus_bytes(n) < n · independent_each
                                 (or NO_CROSSOVER_OBSERVED)
```

An amortized win is never reported without the corpus size required to obtain
it.

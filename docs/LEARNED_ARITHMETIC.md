# LEARNED ARITHMETIC — canonical integer semantics

Every canonical learned evaluation is integer/fixed-point. There is no
floating-point value anywhere in a canonical learned object; training may use
floating point freely, but only the compiled integer hypothesis has semantic
authority.

```text
WEIGHT   i16, Q12 fixed point   (|w| <= 32767, i.e. |w_real| < 8.0)
BIAS     i32, Q12 fixed point
SAMPLE   i32, canonical code domain
ACC      i64, exact integer accumulator
OUTPUT   sat_i32(round_shift_half_away(acc, 12))   (weights)
         sat_i32(round_shift_half_away(acc, 12))   (biases share the Q12 scale)
```

## Accumulator bound

With `K <= MAX_LEARNED_TAPS = 4096` taps, `|w| <= 2^15` and `|x| <= 2^31`, each
product is `<= 2^46`, so the sum is `<= 2^58`; adding an `i32` bias stays far
inside `i64::MAX`. A reduction therefore cannot overflow for any legal learned
model, **independent of accumulation order** — integer addition is associative
and exact, so scalar and SIMD reductions agree bit for bit. This is why the
SIMD path is exact by construction and not by tolerance.

## Rounding

`round_shift_half_away(v, s)` is round-half-away-from-zero in exact integer
arithmetic, computed against a floor quotient so `i64::MIN` cannot overflow:

```text
q = v >> s            (arithmetic shift == floor division)
r = v - (q << s)      (remainder in [0, 2^s))
v >= 0 : q + (r >= half)
v <  0 : q + (r >  half)
```

Verified by exhaustive boundary tests (`-0.5 -> -1`, `+0.5 -> 1`, `±0.499…-> 0`).

## Activations

Integer-only, deterministic, bounded, identical on every backend:

| tag | activation |
| --- | ---------- |
| 0 | identity |
| 1 | clamp (`lo_q12`, `hi_q12`) |
| 2 | saturating linear (`limit_q12`) |
| 3 | piecewise-linear (strictly ascending breakpoints, exact interpolation) |
| 4 | bounded lookup table (`shift`, table) |
| 5 | bounded integer polynomial (frozen Q12 coefficients) |

No transcendental function, no `f32`/`f64` in the normative path. Piecewise
interpolation uses `i128` intermediate products so no `i32` or `i64` overflow is
possible, with round-half-away-from-zero.

## Quantization compiler

Training output is compiled into the canonical form by one place
(`learned::quantize`), which rounds half away from zero and **saturates** rather
than wrapping. The float model is never the archived semantic object.

## Decode complexity

Per-sample and per-extent ceilings (`MAX_LEARNED_OPS_PER_SAMPLE`,
`MAX_LEARNED_OPS_PER_BLOCK`, `MAX_LEARNED_DECODE_OPS`) are enforced at object
validation, so a legal one-second object cannot require an absurd operation
count.

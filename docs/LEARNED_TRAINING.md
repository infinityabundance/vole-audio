# LEARNED TRAINING — non-normative fitting

## Training is disposable

The canonical object format never depends on a trainer, optimizer, floating-point
library, automatic-differentiation engine, random training order, GPU vendor or
framework. Training may use floating point, CUDA, ROCm, SIMD, external numerical
tools, heuristics, search, differentiable proxies and random initialization —
provided its output is compiled into a canonical deterministic learned
representation before acceptance. Only the compiled object participates in
closure and selection.

## No generic autodiff (O.20)

No general-purpose automatic differentiation framework is implemented. Fitting
uses specialized methods only:

* **ridge / ordinary least squares** over the causal tap window, solved through
  the normal equations with partial-pivot Gaussian elimination (with a
  trace-scaled jitter so degenerate feature columns remain solvable);
* **coordinate descent** over the canonical quantized parameters, greedily
  trying ±1 per coordinate in canonical order;
* **specialized nonlinear fitting**: a linear pre-activation from ridge, a
  piecewise-linear activation fitted by binning the pre-activation against the
  target, then a bounded coordinate-descent refinement.

## Ridge/least-squares fit

The design matrix encodes exactly the history the canonical evaluator will see
(zero before frame 0, reset at block boundaries for the block-local class).
Ridge regularization never applies to the bias. The returned predictor is always
the **quantized canonical** form, and closure is verified against the quantized
evaluator, never against the float model.

## Quantization-aware training (O.23)

Post-training quantization can destroy a family that floated well. The
quantization-aware loop therefore optimizes the behaviour of the **compiled
integer** object: coordinate descent minimizes the object's actual complete
encoded bytes, and the base fit is retained whenever the refinement does not
strictly improve it. The final judge is always:

```text
actual canonical model bytes + actual canonical exact residual bytes
```

never a floating-point training loss.

## Training cost (O.41, O.58)

Training cost is separate from playback cost but never hidden. The
`learned-training-cost` court records wall time, CPU time, candidate counts,
iterations, quantization attempts, peak host RSS (best effort), GPU time and
peak VRAM (zero when training is CPU-only), and the declared search budget:
maximum candidates, maximum taps, maximum hidden units, maximum iterations,
maximum model bytes and ridge lambda. An extremely expensive offline fit may
still be useful in archival contexts, but it is never described as free.

## Generalization is not required (O.16)

Three research modes are distinguished: object-specific fitting, closed-corpus
shared fitting, and out-of-corpus generalization. Only the first two are
required for a useful storage representation. A model trained for one object is
already a valid representation win for that object if its complete cost beats
the alternatives; a model shared across a closed corpus is useful even with zero
usefulness outside it.

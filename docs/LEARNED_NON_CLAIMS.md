# LEARNED NON-CLAIMS

Phase O explicitly rejects the following claims:

* learned representation is always smaller;
* learned representation is always faster;
* learned representation is always better than conventional codecs;
* learned representation eliminates literal storage;
* learned representation makes residuals negligible;
* learned representation improves perceptual quality;
* learned representation recovers the true source process;
* learned representation universally generalizes;
* neural networks are necessary for VOLE-Audio;
* generalization is required for object-specific compression;
* GPU always improves learned evaluation;
* tensor cores are normative;
* learned prediction changes VOLE's exactness model;
* Phase O changes `u1/v1`.

The literal fallback remains universal. Existing non-learned representations
remain first-class. A learned candidate that loses is a preserved result, not a
hidden one.

## Additional explicit limitations of this build

* The learned device kernel is not part of the frozen device artifact, so
  CUDA/ROCm learned execution is `NOT_IMPLEMENTED` / `UNSUPPORTED_BY_HARDWARE`;
  only the host scalar and SIMD surfaces are measured, with exact parity.
* The canonical container stores i16 Q12 weights; i8 and mixed precision are
  `NOT_IMPLEMENTED` (they need a weight-bits/scale format extension).
* The nonlinear, stateful and transfer families are mono-only in this build
  (transfer operators support cross-channel source/target counts, but the
  stateful realization is mono).
* Transfer nonlinear operators are not implemented; the implemented transfer
  family is a bounded causal cross-channel FIR.
* Cold/warm labels in `learned-random-access` mean first vs repeated in-process
  access, not verified host cache state.
* The learned corpora are deterministic synthetic material; the real-recording
  stratum remains `VACANT_DECLARED` as in Phase M.
* No perceptual, latency, energy or endpoint claim is made for a learned
  representation before a learned device artifact exists.

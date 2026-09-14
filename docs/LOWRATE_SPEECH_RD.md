# Low-rate speech coding: rate–distortion and mechanism design

Reference note for a 16 kHz CELP-family codec at 0.5–1.0 bit/sample of residual.
Dense, formula-first. Every claim is tagged:

| tag | meaning |
| --- | --- |
| ✅ | **Verified this session** from the fetched URL listed in §7 (quote/derivation reproducible from that page). |
| 📐 | **Derived or computed here** (arithmetic shown; no external claim). |
| 📖 | **Standard textbook result**, source named but *not fetched here* — verify against the named source before relying on the 3rd digit. |
| 🔶 | **Inference / engineering range** — consistent with ✅ facts but not itself a fetched number. |

Notation: `A(z) = 1 − Σ_{k=1}^{p} a_k z^{−k}` (AR analysis), `H(z) = 1/A(z)` (synthesis),
`W(z)` perceptual weighting, `Q(·)` scalar quantizer, `q_err[n] = Q(u[n]) − u[n]`,
`σ_x²` signal variance, `R` bits/sample, `D` MSE.

---

## 1. VQ vs. scalar quantization at 0.5–1.0 bit/sample

### 1.1 High-rate limits (the only place clean closed forms exist)

Rate–distortion function, memoryless Gaussian, MSE distortion ✅
(https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory):

$$ R(D) = \tfrac12\log_2(\sigma_x^2/D) \quad\Longleftrightarrow\quad D = \sigma_x^2\,2^{-2R}. $$

High-resolution quantizer distortion (Zador–Gersho form) 📖 (Gersho & Gray 1992, Thm. 5.2/5.3;
Wikipedia "Quantization (signal processing)" states the 6 dB/bit corollary ✅):

$$ D \;\approx\; G(\Lambda)\,M_{L}(f)\,2^{-2R}, \qquad M_L(f)=\Big(\textstyle\int f^{L/(L+2)}\Big)^{(L+2)/L}, $$

where `G(Λ)` is the **normalized second moment** of the L-dimensional cell (the
space-filling term) and `M_L(f)` is the source-dependent density term. For a unit-variance
uniform source the `G` ratios below are exact SNR gains.

### 1.2 Space-filling (granular) gain — normalized second moments

`G` of the 1-D uniform quantizer sets the reference: `G₁ = 1/12 = 0.08333`. `G∞ = 1/(2πe) = 0.05855`.
Gain over scalar `= 10·log₁₀(G₁/G_L)` dB. Values below are 📐 computed from the standard
tabulated `G` values 📖 (Conway & Sloane, *SPLAG*, Table 2.3; Gersho & Gray 1992 Table 5.3):

| L | lattice | G(Λ) | gain over 1-D uniform | dB |
| --- | --- | --- | --- | --- |
| 1 | Z | 1/12 = 0.08333 | 1.000 | 0.000 |
| 2 | A₂ (hexagonal) | 5/(36√3) = 0.08019 | 1.0392 | **0.167** |
| 3 | A₃\*/D₃\* (BCC) | 0.07854 | 1.0610 | **0.257** |
| 4 | D₄ | 0.07660 | 1.0879 | **0.366** |
| 8 | E₈ | 0.07168 | 1.1625 | **0.654** |
| 16 | Λ₁₆ (Barnes–Wall) | ≈0.0693 | 1.2018 | **0.799** |
| 24 | Λ₂₄ (Leech) | 0.06577 | 1.2669 | **1.027** |
| ∞ | optimal (Zador) | 1/(2πe) = 0.05855 | 1.4233 | **1.533** |

Sanity identities 📐: `πe/6 = 1.4233 → 1.5329 dB`, and `G₁/G∞ = 2πe/12 = πe/6` — the
granular limit and the shaping limit are *the same number*, which is the root of the
double-counting error discussed next.

### 1.3 Shaping (codebook-shaping) gain

For a fixed-rate, *entropy-coded* quantizer the optimal point density is
`∝ f(x)^{L/(L+2)}`, not uniform. The maximum improvement from matching the density (rather
than the cell shape) to a Gaussian is the **shaping gap = 1.53 dB** ✅
(https://en.wikipedia.org/wiki/Shaping_codes: "This 1.53 dB difference is known as the
shaping gap", citing Forney, *Trellis shaping*, IEEE T-IT 1992). Practical shaping
(Shell mapping) realizes **≈0.8 dB** ✅ (same page).

### 1.4 The accounting trap (this is the part most codec papers get wrong)

For a **memoryless Gaussian**, entropy-coded scalar quantization at high rate has ✅+📐

$$ D_{\text{ECSQ}} = \frac{\pi e}{6}\,\sigma_x^2\,2^{-2R} \;\Rightarrow\; 10\log_{10}\!\frac{\pi e}{6} = 1.53\ \text{dB above } R(D). $$

Optimal high-dimensional VQ + entropy coding attains `R(D)` (gap → 0). Therefore:

> **The entire VQ-over-scalar advantage for a memoryless Gaussian source is at most 1.53 dB
> at high rate.** Granular gain and shaping gain are *not additive beyond this*; they are two
> views of the same `πe/6` gap, whose decomposition depends on the baseline you choose.
> Statements of the form "space-filling 1.53 dB *plus* shaping 1.53 dB = 3 dB" double-count.
> 🔶 (This is my accounting; the two ✅ 1.53 dB facts are independent and both real.)

### 1.5 Memory (statistical) gain

The genuine large axis is **memory**, and it is available to scalar DPCM and VQ alike — it is
not a VQ-over-SQ gain. For a predictive-residual source it is the prediction gain
`G_p = σ_x²/σ_d²`. Measured in this tree over 600×320-sample frames ✅
(`docs/VOICE_RD.md` §2.1):

```
short-term LPC (order ≤ 16)     : 18.31 dB
+ long-term integer pitch       : 18.97 dB
```

So the residual the quantizer actually sees is already 18–19 dB below the signal. The
question "VQ vs SQ" applies **to that residual**, at 0.5–1.0 bit/sample, where the `πe/6`
budget is the high-rate bound and the practical edge is larger because the scalar side is
near overload (below).

### 1.6 The defensible numbers at 0.5–1.0 bit/sample

| regime | VQ advantage over scalar | tag |
| --- | --- | --- |
| memoryless Gaussian, high rate, entropy-coded both sides | **≤ 1.53 dB** (attained only as L→∞) | 📖📐 |
| same, practical L = 4–16, entropy-coded | **≈ 0.4–0.7 dB** (D₄…E₈ granular) + shaping | 📐 |
| fixed-rate scalar (no entropy coding), fine regime | grows: adds scalar's *granular loss* (~1.53 dB) on top | 📐 |
| **low rate (0.5–1 bit/sample), scalar in/near overload, no entropy coding** | **≈ 3–6 dB** | 🔶 |
| speech codec practice (CELP excitation VQ vs scalar residual) | **≈ 3–6 dB** | 🔶 |

The **3–6 dB** figure is a *low-rate, entropy-less, overload-inclusive* comparison, not the
high-rate granular gain. Makhoul, Roucos & Gish, *Vector quantization in speech coding*,
Proc. IEEE 73(11), 1985 📖 is the canonical speech-specific reference that establishes the
multi-dB practical edge; the open-access review by O'Shaughnessy ✅
(https://link.springer.com/article/10.1186/s13636-023-00274-x) states the mechanism without a
single dB number: "Block encoding is more efficient than memoryless … coding … sending
individual data samples independently ignores correlations" and "L is often 1024, allowing
transmission of a single 10-bit code rather than N LPC coefficients, each needing more than
5 bits." Treat 3–6 dB as engineering convention; the theorem-backed part is the ≤1.53 dB
high-rate envelope **plus** whatever overload relief the VQ/gain adaptation buys at low rate.

---

## 2. Closed-loop scalar prediction: the error recursion, overload, and the cliff

### 2.1 Two topologies — and the recursion only belongs to one of them

**(a) Reconstruction-fed DPCM** (predictor input = reconstruction at both ends). Encoder target
`t[n] = x[n] − Σ a_k x̂[n−k]`; quantize `u[n] = Q(t[n]) = t[n] + q_err[n]`; reconstruct
`x̂[n] = Σ a_k x̂[n−k] + u[n]`. Substituting:

$$ \hat{x}[n] = \sum_k a_k\hat{x}[n{-}k] + x[n] - \sum_k a_k\hat{x}[n{-}k] + q_{err}[n] = x[n] + q_{err}[n] $$

$$ \boxed{\,e[n] := \hat{x}[n]-x[n] = q_{err}[n]\,} $$

✅+📐 The output error is the **quantizer error, white, with no accumulation**. This is the
classical DPCM result and it is the topology of this tree's scalar path: `close_loop()`
computes `st = Σ w_j·probe.at(j)` from the *reconstruction* history and pushes the
reconstruction back (`src/voice/predict.rs` L630–664, verified in-tree). The error-feedback
in (a) lives in the *quantizer input*, not the output:

$$ t[n] = d[n] - \sum_k a_k\,q_{err}[n-k],\qquad d[n]=x[n]-\sum_k a_k x[n-k] \quad(\text{exact}) $$

📐 i.e. the quantizer sees the residual **minus a filtered copy of past quantization error**.
Its dynamic range is inflated, which is the low-rate overload mechanism in §2.3.

**(b) Open-loop analysis, closed-loop synthesis** (residual formed from the *original*; decoder
synthesizes with `1/A(z)`). Now

$$ X(z)=\frac{D(z)}{A(z)},\qquad \hat X(z)=\frac{\hat U(z)}{A(z)}
\;\Longrightarrow\;
E(z)=\hat X(z)-X(z)=\frac{\hat U(z)-D(z)}{A(z)}=\frac{Q_{err}(z)}{A(z)} $$

$$ \boxed{\,e[n]=\sum_{k=1}^{p} a_k\,e[n-k] + q_{err}[n]\,}\qquad(A(z)E(z)=Q_{err}(z)) $$

✅+📐 This is **exactly the recursion in the question** (`w_j = a_j`). It is the transfer
function of any codec that quantizes an open-loop residual and synthesizes through `1/A(z)`
(RELP-class), and it is *also* the exact output-error transfer function of a noise-shaping
quantizer designed with `NTF(z)=1/A(z)` (§3). Error power is amplified by the synthesis
filter's prediction gain:

$$ \mathrm{Var}(e)=\sigma_q^2\cdot\frac{1}{2\pi}\!\int_{-\pi}^{\pi}\!\frac{d\omega}{|A(e^{j\omega})|^2}
=\sigma_q^2\,G_p . $$

**Stability.** With `p` fixed and `A(z)` minimum-phase (all poles inside the unit circle — the
Levinson–Durbin step-up guarantees `|k_j|<1`, enforced here in `quantise_k`/`clamp_k_codes` ✅
`src/voice/predict.rs`), the linear recursion is BIBO-stable. The failure at low rate is **not**
linear instability; it is the *quantizer nonlinearity* (saturation) breaking the small-signal
analysis.

**Note on the tree's own narrative.** `docs/VOICE_RD.md` §3 says the scalar path's error is
"coloured by `1/A(z)`" and "fed back through the synthesis filter". For the *actual*
reconstruction-fed loop (a) above this is **not literally true**: `e[n]=q_err[n]` is white.
The coloring argument applies to topology (b), or to (a)'s *quantizer input*. The measured
cliff is real; the stated mechanism is imprecise. 🔶 (correction, not a refutation of the
measurement).

### 2.2 Granular vs. overload — why the curve bends

✅ (https://en.wikipedia.org/wiki/Quantization_(signal_processing)): distortion splits into
**granular** (`Δ²/12` in the fine regime) and **overload** (clipping when |input| exceeds the
supported range), and "it is common for the design … to involve determining the proper balance
between granular distortion and overload distortion. For a given supported number of possible
output values, reducing the average granular distortion may involve increasing the average
overload distortion, and vice versa." The `6.02 dB/bit` slope is only asymptotically valid:
"this derivation is only for a uniform quantizer applied to a uniform source."

**Slope overload** is the 1-bit case of the same failure ✅ (https://en.wikipedia.org/wiki/Delta-sigma_modulation):
"Delta modulation suffers from slope overload if signals move too fast."

### 2.3 The low-rate cliff in (a) — mechanism

📐 From §2.1(a), the quantizer input is `t[n] = d[n] − Σ a_k q_err[n−k]`. A fixed-step,
fixed-range scalar quantizer must cover the *peak* of `t`, which carries the filtered past
error. At rate `R = log₂M` the step is `Δ = 2·range/M`; as `M` falls, `Δ` and `σ_q²` rise,
which *further inflates* `Σ a_k q_err[n−k]` → more overload → larger effective `q_err`. This
positive term creates a threshold in `M` below which the loop is overload-dominated and the
SNR-vs-rate slope becomes steeper than 6 dB/bit — the cliff. 🔶 (derivation is mine; the
qualitative granular/overload tradeoff is ✅).

Measured in this tree ✅ (`docs/VOICE_RD.md` §2.2, encoder-side `diag`):

```
budget  used      rate        SNR
  64 B   58.5 B   23.4 kbps   6.92 dB
  96 B   91.7 B   36.7 kbps  19.53 dB     +12.6 dB for 2.75x rate
```

At a clean 6 dB/bit the expected gain is `6·log₂(2.75) = 8.8 dB`; measured is `12.6 dB`, i.e.
steeper than scalar granular scaling — consistent with relief of overload, not just a finer
step. 🔶

### 2.4 Exact error-feedback / noise-shaping formulation and stability

The general **error-feedback (noise-shaping) quantizer** ✅ (https://en.wikipedia.org/wiki/Noise_shaping
gives the first-order case `y[n]=x[n]+b·e[n−1]`, `e[n]=y_q[n]−y[n]`):

```
w[n] = x[n] − Σ_{k=1}^{K} c_k · e_q[n−k]      # feed back past quantization error
y[n] = Q(w[n])
e_q[n] = y[n] − w[n]
```

📐 Taking z-transforms:

$$ Y(z) = X(z) + \big(1 - C(z)\big)E_q(z),\qquad C(z)=\sum_{k=1}^{K} c_k z^{-k} $$

The **noise transfer function** is `NTF(z) = 1 − C(z) = Y(z)/E_q(z)` evaluated with `X=0`.
To force the output noise to follow `1/W(z)`:

$$ \mathrm{NTF}(z)=\frac{1}{W(z)} \;\Longrightarrow\; \boxed{\,C(z)=1-\frac{1}{W(z)}=\frac{W(z)-1}{W(z)}\,} $$

**Stability conditions.**
1. `C(z)` causal and all poles strictly inside `|z|=1` (BIBO). For `W(z)=A(z/γ₁)/A(z/γ₂)`,
   `C(z)=1 − A(z/γ₂)/A(z/γ₁)`; its poles are the roots of `A(z/γ₁)`, pulled toward the origin
   by `γ₁<1` because `A` is minimum-phase ⇒ **C is stable** ✅+📐.
2. `|C(e^{jω})|` may exceed 1 (noise *enhancement* in bands), so intermediate states need
   headroom; the feedback must be computed with sufficient dynamic range.
3. First-order (`K=1`) noise-feedback loops are **unconditionally stable**; higher orders
   require explicit stability analysis ✅ (ΔΣ page: "first-order modulators are unconditionally
   stable, stability analysis must be performed for higher-order noise-feedback modulators").
4. Whereas a *feedforward* noise shape (put `1−C` in the forward path) is always stable ✅
   (same page: "noise-feedforward configurations are always stable and have simpler analysis").

### 2.5 How the two real remedies avoid it

**(a) Analysis-by-synthesis CELP.** ✅ (Aalto ITSp CELP note;
https://raw.githubusercontent.com/Speech-Interaction-Technology-Aalto-U/itsp/main/Transmission/Code-excited_linear_prediction_CELP.md):
"Since LPC-filtering is autoregressive (IIR), it … has a non-linear effect on the output such
that quantization has a non-linear effect on the output. We therefore cannot know which
quantization is the best one without trying out *all* of them." CELP replaces the fixed scalar
quantizer with a **gain-normalized excitation vector** chosen to minimize the *weighted
synthesized* error `‖W(x − x̂)‖²` (see §4). Two independent overload defenses:

- **Gain adaptivity**: `x̂ = γ_{F0}·x_{F0} + γ_noise·x_noise`, with `γ*` quantized per subframe,
  so the effective step tracks the residual's level — no slope overload.
- **Closed-loop search through `1/A(z)`**: the candidate error is evaluated *after* synthesis,
  so the search accounts for exactly the coloring that recursion §2.1(b) would otherwise inflict.

The residual after LPC+F0 is modeled as Laplacian, and algebraic codebooks of constant `1`-norm
match that pdf ✅ (same note). Vector codebook ⇒ granular gain of §1.2 applies.

**(b) SILK noise-shaping quantizer (NSQ).** ✅ primary source: `draft-vos-silk-01` text
in-tree (`session3.md` lines 115255–115448, 115708–115727). NSQ avoids overload by
**per-subframe gain normalization** and avoids colored noise by **explicit analysis/synthesis
shaping**:

```
prefilter output × adjustment gain G
  + synthesis shaping filter output
  − prediction filter output
  = residual  →  × (inverse quantized quantization gain)  →  scalar quantizer
quantizer indices → pyramid range coder
quantizer output × (quantized quantization gain) = excitation
excitation + prediction-filter output = quantized output y(n) → back into shaping/prediction filters
```

The shaping filters ✅ (SILK draft eqs. on pp. 14–16):

$$ H(z)=G\,(1-c_{tilt}z^{-1})\frac{W_{ana}(z)}{W_{syn}(z)},\qquad
W_{ana}(z)=\Big(1-\sum_{k=1}^{16}a_{ana}(k)z^{-k}\Big)\Big(1-z^{-L}\sum_{k=-d}^{d}b_{ana}(k)z^{-k}\Big) $$

with `a_ana(k)=a(k)·g_ana^k`, `a_syn(k)=a(k)·g_syn^k`, and

$$ g_{ana}=0.94-0.02C,\qquad g_{syn}=0.94+0.02C,\qquad C\in[0,1] $$

`b_ana = F_ana·[0.25,0.5,0.25]`, `b_syn = F_syn·[0.25,0.5,0.25]` (voiced only, 3 taps);
`c_tilt = 0.4` unvoiced, `c_tilt = 0.04 + 0.06C` voiced. Because `g_ana < g_syn` (more
bandwidth expansion on analysis than synthesis), spectral valleys are de-emphasized, which
"reduces the entropy of the signal … thus lowering the bitrate" ✅. The stated design intent:
"The quantization gains determine the step size … increasing the quantization gain amplifies
quantization noise, but also reduces the bitrate"; noise is shaped to "follow the signal
spectrum … In practice, best results are obtained by making the shape of the noise spectrum
slightly flatter than the signal spectrum" ✅. SILK additionally uses delayed-decision
(trellis) pulse selection (`SKP_Silk_NSQ_del_dec`), i.e. it searches pulse positions under the
shaped-error criterion rather than committing greedily.

**Why gain normalization kills slope overload:** the step scales with the local residual
energy, so `Δ/σ_d` stays ~constant across the frame instead of `Δ` being fixed against a
Laplacian tail. This is the same idea CELP gets from joint gain quantization.

---

## 3. Error-feedback filter for `W(z) = A(z/γ₁)/A(z/γ₂)`

Derivation, 📐 (uses the §2.4 result; `NTF=1−C`):

$$ W(z)=\frac{A(z/\gamma_1)}{A(z/\gamma_2)},\qquad
\mathrm{NTF}(z)=\frac{1}{W(z)}=\frac{A(z/\gamma_2)}{A(z/\gamma_1)} $$

$$ \boxed{\,C(z)=1-\frac{1}{W(z)}=1-\frac{A(z/\gamma_2)}{A(z/\gamma_1)}
=\frac{A(z/\gamma_1)-A(z/\gamma_2)}{A(z/\gamma_1)}\,} $$

Concretely, with `A(z/γ)=1−Σ a_k γ^k z^{−k}`:

`C(z) = [ Σ_k a_k(γ₁^k − γ₂^k) z^{−k} ] / [ 1 − Σ_k a_k γ₁^k z^{−k} ]`.

Feedback coefficients for a direct-form-II / transversal implementation are obtained by inverse-z
of `C(z)` (an all-pole `1/A(z/γ₁)` applied to the numerator `Σ a_k(γ₁^k−γ₂^k)z^{−k}`).
**Stability** as §2.4: poles inside the unit circle (⇒ stable), gain may exceed 1 (⇒ headroom),
order `≥2` needs nonlinear analysis.

### Standard γ values and rationale

| codec | weighting | values | tag |
| --- | --- | --- | --- |
| AMR (TS 26.090) | `W(z)=A(z/γ₁)/A(z/γ₂)` | γ₁ = **0.92**, γ₂ = **0.6** (γ₁ varies by bit rate) | ✅ arXiv 1905.09754 |
| AMR-WB (TS 26.190) / EVS (TS 26.445) | equivalent `W'(z) = 1−A'(z/γ₁)`; pre-emphasis `H_pre(z)=1−βz^{−1}` | **β = 0.68** | ✅ arXiv 1905.09754 |
| this tree (VOLE) | `W(z)=A(z/γ₁)/A(z/γ₂)` | γ₁ = **0.9**, γ₂ = **0.6** | ✅ `src/voice/predict.rs` L70–76 |
| G.729 | adaptive γ₂ from min LSP spacing | γ₂ = 1 − 6π·d_min (reported) | 🔶 secondary source (dsprelated forum thread); verify against the standard |
| SILK (NSQ) | bandwidth-expansion pair | g_ana = 0.94−0.02C, g_syn = 0.94+0.02C | ✅ SILK draft, in-tree |

**Perceptual rationale** ✅ (arXiv 1905.09754): `A(z/γ₁)` in the numerator bandwidth-expands the
poles, so near a formant `|W|<1` and the allowed error `1/|W|` is *larger* there; `A(z/γ₂)` in
the denominator concentrates poles more, so in the spectral *valleys* `|W|` is larger and the
allowed error is *smaller*. "More energy of the quantization error will be in the speech
formant regions, as `1/W(z)` is somewhat below the spectral envelope there." For AMR-WB/EVS the
single-`γ₁` form plus pre-emphasis `β` plays the two-parameter role. The rule of thumb is
`γ₁ > γ₂`, `γ₁ ∈ [0.9, 0.994]`, `γ₂ ∈ [0.2, 0.8]`; larger `γ₁` ⇒ flatter weighting (closer to
plain MSE); larger `γ₁−γ₂` ⇒ more aggressive formant-biased shaping 🔶.

---

## 4. The perceptual weighting filter in CELP

**Exact definition** ✅ (Wikipedia "Code-excited linear prediction", in `session3.md`;
arXiv 1905.09754 eq. (2)):

$$ W(z)=\frac{A(z/\gamma_1)}{A(z/\gamma_2)},\qquad \gamma_1>\gamma_2 . $$

**Why W-weighted error beats plain MSE** ✅ (arXiv 1905.09754): minimizing the weighted error
makes the *weighted* error spectrally white, so "the final (unweighted) coding error is
proportional to the inverse weighting filter `1/W(z)`", and `1/W` is "somewhat below the
spectral envelope" at the formants. Error is therefore concentrated where the speech energy
(and hence masking) is large — exactly the behaviour MSE cannot express. The review ✅
(O'Shaughnessy 2023) puts the same point as: "some coders weight SNR so that the coding noise
is shaped to exploit masking … 'hide' quantization noise in frequency ranges of speech where
speech is strong."

**Application in analysis-by-synthesis.** ✅+📐 Write `x̂ = H·u` with `H` the synthesis
convolution matrix. With `W` the weighting convolution matrix:

$$ \arg\min_{u}\; d_W(x,\hat x)=\|W(x-Hu)\|^2=\|Wx-(WH)u\|^2 . $$

So the implementation is literally:

1. **filter the target** by `W`: `x_w = W·x` (in practice, target = original minus the
   zero-input response of the weighted synthesis filter);
2. **filter the impulse response** by `W` (equivalently precompute the impulse response of
   `W(z)/A(z)`), yielding `h_w[n]`;
3. correlate each candidate against `x_w` through `h_w`.

For a scaled codevector `γ·y`, eliminating `γ` analytically gives the standard closed forms ✅
(Aalto ITSp CELP note):

$$ \gamma^\star=\frac{x^{T}W^{T}W\,y}{y^{T}W^{T}W\,y},
\qquad
y^\star=\arg\max_{y}\frac{\big(x^{T}W^{T}W\,y\big)^2}{y^{T}W^{T}W\,y}. $$

**Measured benefit.** I found **no fetched codec-level ΔMOS**. What is fetched and quantified
is the *mechanism* (arXiv 1905.09754, DNN-enhancement loss, AMR weighting filter,
16 kHz, `γ₁=0.92, γ₂=0.6`): replacing MSE with the weighted loss improved **PESQ by 0.07–0.11**
and **ΔSNR by ≈0.6 dB**, and **SNRI by 3.5–4.5 dB**, while **SSDR (plain MSE) got *worse* by
1.5–2.5 dB** — direct evidence that weighted error deliberately trades MSE for perceptual
quality. 🔶 A codec-level benefit of order +0.2–0.5 MOS for adding perceptual weighting at
fixed rate is the conventional engineering expectation; do not cite a specific number without a
listening test. This tree's own record ✅ (`docs/VOICE_RD.md` P1) is that weighting applied to a
*broken* quantizer lowered plain SNR and breached the 5 ms encode budget — so re-judge on
ViSQOL after the excitation coder is efficient, not on SNR.

---

## 5. Entropy coding: adaptive arithmetic/range vs. Rice/Golomb

### 5.1 When Rice/Golomb is already near-optimal

Golomb codes are the **optimal prefix code for a geometric distribution** ✅
(https://en.wikipedia.org/wiki/Golomb_coding). For a matched geometric source, the measured
redundancy is tiny ✅ (same page): for `p(0)=0.2`, `M=3`, entropy `3.610` bits vs code rate
`3.639` bits ⇒ **redundancy 0.030 bits (0.83%)**. For a run-length case with `p=0.99`, `b=6`,
Rice attains `91.89%` compression vs the `91.92%` entropy limit (0.03 pp) ✅. Independent check
on a Bernoulli(0.95/0.05) source ✅ (https://en.wikipedia.org/wiki/Arithmetic_coding):
arithmetic coding `≈71.4%` vs "Golomb-Rice code with a four-bit remainder … 71.1%" — a
**~0.4% relative** difference. Conclusion 🔶: **if your symbols are genuinely memoryless
geometric and the Rice parameter is chosen well, an arithmetic coder saves ~0–1%.**

### 5.2 Where the arithmetic/range coder actually earns bits

The gains are **not** from the coding algorithm; they are from **context modeling** and from
**non-geometric / non-stationary** distributions, which Rice/Golomb's single parameter cannot
track:
- Arithmetic coding "applies especially well to adaptive data compression tasks where the
  statistics vary and are context-dependent" ✅ (https://en.wikipedia.org/wiki/Data_compression).
- Empirically, re-coding Huffman JPEG into arithmetic/ANS with context models (JPEG XL,
  PackJPG, Brunsli, Lepton) shows "**up to 25% size saving**" ✅
  (https://en.wikipedia.org/wiki/Arithmetic_coding). Baseline is Huffman, not Rice, and the
  gain includes context modeling — but it is the right order of magnitude for a *modeled*
  source.
- Arithmetic coding's own overhead is bounded: "at most 1 bit" per message (worked example
  8 bits vs 7.381 bits entropy = 8.4% on a 3-symbol message, vanishing as length grows), and
  with a *wrong* model it can **expand** data ✅ (same page).

**Typical saving, adaptive/context arithmetic vs single-parameter Rice, low rate** 🔶:

| symbol class | typical saving | why |
| --- | --- | --- |
| gains, pitch lag, LSF/VQ **indices** (non-geometric, non-stationary, strong context) | **≈10–25%** | tails + adaptation + context (e.g. lag context on previous lag) |
| Laplacian residual excursions, stationary, parameter tracked per block | **≈1–5%** | near-geometric locally |
| **ACELP pulse positions** (near-uniform, algebraic) | **≈0%** | already entropy-maximal; use combinatorial enumeration |

The last row is the important engineering result. ACELP pulse positions are *designed*
near-uniform, so there is nothing for an arithmetic coder to exploit; the correct tool is
enumerative/combinatorial coding. This tree already does exactly that ✅ (`docs/VOICE_RD.md` §2.3):
pulse positions coded as `⌈log₂ C(80,n)⌉` bits — 21 bits for four pulses instead of 28, 29
instead of 42. SILK, by contrast, range-codes every gain/LSF/LTP index against trained tables ✅
(SILK draft §2.1.2.10 "Range Encoder"; and `docs/VOICE_RD.md` §3). The native codec here is rANS
with `scale_bits=14`, `MODEL_TOTAL=16384`, `STATE_L=2^23`, verified byte-parity against an ryg
oracle ✅ (`docs/RANS.md`) — so the machinery to exploit §5.2 already exists and is frozen;
what is missing is applying it to transmitted model indices, with the model bytes charged
honestly ✅ (`docs/ENTROPY_ACCOUNTING.md`: "A 100-byte payload backed by a 512-byte model is
not a 100-byte representation").

---

## 6. What this means for a 16 kHz CELP codec (this tree)

Ordered by measured loss (`docs/VOICE_RD.md`, court `learned-voice-stream`, frozen corpus) ✅:

| priority | mechanism | theory anchor | status here |
| --- | --- | --- | --- |
| P1 | **Weighted error inside AbS** | §4 | implemented in `celp.rs`; scalar path does not use it |
| P2 | **Efficient ACELP excitation** (interleaved tracks, lag once/frame + differential, joint gains, bounded beam search) | §1.2, §2.5(a) | current coder is a net −1.6 dB at 12 kbps because 23 b/subframe overhead + 8 b/pulse |
| P3 | Fractional pitch (1/4 sample) + interpolation | §1.5 memory gain | integer-lag single-tap adds only 0.66 dB open-loop |
| P4 | ISP domain + MA prediction over transmitted indices | §1.3, §5.2 | LSF MSVQ already cut model 40–128 b → 28 b |
| P5 | **Entropy-code transmitted indices only** (not pulse positions) | §5.2 | rANS frozen; adaptive/shared model needed |
| P6 | Subframe bit allocation + voicing mode switching | §2.3, §2.5 | not implemented |

Measured position at matched actual bitrate ✅ (`docs/VOICE_RD.md` §1): −7.15 dB SNR vs Opus,
−4.32 dB vs EVS after combinatorial pulse coding; per-cell VOLE SNR ≈ 6.7–10.9 dB vs Opus
17.2–19.9 dB. The single largest identified loss is the **scalar-residual overload cliff**
(§2.3), which is a *fixed-scalar-quantizer* defect, not a predictor defect (prediction gain is
healthy at 18.3 dB ✅).

**Three mechanics to get right in the rewrite, from §1–§5:**

1. **Don't expect more than ~1.5 dB from "VQ instead of SQ" per se.** The 3–6 dB at low rate
   comes from *leaving the overload regime* and *gaining granular+shaping* together. Budget it
   as such. 📐
2. **Make the excitation gain-normalized and searched in the weighted synthesis domain** (§2.5,
   §4). That is what removes the cliff; a finer scalar step alone only moves down the slope.
3. **Spend entropy coder effort only on gains/lags/LSF indices** (§5.2). Pulse positions are
   already optimal under combinatorial coding; arithmetic-coding them is wasted complexity.

---

## 7. Sources

**Fetched this session (✅):**

1. https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory — Gaussian `R(D)`, Shannon lower bound.
2. https://en.wikipedia.org/wiki/Quantization_(signal_processing) — 6.02 dB/bit, `Δ²/12`, granular vs overload, Lloyd–Max.
3. https://en.wikipedia.org/wiki/Noise_shaping — first-order feedback `y=x+b·e[n−1]`, "any feedback loop functions as a filter".
4. https://en.wikipedia.org/wiki/Delta-sigma_modulation — `NTF=(1−z^{−1})^Θ`, first-order unconditional stability, slope overload.
5. https://en.wikipedia.org/wiki/Golomb_coding — geometric optimality, `0.030`-bit redundancy example, `91.89%` vs `91.92%`.
6. https://en.wikipedia.org/wiki/Arithmetic_coding — Bernoulli(0.95) AC vs Golomb-Rice (~0.4% relative), ≤1-bit overhead, "up to 25% size saving" (JPEG XL/PackJPG).
7. https://en.wikipedia.org/wiki/Data_compression — AC for context-dependent/adaptive statistics.
8. https://en.wikipedia.org/wiki/Shaping_codes — shaping gap 1.53 dB, Shell mapping ≈0.8 dB, Forney trellis shaping.
9. https://en.wikipedia.org/wiki/Constellation_shaping — probabilistic vs geometric shaping.
10. https://en.wikipedia.org/wiki/Vector_quantization — LBG, density matching (thin on quantitative gain).
11. https://en.wikipedia.org/wiki/G.729 and https://en.wikipedia.org/wiki/Adaptive_Multi-Rate_audio_codec — codec parameters, MOS/PSQM context.
12. https://en.wikipedia.org/wiki/FLAC — Rice-coding entropy stage, 40–50% lossless reduction.
13. https://ar5iv.labs.arxiv.org/html/1905.09754 — Zhao, Elshamy & Fingscheidt, *A Perceptual Weighting Filter Loss for DNN Training in Speech Enhancement*: AMR `W(z)` eq. (2), γ₁=0.92/γ₂=0.6, AMR-WB/EVS `1−A'(z/γ₁)` + β=0.68, `1/W` coloring, PESQ/ΔSNR/SSDR results.
14. https://link.springer.com/article/10.1186/s13636-023-00274-x — O'Shaughnessy, *Review of methods for coding of speech signals*, EURASIP JASMP 2023 (open access).
15. https://raw.githubusercontent.com/Speech-Interaction-Technology-Aalto-U/itsp/main/Transmission/Code-excited_linear_prediction_CELP.md — AbS weighted norm, optimal gain/correlation forms, Laplacian→algebraic codebook argument.
16. https://www.dsprelated.com/groups/speechcoding/kw/Perceptual.php — **secondary**: G.729 `γ₂ = 1 − 6π·d_min` (forum snippet).

**Primary specifications / drafts used in-tree (✅ as committed bytes):**

17. `draft-vos-silk-01`, in `session3.md` lines 115255–115448 (noise-shaping analysis) and 115708–115727 (NSQ) — shaping filters, gains, range coder.
18. Wikipedia "Code-excited linear prediction" text, in `session3.md` lines 6886–6945 — `W(z)` definition.

**In-tree measurements and normative specs (✅):**

19. `docs/VOICE_RD.md` — 18.31/18.97 dB prediction gain, scalar RD table and cliff, CELP A/B, competitor gaps, P1–P6 plan.
20. `docs/VOICE_VQ.md` — LSF MSVQ 28-bit wire index; codebook-as-shared-decoder-resource accounting.
21. `docs/RANS.md` — frozen rANS parameters and oracle parity.
22. `docs/ENTROPY_ACCOUNTING.md` — complete-cost rules.
23. `src/voice/predict.rs` — `WEIGHT_GAMMA1=0.9`, `WEIGHT_GAMMA2=0.6`; `close_loop()` reconstruction-fed scalar loop.

**Standard references named but NOT fetched here (📖) — verify before quoting a digit:**

- A. Gersho & R. M. Gray, *Vector Quantization and Signal Compression*, Kluwer, 1992 — Zador–Gersho high-resolution distortion, normalized second moments, shaping/granular decomposition.
- J. H. Conway & N. J. A. Sloane, *Sphere Packings, Lattices and Groups*, 3rd ed., Table 2.3 — `G(Λ)` for Z, A₂, A₃\*, D₄, E₈, Λ₁₆, Λ₂₄.
- N. S. Jayant & P. Noll, *Digital Coding of Waveforms*, Prentice-Hall, 1984 — DPCM error analysis, granular/overload, one-word memory DPCM.
- G. D. Forney Jr., "Trellis shaping," IEEE T-IT 38(2), 1992 — 1.53 dB shaping gain.
- J. Makhoul, A. Roucos & H. Gish, "Vector quantization in speech coding," Proc. IEEE 73(11), 1985 — speech-specific VQ-vs-SQ gains.
- 3GPP TS 26.090 / 26.190 / 26.445 — AMR / AMR-WB / EVS weighting parameters (cited transitively via ✅ #13).
- ITU-T G.729 — adaptive `γ₂`.

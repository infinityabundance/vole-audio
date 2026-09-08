# DSFB SEARCH — zero-authority search governance (H.2.26–29)

> Owner of the DSFB integration boundary. Prior art:
> `infinityabundance/dsfb` (published crate 0.1.2), VOLE Video
> `docs/dsfb-search.md` and ADR 0003 (dsfb zero authority). DSFB is
> encoder/search governance only; it has **zero decoder authority**.

## Role in H.2

Full general inverse proceduralization remains Phase K. H.2 uses DSFB to
govern H.2's already-defined **bounded entropy search universe**, giving
Phase K a working governance pattern and an entropy-cost oracle. Candidate
dimensions in the H.2 universe (H.2.27):

- raw vs rANS;
- symbolization;
- model family (inline per-page, shared, pooled);
- model sharing mode;
- entropy page size;
- residual coding choice;
- existing procedural hypothesis + entropy residual;
- literal entropy floor.

For each candidate `h`, the observer exposes:

- current encoded residual bytes;
- residual zero fraction;
- residual entropy estimate;
- cost improvement (first and second cost difference);
- model cost;
- search effort;
- candidate stability.

DSFB may use drift/slew/trust/regime change to decide which expensive
candidates to evaluate next. It must never change normative samples, alter
entropy decoding, make an invalid candidate valid, bypass exact closure,
suppress the universal fallback, or override the final exact complete-cost
comparison.

## Adapter

- Prefer a thin adapter around the actual `dsfb` implementation where its
  published API fits.
- If the published API is insufficient, implement only the **minimum
  deterministic observer** needed inside `src/search/dsfb.rs`, preserve
  conceptual provenance in comments/docs, and never invent a second unrelated
  algorithm and call it DSFB.
- Default playback must not link or execute DSFB (feature `dsfb`,
  default-off, acceptable if useful).

## Strategies over ONE candidate universe (H.2.28)

Comparable encoder-side strategies:

- `Exhaustive` — evaluate every candidate in the universe;
- `FixedHeuristic` — a frozen deterministic ordering with a bounded budget;
- `DsfbGuided` — DSFB-governed evaluation order with a bounded budget.

The complete valid candidate set is identical across strategies; the
strategies differ **only** in which candidates are evaluated, evaluation
order, and the bounded search budget. Universal fallback can never be
suppressed.

## Success criterion (H.2.29)

Primary: `N_dsfb < N_exhaustive` while achieving `J_dsfb == J_exhaustive`
where possible; `N` = search work/candidates, `J` = complete representation
cost. For regime-changing content, bounded regret is acceptable only when
measured, explicit, and reproducible. **If a fixed heuristic is cheaper and
equal, DSFB loses — record it.** DSFB does not "compress audio".

## Claim boundary (H.2.41)

A successful DSFB entropy court permits stating: *DSFB reduced bounded
candidate-search work on the measured corpus while retaining equal or
measured-near-oracle representation cost.* It does not permit claiming: DSFB
compresses audio, improves the decoder, is needed for playback, determines
truth, or bypasses exhaustive validation.

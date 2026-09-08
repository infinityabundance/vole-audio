# ADR 0003 — DSFB has zero authority in VOLE-Audio

- Status: accepted (Phase H.2)
- Owner: `docs/DSFB_SEARCH.md`

## Context

VOLE's DSFB work governs expensive search with a drift–slew–fusion observer
budgeted by *trust*. VOLE-Audio must not let any search governor reach the
decoder, the normative samples, or the universal fallback.

## Decision

DSFB (the real published `dsfb` crate behind the default-off `dsfb` feature,
plus a minimum deterministic observer in `src/entropy/search/dsfb.rs` where
the published API needs adaptation) governs **encoder-side candidate search
only**:

- Search universe in H.2: representation/symbolization/model/page-size/
  residual-coding choices over frozen candidates (Phase K later expands the
  mechanism to the procedural hypothesis universe).
- All strategies (`exhaustive`, `fixed-heuristic`, `dsfb-guided`) operate over
  the **same valid candidate set** and differ only in evaluation order and
  bounded work; DSFB may use drift/slew/trust/regime detection to choose
  which expensive candidates to evaluate next.
- DSFB must never: change normative samples, alter entropy decoding, make an
  invalid candidate valid, bypass exact closure, suppress universal fallback,
  override the final exact complete-cost comparison, or be required by
  playback.
- Success criterion: `N_dsfb < N_exhaustive` while `J_dsfb == J_exhaustive`
  where measurable; if a fixed heuristic is cheaper and equal, DSFB loses —
  recorded. DSFB does not "compress audio"; it budgets search work.

## Consequences

Without the feature, `court dsfb-entropy` reports `INCONCLUSIVE` plus a
limitation. Default playback never links or executes DSFB.

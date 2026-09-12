//! Admissible search bounds (Phase 6 mechanism 10, `AdmissibleSearchBounds`).
//!
//! A bounded encoder search spends most of its time fully evaluating candidates
//! that a cheap, *provable* lower bound would already have rejected. This module
//! provides the general instrument and one concrete integration.
//!
//! The correctness rule is exactly one inequality. If `lower_bound(i)` is at
//! most the true exact cost of candidate `i`, then
//!
//! ```text
//! lower_bound(i) >= best_exact  =>  exact(i) >= best_exact
//! ```
//!
//! so candidate `i` cannot beat the incumbent and may be skipped. Candidates are
//! processed in ascending lower-bound order, so the first candidate whose bound
//! reaches the incumbent proves that *every* remaining candidate is dead too:
//! the bounded result is identical to exhaustive exact evaluation, including
//! ties (broken by ascending candidate index).
//!
//! The bound must be *admissible* — never a heuristic guess. A bound that is
//! merely usually tight can silently drop the true winner, which is why the
//! court compares the bounded winner with the exhaustive winner on every case.

/// The outcome of an admissible-bounded selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissibleReport {
    /// Index of the winning candidate (`None` when every candidate is invalid).
    pub winner: Option<usize>,
    /// The winner's exact cost (`u64::MAX` when no candidate is valid).
    pub winner_cost: u64,
    /// Total candidates considered.
    pub candidates: usize,
    /// Candidates whose exact cost was evaluated.
    pub evaluated: usize,
    /// Candidates skipped because their admissible lower bound could not beat
    /// the incumbent.
    pub pruned: usize,
}

impl AdmissibleReport {
    /// True when the search pruned at least one candidate.
    pub fn pruned_any(&self) -> bool {
        self.pruned > 0
    }
}

/// Select the minimum-exact-cost candidate under admissible lower bounds.
///
/// `lower_bound(i)` must satisfy `lower_bound(i) <= exact_cost(i)` for every
/// `i`. Cost ties are broken by ascending candidate index, matching an
/// exhaustive in-order minimum with a strict `<` comparison; the pruning rule
/// is therefore `lower_bound > best`, or `lower_bound == best` with a larger
/// index than the incumbent (an equal-cost candidate with a *smaller* index
/// would win the tie and is still evaluated).
pub fn admissible_select<L, E>(
    count: usize,
    mut lower_bound: L,
    mut exact_cost: E,
) -> AdmissibleReport
where
    L: FnMut(usize) -> u64,
    E: FnMut(usize) -> u64,
{
    if count == 0 {
        return AdmissibleReport {
            winner: None,
            winner_cost: u64::MAX,
            candidates: 0,
            evaluated: 0,
            pruned: 0,
        };
    }
    let mut order: Vec<(u64, usize)> = (0..count).map(|i| (lower_bound(i), i)).collect();
    order.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut winner: Option<usize> = None;
    let mut best = u64::MAX;
    let mut evaluated = 0usize;
    let mut pruned = 0usize;
    for (lb, i) in order {
        if let Some(w) = winner
            && (lb > best || (lb == best && i > w))
        {
            pruned += 1;
            continue;
        }
        let c = exact_cost(i);
        evaluated += 1;
        match winner {
            None => {
                best = c;
                winner = Some(i);
            }
            Some(w) => {
                if c < best || (c == best && i < w) {
                    best = c;
                    winner = Some(i);
                }
            }
        }
    }
    AdmissibleReport {
        winner,
        winner_cost: best,
        candidates: count,
        evaluated,
        pruned,
    }
}

/// Exhaustive reference: the minimum exact cost in ascending index order.
pub fn exhaustive_select<E>(count: usize, mut exact_cost: E) -> AdmissibleReport
where
    E: FnMut(usize) -> u64,
{
    let mut winner: Option<usize> = None;
    let mut best = u64::MAX;
    for i in 0..count {
        let c = exact_cost(i);
        if c < best {
            best = c;
            winner = Some(i);
        }
    }
    AdmissibleReport {
        winner,
        winner_cost: best,
        candidates: count,
        evaluated: count,
        pruned: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_equals_exhaustive_when_bounds_are_admissible() {
        // Deterministic candidate costs.
        let costs: Vec<u64> = (0..64u64).map(|i| (i * 2654435761) % 997 + 3).collect();
        let lo: Vec<u64> = costs.iter().map(|&c| c / 2).collect();
        let bounded = admissible_select(costs.len(), |i| lo[i], |i| costs[i]);
        let exhaustive = exhaustive_select(costs.len(), |i| costs[i]);
        assert_eq!(bounded.winner, exhaustive.winner);
        assert_eq!(bounded.winner_cost, exhaustive.winner_cost);
        assert!(bounded.pruned > 0, "expected some pruning");
        assert_eq!(bounded.candidates, bounded.evaluated + bounded.pruned);
    }

    #[test]
    fn a_zero_lower_bound_never_prunes_anything() {
        let costs: Vec<u64> = (0..32u64).map(|i| i + 1).collect();
        let bounded = admissible_select(costs.len(), |_| 0, |i| costs[i]);
        assert_eq!(bounded.pruned, 0);
        assert_eq!(bounded.evaluated, 32);
        assert_eq!(bounded.winner, Some(0));
    }

    #[test]
    fn ties_break_on_the_smallest_index() {
        let costs = [5u64, 5, 5];
        let lo = [5u64, 5, 5];
        let bounded = admissible_select(costs.len(), |i| lo[i], |i| costs[i]);
        assert_eq!(bounded.winner, Some(0));
        let exhaustive = exhaustive_select(costs.len(), |i| costs[i]);
        assert_eq!(bounded.winner, exhaustive.winner);
    }

    #[test]
    fn tie_with_a_smaller_index_and_larger_bound_still_wins() {
        // Candidate 1 ties candidate 2 on exact cost but has the smaller index
        // and a larger (still admissible) lower bound, so it must be evaluated.
        let costs = [9u64, 4, 4];
        let lo = [9u64, 4, 2];
        let bounded = admissible_select(costs.len(), |i| lo[i], |i| costs[i]);
        let exhaustive = exhaustive_select(costs.len(), |i| costs[i]);
        assert_eq!(bounded.winner, Some(1));
        assert_eq!(bounded.winner, exhaustive.winner);
    }

    #[test]
    fn empty_and_single_candidate_are_well_defined() {
        let none = admissible_select(0, |_| 0, |_| 0);
        assert_eq!(none.winner, None);
        let one = admissible_select(1, |_| 7, |_| 7);
        assert_eq!(one.winner, Some(0));
        assert_eq!(one.winner_cost, 7);
    }
}

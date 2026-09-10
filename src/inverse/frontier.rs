//! Deterministic Pareto frontier over static candidate objectives.
//!
//! The frontier is a genuine Pareto set: candidate `a` dominates `b` when `a`
//! is no worse on every objective and strictly better on at least one. Costs
//! are never collapsed into a weighted "score".
//!
//! Objectives are **static** (structure-derived, reproducible):
//!
//! 1. `complete_bytes` — the complete stored cost (H.2 cost oracle);
//! 2. `total_ops` — abstract universe work of the representation;
//! 3. `seek_ops` — abstract work to materialize one bounded seek window.
//!
//! Measured wall times are reported per candidate but are deliberately **not**
//! objectives: a frontier whose membership depended on wall-clock noise would
//! not be reproducible, and the contract requires a deterministic frontier.

use super::Acceptance;

/// The non-dominated subset of an accepted-candidate list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Frontier {
    /// Indices into the accepted-candidate slice, in proposal order.
    members: Vec<usize>,
}

impl Frontier {
    /// Build the frontier from accepted candidates (in proposal order).
    pub fn build(accepted: &[Acceptance]) -> Frontier {
        let mut members = Vec::new();
        for (i, a) in accepted.iter().enumerate() {
            let dominated = accepted
                .iter()
                .enumerate()
                .any(|(j, b)| j != i && dominates(b, a));
            if !dominated {
                members.push(i);
            }
        }
        Frontier { members }
    }

    /// Member indices (into the accepted slice), in proposal order.
    pub fn indices(&self) -> &[usize] {
        &self.members
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn contains(&self, index: usize) -> bool {
        self.members.contains(&index)
    }

    /// The member with the smallest `complete_bytes` (ties -> lowest index).
    pub fn cheapest<'a>(&self, accepted: &'a [Acceptance]) -> Option<&'a Acceptance> {
        self.members
            .iter()
            .filter_map(|&i| accepted.get(i))
            .min_by_key(|a| a.cost.complete_bytes)
    }

    /// Self-check: no member dominates another (a valid Pareto set).
    pub fn is_valid(&self, accepted: &[Acceptance]) -> bool {
        for (n, &i) in self.members.iter().enumerate() {
            let Some(a) = accepted.get(i) else {
                return false;
            };
            for &j in &self.members[n + 1..] {
                let Some(b) = accepted.get(j) else {
                    return false;
                };
                if dominates(a, b) || dominates(b, a) {
                    return false;
                }
            }
        }
        true
    }
}

/// True when `a` dominates `b`: no worse on every objective, strictly better
/// on at least one.
pub fn dominates(a: &Acceptance, b: &Acceptance) -> bool {
    let av = a.objective();
    let bv = b.objective();
    av.iter().zip(bv.iter()).all(|(x, y)| x <= y) && av != bv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::CandidateKind;

    fn acceptance(complete: u64, total_ops: u64, seek_ops: u64) -> Acceptance {
        Acceptance {
            kind: CandidateKind::Literal,
            label: "test".into(),
            cost: crate::inverse::cost::CandidateCost {
                complete_bytes: complete,
                ..Default::default()
            },
            work: crate::inverse::cost::AbstractWork::default(),
            content_id: crate::object::id::ContentId([0; 32]),
            reference_target: None,
            intrinsic_exact: true,
            evaluator_exact: true,
            seek_exact: true,
            materialize_ns: 0,
            seek_latency_ns: 0,
            intrinsic_ns: 0,
            seek_start: 0,
            seek_frames: 0,
            accounted_peak_bytes: 0,
            proposal_ns: 0,
            seek_ops,
            total_ops,
        }
    }

    #[test]
    fn dominated_candidates_are_removed() {
        // b is worse in every objective -> dominated.
        let a = acceptance(100, 10, 1);
        let b = acceptance(200, 20, 2);
        let c = acceptance(50, 30, 3); // cheaper bytes, more work: non-dominated
        let list = vec![a, b, c];
        let f = Frontier::build(&list);
        assert_eq!(f.indices(), &[0, 2]);
        assert!(f.is_valid(&list));
        assert_eq!(f.cheapest(&list).unwrap().cost.complete_bytes, 50);
    }

    #[test]
    fn equal_objectives_do_not_dominate_each_other() {
        let a = acceptance(100, 10, 1);
        let b = acceptance(100, 10, 1);
        let list = vec![a, b];
        let f = Frontier::build(&list);
        assert_eq!(f.len(), 2, "identical vectors: neither dominates the other");
        assert!(f.is_valid(&list));
    }

    #[test]
    fn single_candidate_is_the_frontier() {
        let list = vec![acceptance(1, 1, 1)];
        let f = Frontier::build(&list);
        assert_eq!(f.indices(), &[0]);
        assert!(f.is_valid(&list));
    }
}

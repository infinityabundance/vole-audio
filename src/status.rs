//! Court/evidence status vocabulary — the single source of truth shared by
//! host receipts and device status words.
//!
//! Every measured result in vole-audio must be classified into exactly one of
//! these classes (paper §1 "Operating principle", implementation prompt §1).
//! A missing or ambiguous class is a defect, not a result.
//!
//! This module is `no_std`-clean so GPU kernels can emit the same codes as
//! plain `u8` status words without duplicating the vocabulary.

use core::fmt;

/// Result classification for courts, probes, and backend paths.
///
/// Wire representation is the `u8` discriminant (`as_u8`); device kernels and
/// host code share this module, so codes can never drift apart.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "std", serde(rename_all = "SCREAMING_SNAKE_CASE"))]
pub enum Verdict {
    /// The path worked and satisfied its correctness/deadline criteria.
    Supported = 0,
    /// The API surface does not expose the operation (runtime/library level).
    UnsupportedByApi = 1,
    /// The hardware (or driver/firmware combination) does not support it.
    UnsupportedByHardware = 2,
    /// The operation is impossible given the PCI/fabric topology.
    UnsupportedByTopology = 3,
    /// The path executed but produced incorrect results.
    FailedCorrectness = 4,
    /// The path executed correctly but missed its deadline budget.
    FailedDeadline = 5,
    /// A D1/D2 attempt was not possible and the run legitimately used D0;
    /// recorded explicitly rather than reported as success of D1/D2.
    FellBackToD0 = 6,
    /// Evidence is insufficient to classify (recorded with a reason).
    Inconclusive = 7,
    /// The class of operation does not apply to this court/configuration.
    NotApplicable = 8,
    /// Declared future/conceptual surface (e.g. D3 endpoint-native) — never a
    /// real result class, only a marker that the surface is not implemented.
    NotImplemented = 9,
}

impl Verdict {
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Map a device/host status byte back to a verdict.
    pub const fn from_u8(v: u8) -> Option<Verdict> {
        match v {
            0 => Some(Verdict::Supported),
            1 => Some(Verdict::UnsupportedByApi),
            2 => Some(Verdict::UnsupportedByHardware),
            3 => Some(Verdict::UnsupportedByTopology),
            4 => Some(Verdict::FailedCorrectness),
            5 => Some(Verdict::FailedDeadline),
            6 => Some(Verdict::FellBackToD0),
            7 => Some(Verdict::Inconclusive),
            8 => Some(Verdict::NotApplicable),
            9 => Some(Verdict::NotImplemented),
            _ => None,
        }
    }

    /// True for classes that represent an honest negative or partial result.
    pub const fn is_negative_or_inconclusive(self) -> bool {
        !matches!(self, Verdict::Supported)
    }

    /// Human phrase for CLI summaries.
    pub const fn label(self) -> &'static str {
        match self {
            Verdict::Supported => "SUPPORTED",
            Verdict::UnsupportedByApi => "UNSUPPORTED_BY_API",
            Verdict::UnsupportedByHardware => "UNSUPPORTED_BY_HARDWARE",
            Verdict::UnsupportedByTopology => "UNSUPPORTED_BY_TOPOLOGY",
            Verdict::FailedCorrectness => "FAILED_CORRECTNESS",
            Verdict::FailedDeadline => "FAILED_DEADLINE",
            Verdict::FellBackToD0 => "FELL_BACK_TO_D0",
            Verdict::Inconclusive => "INCONCLUSIVE",
            Verdict::NotApplicable => "NOT_APPLICABLE",
            Verdict::NotImplemented => "NOT_IMPLEMENTED",
        }
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable_and_roundtrip() {
        for v in [
            Verdict::Supported,
            Verdict::UnsupportedByApi,
            Verdict::UnsupportedByHardware,
            Verdict::UnsupportedByTopology,
            Verdict::FailedCorrectness,
            Verdict::FailedDeadline,
            Verdict::FellBackToD0,
            Verdict::Inconclusive,
            Verdict::NotApplicable,
            Verdict::NotImplemented,
        ] {
            assert_eq!(Verdict::from_u8(v.as_u8()), Some(v));
        }
        assert_eq!(Verdict::from_u8(200), None);
    }

    #[test]
    fn labels_are_distinct_and_upper_snake() {
        let mut seen = std::collections::BTreeSet::new();
        for v in [
            Verdict::Supported,
            Verdict::UnsupportedByApi,
            Verdict::UnsupportedByHardware,
            Verdict::UnsupportedByTopology,
            Verdict::FailedCorrectness,
            Verdict::FailedDeadline,
            Verdict::FellBackToD0,
            Verdict::Inconclusive,
            Verdict::NotApplicable,
            Verdict::NotImplemented,
        ] {
            assert!(seen.insert(v.label()), "duplicate label {}", v.label());
            assert!(
                v.label()
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
            );
        }
    }
}

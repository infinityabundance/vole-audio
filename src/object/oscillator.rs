//! Oscillator and partial-bank payloads.
//!
//! Both describe *endless* procedural sources: observation is a function of
//! media time (voice phase accumulation), never of stored samples, so no
//! resident object PCM exists. Base frequency domain `[1, fs_max/2]`; the
//! sample-rate-dependent Nyquist clamp happens at voice resolution
//! (`sampler::procedural::eff_incr`).

use crate::hash::sha256::Sha256;
use crate::object::descriptor::{ObjectDescriptor, Representation, canonical_header_bytes};
use crate::object::id::ContentId;
use crate::sampler::procedural::Partial;

/// Oscillator SampleObject parameters (frozen).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Oscillator {
    /// Base frequency in Hz (domain `[1, MAX_SAMPLE_RATE_HZ/2]`).
    pub freq_hz: u32,
    /// Amplitude Q16, unity `1<<16`; domain `[-(1<<17), 1<<17]`.
    pub amp_q16: i32,
}

impl Oscillator {
    pub const fn checked(freq_hz: u32, amp_q16: i32) -> Option<Oscillator> {
        if freq_hz == 0 || freq_hz > crate::limits::MAX_SAMPLE_RATE_HZ / 2 {
            return None;
        }
        if amp_q16.abs() > (1 << 17) {
            return None;
        }
        Some(Oscillator { freq_hz, amp_q16 })
    }
}

/// Canonical oscillator bytes: `header || freq(u32 LE) || amp(i32 LE)`.
pub fn oscillator_canonical_bytes(descriptor: &ObjectDescriptor, o: &Oscillator) -> Vec<u8> {
    let mut d = descriptor.clone();
    d.representation = Representation::Oscillator;
    let mut out = canonical_header_bytes(&d);
    out.extend_from_slice(&o.freq_hz.to_le_bytes());
    out.extend_from_slice(&o.amp_q16.to_le_bytes());
    out
}

/// Partial-bank SampleObject parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialBank {
    /// Fundamental frequency in Hz (domain as `Oscillator`).
    pub freq_hz: u32,
    /// Partial list (ascending distinct harmonics; validated by
    /// `sampler::procedural::validate_partials`).
    pub partials: Vec<Partial>,
}

impl PartialBank {
    pub fn checked(freq_hz: u32, partials: Vec<Partial>) -> Option<PartialBank> {
        if freq_hz == 0 || freq_hz > crate::limits::MAX_SAMPLE_RATE_HZ / 2 {
            return None;
        }
        if !crate::sampler::procedural::validate_partials(&partials) {
            return None;
        }
        Some(PartialBank { freq_hz, partials })
    }
}

/// Canonical partial-bank bytes:
/// `header || freq(u32 LE) || count(u32 LE) || (harmonic u32 LE, amp i32 LE)*`.
pub fn partial_bank_canonical_bytes(descriptor: &ObjectDescriptor, b: &PartialBank) -> Vec<u8> {
    let mut d = descriptor.clone();
    d.representation = Representation::PartialBank;
    let mut out = canonical_header_bytes(&d);
    out.extend_from_slice(&b.freq_hz.to_le_bytes());
    out.extend_from_slice(&(b.partials.len() as u32).to_le_bytes());
    for p in &b.partials {
        out.extend_from_slice(&p.harmonic.to_le_bytes());
        out.extend_from_slice(&p.amp_q16.to_le_bytes());
    }
    out
}

/// Content identity from canonical oscillator/partial-bank bytes.
pub fn content_id(bytes: &[u8]) -> ContentId {
    ContentId(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::universe::layout::Layout;

    fn od(rep: Representation) -> ObjectDescriptor {
        // Endless generators declare extent 0.
        ObjectDescriptor::new(rep, 0, Layout::Mono, None).unwrap()
    }

    #[test]
    fn oscillator_validation_and_identity() {
        assert!(Oscillator::checked(440, 1 << 16).is_some());
        assert!(Oscillator::checked(0, 1 << 16).is_none());
        assert!(Oscillator::checked(440, (1 << 17) + 1).is_none());
        let o = Oscillator::checked(440, 1 << 16).unwrap();
        let d = od(Representation::Oscillator);
        let a = content_id(&oscillator_canonical_bytes(&d, &o));
        let o2 = Oscillator::checked(441, 1 << 16).unwrap();
        let b = content_id(&oscillator_canonical_bytes(&d, &o2));
        assert_ne!(a, b);
        assert_eq!(content_id(&oscillator_canonical_bytes(&d, &o)), a);
    }

    #[test]
    fn partial_bank_validation_and_bytes() {
        let bank = PartialBank::checked(
            220,
            vec![
                Partial {
                    harmonic: 1,
                    amp_q16: 1 << 15,
                },
                Partial {
                    harmonic: 3,
                    amp_q16: 1 << 13,
                },
            ],
        )
        .unwrap();
        let d = od(Representation::PartialBank);
        let b = partial_bank_canonical_bytes(&d, &bank);
        // Deterministic and order-sensitive (direct construction bypasses the
        // ascending-order validator, which is tested separately below).
        let bank2 = PartialBank {
            freq_hz: 220,
            partials: vec![
                Partial {
                    harmonic: 3,
                    amp_q16: 1 << 13,
                },
                Partial {
                    harmonic: 1,
                    amp_q16: 1 << 15,
                },
            ],
        };
        let b2 = partial_bank_canonical_bytes(&d, &bank2);
        assert_ne!(b, b2, "partial order is canonical");
        // Out-of-order input is rejected by construction.
        assert!(
            PartialBank::checked(
                220,
                vec![
                    Partial {
                        harmonic: 2,
                        amp_q16: 1
                    },
                    Partial {
                        harmonic: 1,
                        amp_q16: 1
                    },
                ],
            )
            .is_none()
        );
    }
}

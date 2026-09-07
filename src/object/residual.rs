//! Residual-governed SampleObject (closure semantics, `no_std`-clean).
//!
//! For sampled-origin content the *intrinsic closure* is
//!
//! ```text
//! X_O(f, ch) = H(f, ch) + R(f, ch)          (exact, code domain)
//! ```
//!
//! with `H` a deterministic model hypothesis and `R` an explicit sparse
//! residual (frame/channel/delta records). Closure precedes observation:
//! sampler transforms apply after reconstruction. Interpolation of a
//! residual-governed object reads reconstructed values (H+R) at the two
//! neighboring frames, so integer-frame reads reproduce the closure exactly.
//!
//! Freeze (U1_SPEC §"Residual closure"):
//!
//! * Models in v1: `Zero`, `Constant(level)`, `Periodic(cycle)` (cycle
//!   content inline, mono). Model output is in the code domain (i32).
//! * Residual delta domain: `i32`, records unique per (channel, frame),
//!   sorted by (channel, frame); count `<= MAX_RESIDUAL_RECORDS`.
//! * Closure per (frame, channel): `sat_i32(H + R)`; exact reconstruction of
//!   an intrinsic `X_O` requires `|X_O - H| < 2^31` (the residual fits i32);
//!   otherwise the object is rejected at construction.
//! * Residual addition is a single i64 sum + one saturation — no transform
//!   is ever interleaved with the residual (no `T(H) + T(R)` commutation).
//! * Residual deltas are sample-domain content for exposure accounting.

use crate::hash::sha256::Sha256;
use crate::object::descriptor::{ObjectDescriptor, Representation, canonical_header_bytes};
use crate::object::id::ContentId;
use crate::universe::arithmetic::sat_i32;

/// Model hypotheses supported by v1 residual-governed objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidualModel {
    /// Zero everywhere.
    Zero,
    /// Constant level (code domain).
    Constant(i32),
    /// Mono periodic cycle (the model output at frame f is
    /// `cycle.samples[f % cycle_len]`).
    Periodic { cycle: Vec<i32> },
}

impl ResidualModel {
    pub const fn is_zero(&self) -> bool {
        matches!(self, ResidualModel::Zero)
    }

    /// Model output at intrinsic frame `f` (channel 0). Channels beyond the
    /// model's domain produce 0 (frozen; periodic model is mono).
    pub fn model_sample(&self, f: u64) -> i32 {
        match self {
            ResidualModel::Zero => 0,
            ResidualModel::Constant(l) => *l,
            ResidualModel::Periodic { cycle } => {
                if cycle.is_empty() {
                    0
                } else {
                    cycle[(f % cycle.len() as u64) as usize]
                }
            }
        }
    }
}

/// One sparse residual record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResidualRecord {
    pub frame: u64,
    pub channel: u8,
    pub delta: i32,
}

/// Residual-governed payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Residual {
    pub model: ResidualModel,
    /// Sorted by (channel, frame); unique per (channel, frame).
    pub records: Vec<ResidualRecord>,
}

impl Residual {
    /// Validate records and ceilings; `extent_frames` must exceed every
    /// record frame.
    pub fn new(
        descriptor: &ObjectDescriptor,
        model: ResidualModel,
        mut records: Vec<ResidualRecord>,
    ) -> Option<Residual> {
        if descriptor.extent_frames == 0 {
            return None;
        }
        if records.len() as u32 > crate::limits::MAX_RESIDUAL_RECORDS {
            return None;
        }
        if matches!(
            &model,
            ResidualModel::Periodic { cycle }
                if cycle.is_empty()
                    || cycle.len() > crate::limits::MAX_TABLE_BYTES as usize / 4
                    || descriptor.layout.count() != 1
        ) {
            return None;
        }
        let _ = &model;
        records.sort_by_key(|r| (r.channel, r.frame));
        for w in records.windows(2) {
            if w[0].channel == w[1].channel && w[0].frame == w[1].frame {
                return None; // duplicate
            }
        }
        for r in &records {
            if r.frame >= descriptor.extent_frames {
                return None;
            }
            if r.channel as u32 >= descriptor.layout.count() as u32 {
                return None;
            }
        }
        Some(Residual { model, records })
    }

    /// Compute the sparse residual that closes `intrinsic` exactly under a
    /// model (helper for tests and Phase K candidate verification):
    /// `delta = intrinsic - model`, rejected if it does not fit i32.
    pub fn closing_residual(
        intrinsic: &[i32],
        channels: u8,
        model: &ResidualModel,
    ) -> Option<Vec<ResidualRecord>> {
        let ch = usize::from(channels);
        let frames = intrinsic.len() / ch;
        if intrinsic.len() != frames * ch {
            return None;
        }
        let mut out = Vec::new();
        for f in 0..frames {
            for c in 0..ch {
                let target = i64::from(intrinsic[f * ch + c]);
                let h = if c == 0 {
                    i64::from(model.model_sample(f as u64))
                } else {
                    0
                };
                let delta = target - h;
                if delta < i64::from(i32::MIN) || delta > i64::from(i32::MAX) {
                    return None;
                }
                if delta != 0 {
                    out.push(ResidualRecord {
                        frame: f as u64,
                        channel: c as u8,
                        delta: delta as i32,
                    });
                }
            }
        }
        Some(out)
    }

    /// Closure sample: `sat_i32(H(f, ch) + R(f, ch))`. Binary search over the
    /// sorted records.
    pub fn closure_sample(&self, f: u64, channel: u8) -> i32 {
        let h = if channel == 0 {
            i64::from(self.model.model_sample(f))
        } else {
            0
        };
        let mut delta = 0i64;
        // Records sorted by (channel, frame); find matching channel/frame.
        let lo = self
            .records
            .partition_point(|r| r.channel < channel || (r.channel == channel && r.frame < f));
        if let Some(r) = self
            .records
            .get(lo)
            .filter(|r| r.channel == channel && r.frame == f)
        {
            delta = i64::from(r.delta);
        }
        sat_i32(h + delta)
    }

    /// Canonical payload bytes:
    /// `header || model_tag || model bytes || record_count(u64 LE) || records`
    pub fn canonical_bytes(&self, descriptor: &ObjectDescriptor) -> Vec<u8> {
        let mut d = descriptor.clone();
        d.representation = Representation::PredictorResidual;
        let mut out = canonical_header_bytes(&d);
        match &self.model {
            ResidualModel::Zero => out.push(0),
            ResidualModel::Constant(l) => {
                out.push(1);
                out.extend_from_slice(&l.to_le_bytes());
            }
            ResidualModel::Periodic { cycle } => {
                out.push(2);
                out.extend_from_slice(&(cycle.len() as u64).to_le_bytes());
                for s in cycle {
                    out.extend_from_slice(&s.to_le_bytes());
                }
            }
        }
        out.extend_from_slice(&(self.records.len() as u64).to_le_bytes());
        for r in &self.records {
            out.extend_from_slice(&r.frame.to_le_bytes());
            out.push(r.channel);
            out.extend_from_slice(&r.delta.to_le_bytes());
        }
        out
    }

    pub fn content_id(descriptor: &ObjectDescriptor, r: &Residual) -> ContentId {
        ContentId(Sha256::digest(&r.canonical_bytes(descriptor)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::descriptor::Representation;
    use crate::universe::layout::Layout;

    fn od(frames: u64) -> ObjectDescriptor {
        ObjectDescriptor::new(
            Representation::PredictorResidual,
            frames,
            Layout::Mono,
            None,
        )
        .unwrap()
    }

    #[test]
    fn closure_reconstructs_intrinsic_exactly() {
        // A periodic intrinsic: 1000 frames of a 50-frame triangle wave.
        let intrinsic: Vec<i32> = (0..1000)
            .map(|i| {
                let m = i % 50;
                let v = if m < 25 { m } else { 50 - m };
                v << 20
            })
            .collect();
        // Hypothesis: exact period-50 model (perfect) => empty residual.
        let model = ResidualModel::Periodic {
            cycle: intrinsic[..50].to_vec(),
        };
        let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
        assert!(records.is_empty());
        let d = od(1000);
        let r = Residual::new(&d, model, records).unwrap();
        for (f, &x) in intrinsic.iter().enumerate() {
            assert_eq!(r.closure_sample(f as u64, 0), x);
        }
    }

    #[test]
    fn sparse_residual_closes_model_gap() {
        // Constant model at 1000, but content is 1000 with two glitches.
        let intrinsic: Vec<i32> = (0..200)
            .map(|i| {
                if i == 10 {
                    5000
                } else if i == 42 {
                    -7000
                } else {
                    1000
                }
            })
            .collect();
        let model = ResidualModel::Constant(1000);
        let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
        assert_eq!(records.len(), 2);
        let d = od(200);
        let r = Residual::new(&d, model, records).unwrap();
        for (f, &x) in intrinsic.iter().enumerate() {
            assert_eq!(r.closure_sample(f as u64, 0), x);
        }
    }

    #[test]
    fn validation_rejects_bad_records() {
        let d = od(100);
        // Frame beyond extent.
        assert!(
            Residual::new(
                &d,
                ResidualModel::Zero,
                vec![ResidualRecord {
                    frame: 100,
                    channel: 0,
                    delta: 1
                }],
            )
            .is_none()
        );
        // Duplicate records.
        assert!(
            Residual::new(
                &d,
                ResidualModel::Zero,
                vec![
                    ResidualRecord {
                        frame: 3,
                        channel: 0,
                        delta: 1
                    },
                    ResidualRecord {
                        frame: 3,
                        channel: 0,
                        delta: 2
                    },
                ],
            )
            .is_none()
        );
        // Out-of-range channel.
        assert!(
            Residual::new(
                &d,
                ResidualModel::Zero,
                vec![ResidualRecord {
                    frame: 3,
                    channel: 2,
                    delta: 1
                }],
            )
            .is_none()
        );
        // Zero extent.
        let d0 = od(0);
        assert!(Residual::new(&d0, ResidualModel::Zero, vec![]).is_none());
    }

    #[test]
    fn uncloseable_gap_is_rejected() {
        let intrinsic = vec![i32::MIN; 8];
        // Model at i32::MAX cannot be closed by an i32 delta.
        let model = ResidualModel::Constant(i32::MAX);
        assert!(Residual::closing_residual(&intrinsic, 1, &model).is_none());
    }

    #[test]
    fn identity_is_content_bound() {
        let d = od(64);
        let a = Residual::new(&d, ResidualModel::Zero, vec![]).unwrap();
        let b = Residual::new(
            &d,
            ResidualModel::Zero,
            vec![ResidualRecord {
                frame: 1,
                channel: 0,
                delta: 5,
            }],
        )
        .unwrap();
        assert_ne!(Residual::content_id(&d, &a), Residual::content_id(&d, &b));
        // Model change alters identity.
        let c = Residual::new(&d, ResidualModel::Constant(7), vec![]).unwrap();
        assert_ne!(Residual::content_id(&d, &a), Residual::content_id(&d, &c));
    }
}

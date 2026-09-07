//! Event timeline assembly (host).
//!
//! The world is *functional*: voices are immutable spawn records derived from
//! a sorted event timeline; note-off and (future) automation are timeline
//! facts, not mutable runtime state. This is what makes
//! `chunked == contiguous` and `seek == sequential` true by construction and
//! keeps the GPU flattening trivial.

use crate::error::{Error, Result};
use crate::sampler::voice::VoiceSpec;
use crate::universe::event::{Event as OrderedEvent, EventClass};

/// A timeline event with its payload.
#[derive(Debug, Clone, PartialEq)]
pub enum TimelineEvent {
    /// Spawn a voice with the given spec (its trigger is `spec.trigger_frame`).
    VoiceOn(VoiceSpec),
    /// Note-off a previously spawned voice at `frame`.
    VoiceOff { frame: i64, voice: u32 },
}

/// One assembled voice entry: spawn spec plus resolved note-off (earliest
/// wins). `voice_id` is assigned by trigger order.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceEntry {
    pub voice_id: u32,
    pub spec: VoiceSpec,
    pub note_off: Option<i64>,
}

/// Assembled, validated score.
#[derive(Debug, Clone, Default)]
pub struct Score {
    pub voices: Vec<VoiceEntry>,
}

/// Event with ordering key (frame, class priority, sequence).
#[derive(Debug, Clone)]
struct KeyedEvent {
    frame: i64,
    seq: u64,
    class: EventClass,
    inner: TimelineEvent,
}

impl KeyedEvent {
    fn ord_key(&self) -> OrderedEvent {
        OrderedEvent::new(
            crate::universe::time::MediaFrame(self.frame),
            self.class,
            self.seq,
            0,
        )
    }
}

/// Build a validated `Score` from an unsorted event list.
///
/// Validation: voice specs are validated (their `validate`); note-offs must
/// reference voices triggered at or before the off frame; each voice may have
/// many note-offs (earliest wins); the concurrent-voice ceiling
/// (`MAX_ACTIVE_VOICES`) is checked across the whole timeline.
pub fn assemble(events: Vec<TimelineEvent>) -> Result<Score> {
    let mut keyed: Vec<KeyedEvent> = events
        .into_iter()
        .enumerate()
        .map(|(seq, inner)| {
            let (frame, class) = match &inner {
                TimelineEvent::VoiceOn(spec) => {
                    spec.validate()?;
                    (spec.trigger_frame, EventClass::Start)
                }
                TimelineEvent::VoiceOff { frame, .. } => (*frame, EventClass::Stop),
            };
            Ok(KeyedEvent {
                frame,
                seq: seq as u64,
                class,
                inner,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    keyed.sort_by_key(|e| e.ord_key());

    // Pass 1: assign voice ids to VoiceOn events in trigger order.
    let mut voice_ids = Vec::new();
    for e in &keyed {
        if let TimelineEvent::VoiceOn(_) = &e.inner {
            voice_ids.push(e.seq);
        }
    }
    let mut id_of_on_seq = std::collections::HashMap::new();
    for (id, seq) in voice_ids.iter().enumerate() {
        id_of_on_seq.insert(*seq, id as u32);
    }

    // Pass 2: apply note-offs (earliest per voice wins; multiple allowed).
    let mut offs: Vec<Option<i64>> = vec![None; voice_ids.len()];
    for e in &keyed {
        if let TimelineEvent::VoiceOff { frame, voice } = &e.inner {
            let idx = *voice as usize;
            if idx >= offs.len() {
                return Err(Error::malformed(format!(
                    "VoiceOff references unknown voice {voice}"
                )));
            }
            if offs[idx].is_none() {
                offs[idx] = Some(*frame);
            }
        }
    }

    // Pass 3: build entries, enforce ordering (note_off >= trigger).
    let mut voices = Vec::new();
    for (i, e) in keyed.iter().enumerate() {
        if let TimelineEvent::VoiceOn(spec) = &e.inner {
            let voice_id = id_of_on_seq[&e.seq];
            let note_off = offs[voice_id as usize];
            if note_off.is_some_and(|off| off < spec.trigger_frame) {
                return Err(Error::malformed(format!(
                    "note_off before trigger for voice {voice_id}"
                )));
            }
            voices.push(VoiceEntry {
                voice_id,
                spec: spec.clone(),
                note_off,
            });
        }
        let _ = i;
    }

    // Concurrent-voice ceiling: sweep events.
    let mut active = 0u32;
    for e in &keyed {
        match &e.inner {
            TimelineEvent::VoiceOn(_) => {
                active += 1;
                if active > crate::limits::MAX_ACTIVE_VOICES {
                    return Err(Error::limit(format!(
                        "concurrent voices exceed MAX_ACTIVE_VOICES at frame {}",
                        e.frame
                    )));
                }
            }
            TimelineEvent::VoiceOff { .. } => {
                active = active.saturating_sub(1);
            }
        }
    }

    Ok(Score { voices })
}

/// Convenience constructor for a single voice score (tests, small courts).
pub fn single_voice(spec: VoiceSpec) -> Result<Score> {
    assemble(vec![TimelineEvent::VoiceOn(spec)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::descriptor::{ObjectDescriptor, Representation};
    use crate::object::{Literal, ObjectId, ObjectStore};
    use crate::sampler::envelope::{ENV_UNITY, EnvelopeParams};
    use crate::universe::layout::Layout;

    fn spec(object: ObjectId, trigger: i64) -> VoiceSpec {
        VoiceSpec {
            object,
            trigger_frame: trigger,
            note_off: None,
            start_pos_q24: 0,
            rate_q24: 1 << 24,
            object_channel: 0,
            route: crate::sampler::pan::Route::Mono(0),
            gain_q16: 1 << 16,
            pan_q16: 0,
            envelope: EnvelopeParams::new(0, 0, ENV_UNITY, 0).unwrap(),
            loop_mode: crate::sampler::voice::LoopMode::Off,
            interp: crate::sampler::voice::Interp::Nearest,
        }
    }

    fn store() -> ObjectStore {
        let mut s = ObjectStore::new();
        let d = ObjectDescriptor::new(Representation::Literal, 1, Layout::Mono, None).unwrap();
        let _ = s.insert(
            d.clone(),
            crate::object::ObjectData::Literal(Literal::new(&d, vec![0]).unwrap()),
        );
        s
    }

    #[test]
    fn assembly_orders_and_applies_note_offs() {
        let s = store();
        let id = s.iter().next().unwrap().id;
        let score = assemble(vec![
            TimelineEvent::VoiceOn(spec(id, 100)),
            TimelineEvent::VoiceOn(spec(id, 50)),
            TimelineEvent::VoiceOff {
                frame: 200,
                voice: 1,
            },
        ])
        .unwrap();
        assert_eq!(score.voices.len(), 2);
        // Trigger order assigns voice ids: first VoiceOn (frame 50) = voice 0.
        assert_eq!(score.voices[0].spec.trigger_frame, 50);
        assert_eq!(score.voices[0].voice_id, 0);
        assert_eq!(score.voices[1].spec.trigger_frame, 100);
        assert_eq!(score.voices[1].voice_id, 1);
        assert_eq!(score.voices[0].note_off, None);
        assert_eq!(score.voices[1].note_off, Some(200));
    }

    #[test]
    fn rejects_unknown_and_pre_trigger_offs() {
        let s = store();
        let id = s.iter().next().unwrap().id;
        assert!(
            assemble(vec![
                TimelineEvent::VoiceOn(spec(id, 100)),
                TimelineEvent::VoiceOff {
                    frame: 50,
                    voice: 0
                },
            ])
            .is_err()
        );
        assert!(
            assemble(vec![TimelineEvent::VoiceOff {
                frame: 50,
                voice: 9
            }])
            .is_err()
        );
    }

    #[test]
    fn enforces_voice_ceiling() {
        let s = store();
        let id = s.iter().next().unwrap().id;
        let n = crate::limits::MAX_ACTIVE_VOICES + 1;
        let mut evs = Vec::new();
        for i in 0..n {
            evs.push(TimelineEvent::VoiceOn(spec(id, i64::from(i))));
        }
        assert!(assemble(evs).is_err());
    }
}

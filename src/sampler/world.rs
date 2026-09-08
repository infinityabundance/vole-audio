//! The sampler world: deterministic functional observation.
//!
//! A `World` binds a validated `Score` (voice timeline) and output layout.
//! Observation is a pure function of `(world, store, start_frame, frames)`:
//! voices are immutable records, so rendering is randomly accessible and
//! chunk-safe (`chunked == contiguous` by construction). All heavy sampler
//! state stays out of this module's hot path (it is in the store + voice
//! specs, exactly what later phases flatten to VRAM).

use crate::error::{Error, Result};
use crate::object::ObjectStore;
use crate::sampler::mix::Mixer;
use crate::sampler::pan::Route;
use crate::sampler::scheduler::{Score, TimelineEvent};
use crate::sampler::voice::{ResolvedVoice, VoiceSpec};

/// Sampler world (host).
#[derive(Debug, Clone)]
pub struct World {
    pub nominal_rate_hz: u32,
    pub output_channels: u8,
    score: Score,
}

impl World {
    /// Build a world from an event timeline.
    pub fn new(
        nominal_rate_hz: u32,
        output_channels: u8,
        events: Vec<TimelineEvent>,
    ) -> Result<World> {
        if !(crate::limits::MIN_SAMPLE_RATE_HZ..=crate::limits::MAX_SAMPLE_RATE_HZ)
            .contains(&nominal_rate_hz)
        {
            return Err(Error::limit("sample rate outside domain"));
        }
        if output_channels == 0 || output_channels as u32 > crate::limits::MAX_CHANNELS {
            return Err(Error::limit("output channels outside domain"));
        }
        let score = crate::sampler::scheduler::assemble(events)?;
        // Route validation against output layout.
        for v in &score.voices {
            match v.spec.route {
                Route::Mono(ch) if ch >= output_channels => {
                    return Err(Error::malformed(format!(
                        "voice route channel {ch} >= outputs {output_channels}"
                    )));
                }
                Route::StereoPair(base) if base as u32 + 1 >= u32::from(output_channels) => {
                    return Err(Error::malformed(format!(
                        "stereo route base {base} out of range for {output_channels} outputs"
                    )));
                }
                _ => {}
            }
        }
        Ok(World {
            nominal_rate_hz,
            output_channels,
            score,
        })
    }

    /// Build from a pre-assembled score.
    pub fn from_score(nominal_rate_hz: u32, output_channels: u8, score: Score) -> Result<World> {
        World::new(
            nominal_rate_hz,
            output_channels,
            score
                .voices
                .into_iter()
                .flat_map(|v| {
                    let mut evs = vec![TimelineEvent::VoiceOn(v.spec)];
                    if let Some(off) = v.note_off {
                        evs.push(TimelineEvent::VoiceOff {
                            frame: off,
                            voice: v.voice_id,
                        });
                    }
                    evs
                })
                .collect(),
        )
    }

    pub fn voices(&self) -> &[crate::sampler::scheduler::VoiceEntry] {
        &self.score.voices
    }

    /// Resolve every voice in this world against a store (single source of
    /// resolution; engines render from the result).
    pub fn resolve_all(&self, store: &ObjectStore) -> Result<Vec<ResolvedVoice>> {
        self.score
            .voices
            .iter()
            .map(|v| {
                let mut spec = v.spec.clone();
                spec.note_off = v.note_off;
                ResolvedVoice::resolve(store, spec, self.nominal_rate_hz)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Observe `[start, start+frames)` into canonical interleaved output.
    ///
    /// Voices triggered before the window contribute from their trigger; the
    /// result is identical whether the window is rendered in one call or
    /// split at any boundary (pure per-frame functions).
    pub fn observe(
        &self,
        store: &ObjectStore,
        start_frame: i64,
        frames: usize,
    ) -> Result<Vec<i32>> {
        if frames > crate::limits::MAX_QUANTUM_FRAMES as usize {
            return Err(Error::limit("observation window exceeds quantum ceiling"));
        }
        let resolved = self.resolve_all(store)?;

        let channels = usize::from(self.output_channels);
        let mut mixer = Mixer::new(channels, frames);
        // Skip voices whose entire life is outside the window.
        let end_frame = start_frame + frames as i64;
        for voice in &resolved {
            let spec = &voice.spec;
            if spec.trigger_frame >= end_frame {
                continue;
            }
            for fi in 0..frames {
                let t = start_frame + fi as i64;
                if t < spec.trigger_frame {
                    continue;
                }
                if voice.end_frame().is_some_and(|end| t >= end) {
                    continue;
                }
                if spec
                    .envelope
                    .silent_at(spec.trigger_frame, voice.spec.note_off, t)
                {
                    continue;
                }
                let c = voice.contribution_at(store, t);
                if let Some((ch, v)) = c.a {
                    mixer.add(fi, ch, v);
                }
                if let Some((ch, v)) = c.b {
                    mixer.add(fi, ch, v);
                }
            }
        }
        let mut out = vec![0i32; frames * channels];
        mixer.finalize_interleaved(&mut out);
        Ok(out)
    }
}

/// Convenience: build a world with a single voice spec.
pub fn world_with_one_voice(rate_hz: u32, channels: u8, spec: VoiceSpec) -> Result<World> {
    World::new(rate_hz, channels, vec![TimelineEvent::VoiceOn(spec)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::descriptor::{ObjectDescriptor, Representation};
    use crate::object::{Literal, ObjectData, ObjectId, ObjectStore};
    use crate::sampler::envelope::{ENV_UNITY, EnvelopeParams};
    use crate::sampler::procedural::Partial;
    use crate::sampler::voice::{Interp, LoopMode};
    use crate::universe::layout::Layout;

    fn ramp_store(len: usize, start: i32, step: i32) -> ObjectStore {
        let mut store = ObjectStore::new();
        let samples: Vec<i32> = (0..len).map(|i| start + step * i as i32).collect();
        let d =
            ObjectDescriptor::new(Representation::Literal, len as u64, Layout::Mono, None).unwrap();
        store
            .insert(
                d.clone(),
                ObjectData::Literal(Literal::new(&d, samples).unwrap()),
            )
            .unwrap();
        store
    }

    fn voice(id: crate::object::ObjectId, trigger: i64, rate: i64) -> VoiceSpec {
        VoiceSpec {
            object: id,
            trigger_frame: trigger,
            note_off: None,
            start_pos_q24: 0,
            rate_q24: rate,
            object_channel: 0,
            route: Route::Mono(0),
            gain_q16: 1 << 16,
            pan_q16: 0,
            envelope: EnvelopeParams::new(0, 0, ENV_UNITY, 0).unwrap(),
            loop_mode: LoopMode::Off,
            interp: Interp::Linear,
        }
    }

    #[test]
    fn observe_repeated_calls_are_identical() {
        let store = ramp_store(64, 100, 10);
        let id = store.iter().next().unwrap().id;
        let spec = voice(id, 10, 1 << 24);
        let world = world_with_one_voice(48_000, 1, spec).unwrap();
        let a = world.observe(&store, 0, 64).unwrap();
        let b = world.observe(&store, 0, 64).unwrap();
        assert_eq!(a, b);
        // Deterministic hash.
        let h1 = crate::universe::observation::observation_sha256(&a);
        let h2 = crate::universe::observation::observation_sha256(&b);
        assert_eq!(h1, h2);
    }

    #[test]
    fn chunked_equals_contiguous() {
        let store = ramp_store(256, 1000, 7);
        let id = store.iter().next().unwrap().id;
        // Multiple voices with offsets/releases for a harder case.
        let mut evs = vec![];
        for i in 0..4 {
            let mut s = voice(id, 5 + i * 20, (1 << 24) / 2); // half speed
            s.gain_q16 = (1 << 16) / (i + 1) as i32;
            evs.push(TimelineEvent::VoiceOn(s));
        }
        evs.push(TimelineEvent::VoiceOff {
            frame: 90,
            voice: 1,
        });
        let world = World::new(48_000, 1, evs).unwrap();
        let contiguous = world.observe(&store, 0, 200).unwrap();
        let mut joined = Vec::new();
        for chunk in [0..80usize, 80..140, 140..200] {
            let c = world
                .observe(&store, chunk.start as i64, chunk.len())
                .unwrap();
            joined.extend_from_slice(&c);
        }
        assert_eq!(joined.len(), contiguous.len());
        assert_eq!(joined, contiguous, "chunked must equal contiguous");
    }

    #[test]
    fn seek_equals_sequential() {
        let store = ramp_store(512, -5000, 13);
        let id = store.iter().next().unwrap().id;
        let spec = voice(id, 0, 1 << 24);
        let world = world_with_one_voice(48_000, 1, spec).unwrap();
        // Sequential playback from 0 in two calls equals one call from 0.
        let seq = world.observe(&store, 0, 256).unwrap();
        // Seek: observing a window whose voices began earlier but only this
        // window is requested; equal to slicing the contiguous result.
        let contig = world.observe(&store, 0, 512).unwrap();
        let seeked = world.observe(&store, 100, 256).unwrap();
        assert_eq!(&contig[100..356], &seeked[..]);
        let _ = seq;
    }

    /// Regression: endless procedural voices (constant/oscillator/noise/
    /// partial bank) must be *audible* — the world's natural-end pre-check
    /// used to treat extent-0 sources as already ended, silencing every
    /// endless class (Phase D latent bug surfaced during Phase F work). Each
    /// class must produce a distinct nonzero observation.
    #[test]
    fn endless_procedural_voices_are_audible_and_class_distinct() {
        let mut store = ObjectStore::new();
        let mut mk_obj = |rep, data| {
            let d = ObjectDescriptor::new(rep, 0, Layout::Mono, None).unwrap();
            store.insert(d.clone(), data).unwrap()
        };
        let cst = mk_obj(
            Representation::Constant,
            ObjectData::Constant(crate::object::Constant::new(65_536)),
        );
        let osc = mk_obj(
            Representation::Oscillator,
            ObjectData::Oscillator(crate::object::Oscillator::checked(440, 1 << 16).unwrap()),
        );
        let nse = mk_obj(
            Representation::Noise,
            ObjectData::Noise(crate::object::Noise::new(0x5EED_2026)),
        );
        let bank = mk_obj(
            Representation::PartialBank,
            ObjectData::PartialBank(
                crate::object::PartialBank::checked(
                    55,
                    vec![
                        Partial {
                            harmonic: 1,
                            amp_q16: 1 << 15,
                        },
                        Partial {
                            harmonic: 3,
                            amp_q16: 1 << 14,
                        },
                        Partial {
                            harmonic: 5,
                            amp_q16: 1 << 13,
                        },
                    ],
                )
                .unwrap(),
            ),
        );

        let mut class_hash = std::collections::BTreeMap::new();
        for id in [cst, osc, nse, bank] {
            // Include windows that start at the trigger (the old bug ended the
            // voice at exactly t0) and later windows (sustain must persist).
            for start in [0i64, 333, 5000] {
                let world = world_with_one_voice(48_000, 1, voice(id, 0, 1 << 24)).unwrap();
                let out = world.observe(&store, start, 512).unwrap();
                assert!(
                    out.iter().any(|&x| x != 0),
                    "object {id:?} silent in window [{start}, {})",
                    start + 512
                );
                let h = crate::hash::sha256::hex(
                    &crate::universe::observation::observation_sha256(&out),
                );
                class_hash.insert(id, h);
            }
        }
        // Every class must have a distinct observation fingerprint (none of
        // them collapses to silence or to another class's signal).
        let mut seen = std::collections::BTreeSet::new();
        for (id, h) in &class_hash {
            assert!(
                seen.insert(h.clone()),
                "object {id:?} observation collides with another class"
            );
        }
        assert_eq!(class_hash.len(), 4);
        // Sustain: an endless voice must still be audible deep after trigger.
        let world = world_with_one_voice(48_000, 1, voice(cst, 0, 1 << 24)).unwrap();
        let out = world.observe(&store, 100_000, 64).unwrap();
        assert!(out.iter().any(|&x| x != 0), "constant must sustain");
    }

    #[test]
    fn procedural_objects_observe_without_resident_pcm() {
        // Build one of every procedural class; verify (a) zero resident sample
        // bytes, (b) deterministic repeated observation, (c) chunk equality.
        let mut store = ObjectStore::new();
        let ed = ObjectDescriptor::new(Representation::Oscillator, 0, Layout::Mono, None).unwrap();
        let osc = store
            .insert(
                ed.clone(),
                ObjectData::Oscillator(crate::object::Oscillator::checked(440, 1 << 16).unwrap()),
            )
            .unwrap();
        let cd = ObjectDescriptor::new(Representation::Constant, 0, Layout::Mono, None).unwrap();
        let cst = store
            .insert(
                cd.clone(),
                ObjectData::Constant(crate::object::Constant::new(123_456)),
            )
            .unwrap();
        let nd = ObjectDescriptor::new(Representation::Noise, 0, Layout::Mono, None).unwrap();
        let nse = store
            .insert(
                nd.clone(),
                ObjectData::Noise(crate::object::Noise::new(0xABCD)),
            )
            .unwrap();
        let bd = ObjectDescriptor::new(Representation::PartialBank, 0, Layout::Mono, None).unwrap();
        let bank = store
            .insert(
                bd.clone(),
                ObjectData::PartialBank(
                    crate::object::PartialBank::checked(
                        110,
                        vec![
                            Partial {
                                harmonic: 1,
                                amp_q16: 1 << 15,
                            },
                            Partial {
                                harmonic: 2,
                                amp_q16: 1 << 14,
                            },
                            Partial {
                                harmonic: 3,
                                amp_q16: 1 << 14,
                            },
                            Partial {
                                harmonic: 4,
                                amp_q16: 1 << 12,
                            },
                        ],
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
        let sd = ObjectDescriptor::new(Representation::Silence, 0, Layout::Mono, None).unwrap();
        let sil = store.insert(sd.clone(), ObjectData::Silence).unwrap();

        // (a) No resident sample bytes for endless classes.
        for id in [osc, cst, nse, bank, sil] {
            let o = store.get(id).unwrap();
            assert_eq!(o.data.resident_sample_bytes(), 0, "object {id}");
        }

        // (b)+(c) deterministic and chunk-safe through the world.
        let instant = EnvelopeParams::new(0, 0, ENV_UNITY, 0).unwrap();
        let mk = |object: ObjectId, trigger: i64| -> VoiceSpec {
            VoiceSpec {
                object,
                trigger_frame: trigger,
                note_off: None,
                start_pos_q24: 0,
                rate_q24: 1 << 24,
                object_channel: 0,
                route: Route::Mono(0),
                gain_q16: 1 << 15,
                pan_q16: 0,
                envelope: instant,
                loop_mode: LoopMode::Off,
                interp: Interp::Linear,
            }
        };
        let mut evs = vec![
            TimelineEvent::VoiceOn(mk(osc, 0)),
            TimelineEvent::VoiceOn(mk(cst, 0)),
            TimelineEvent::VoiceOn(mk(nse, 16)),
            TimelineEvent::VoiceOn(mk(bank, 32)),
            TimelineEvent::VoiceOn(mk(sil, 0)),
        ];
        // A wavetable (cycle) voice as well.
        let cyc: Vec<i32> = (0..64)
            .map(|i| ((i as i64 - 32) * (1 << 22)) as i32)
            .collect();
        let wd = ObjectDescriptor::new(Representation::Wavetable, 64, Layout::Mono, None).unwrap();
        let wt = store
            .insert(
                wd.clone(),
                ObjectData::Wavetable(crate::object::Cycle::new(&wd, cyc).unwrap()),
            )
            .unwrap();
        evs.push(TimelineEvent::VoiceOn(mk(wt, 8)));

        let world = World::new(48_000, 1, evs).unwrap();
        let a = world.observe(&store, 0, 2048).unwrap();
        let b = world.observe(&store, 0, 2048).unwrap();
        assert_eq!(a, b);
        let mut joined = Vec::new();
        for (s, l) in [(0usize, 777usize), (777, 900), (1677, 371)] {
            joined.extend_from_slice(&world.observe(&store, s as i64, l).unwrap());
        }
        assert_eq!(joined, a);
        // Wavetable cycles repeat: positions 8..72 in the world correspond to
        // cycle indices 0..64 repeated; verify the first 64 samples after
        // trigger equal the cycle (mono route, gain applied: half => cycle/2).
        let world2 = World::new(48_000, 1, vec![TimelineEvent::VoiceOn(mk(wt, 8))]).unwrap();
        let seg = world2.observe(&store, 8, 64).unwrap();
        let expected: Vec<i32> = (0..64)
            .map(|i| crate::universe::arithmetic::sat_i32((i as i64 - 32) * (1 << 22) / 2))
            .collect();
        assert_eq!(seg, expected, "cycle content repeats exactly at unity rate");
    }

    #[test]
    fn residual_closure_equals_literal_through_the_world() {
        // Phase E end-to-end closure proof: sampled-origin content ingested
        // from WAV as a literal, and the same content closed by a
        // model+sparse-residual object, must observe identically through the
        // sampler (closure precedes observation).
        use crate::object::{Residual, ResidualModel};
        use crate::sampler::voice::Interp;

        // Synthesize a 40-frame periodic triangle with a glitch at frame 200.
        let intrinsic: Vec<i32> = (0..1200)
            .map(|i| {
                let m = i % 40;
                let v = if m < 20 { m } else { 40 - m };
                let mut s = v << 20;
                if i == 200 {
                    s = 777_000;
                }
                s
            })
            .collect();

        let mut store = ObjectStore::new();
        // Literal from canonical codes (the WAV parser maps into exactly this
        // domain; see format::wav tests for the byte-level roundtrip).
        let ld = ObjectDescriptor::new(Representation::Literal, 1200, Layout::Mono, None).unwrap();
        let lit_id = store
            .insert(
                ld.clone(),
                ObjectData::Literal(Literal::new(&ld, intrinsic.clone()).unwrap()),
            )
            .unwrap();
        // Residual-governed twin: periodic-40 model + sparse residual.
        let model = ResidualModel::Periodic {
            cycle: intrinsic[..40].to_vec(),
        };
        let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
        assert_eq!(records.len(), 1, "only the glitch needs a residual");
        let rd = ObjectDescriptor::new(Representation::PredictorResidual, 1200, Layout::Mono, None)
            .unwrap();
        let res_id = store
            .insert(
                rd.clone(),
                ObjectData::PredictorResidual(Residual::new(&rd, model, records).unwrap()),
            )
            .unwrap();

        let instant = EnvelopeParams::new(0, 0, ENV_UNITY, 0).unwrap();
        let voice = |object: ObjectId| VoiceSpec {
            object,
            trigger_frame: 0,
            note_off: None,
            start_pos_q24: 0,
            rate_q24: 1 << 24,
            object_channel: 0,
            route: Route::Mono(0),
            gain_q16: 1 << 16,
            pan_q16: 0,
            envelope: instant,
            loop_mode: LoopMode::Off,
            interp: Interp::Linear,
        };
        let wl = World::new(48_000, 1, vec![TimelineEvent::VoiceOn(voice(lit_id))]).unwrap();
        let wr = World::new(48_000, 1, vec![TimelineEvent::VoiceOn(voice(res_id))]).unwrap();
        let a = wl.observe(&store, 0, 1200).unwrap();
        let b = wr.observe(&store, 0, 1200).unwrap();
        assert_eq!(a, b, "residual closure must equal literal observation");
        // Linear interpolation between frames also matches (readers agree on
        // reconstructed neighbors).
        let al = wl.observe(&store, 0, 2000).unwrap();
        let bl = wr.observe(&store, 0, 2000).unwrap();
        assert_eq!(al, bl, "interpolated closure must match literal");
    }
}

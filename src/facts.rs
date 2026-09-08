//! Semantic facts — independent-oracle coverage (Phase G prerequisite).
//!
//! Differential parity (`scalar == SIMD == CUDA == ROCm`) proves *backend*
//! equality, but a bug shared by every backend survives it. Each fact below is
//! therefore an **independent semantic statement** about `vole.audio.u1`: its
//! expected samples are derived from first principles (closed-form integer
//! math, hand-enumerated boundary sequences, or vectors produced by an
//! independent implementation), never by calling the sampler/evaluator code
//! under test, and then observed through the full `World` observation path on
//! every available host surface.
//!
//! Coverage matrix (docs/SEMANTIC_FACTS.md) — every representation and
//! transform that ships from this phase on carries at least one fact row:
//!
//! ```text
//!                    scalar   simd/floors   cuda*   rocm*
//! semantic fact        X          X          —       —      (*: backend arrives
//! reference hash       X          X          —       —       with its phase;
//! random differential  —          X          —       —       rows stay visible)
//! ```
//!
//! Fact ids are stable (`F01`…); adding a representation requires adding a
//! fact (see docs/PROJECT_STATE.md, Phase G prerequisites).

use crate::error::Result;
use crate::eval::backend::Isa;
use crate::eval::{ScalarOracle, SimdOracle};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::residual::{Residual, ResidualModel};
use crate::object::{
    Constant, Cycle, Literal, LoopRegion, Noise, ObjectData, ObjectId, ObjectStore, Oscillator,
};
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::pan::Route;
use crate::sampler::scheduler::TimelineEvent;
use crate::sampler::voice::{Interp, LoopMode, VoiceSpec};
use crate::sampler::world::World;
use crate::universe::layout::Layout;

const RATE: u32 = 48_000;
const ENV_U: i32 = 1 << 16;

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

/// Host execution surface a world fact is verified against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Scalar oracle (semantic authority).
    Scalar,
    /// SIMD engine on a concrete ISA floor.
    Simd(Isa),
}

impl Surface {
    pub fn label(self) -> String {
        match self {
            Surface::Scalar => "scalar".to_string(),
            Surface::Simd(isa) => format!("simd/{}", isa.label()),
        }
    }
}

/// All host surfaces: the scalar authority, the SIMD engine's scalar floor,
/// and every runtime-available vector floor.
pub fn host_surfaces() -> Vec<Surface> {
    let mut v = vec![Surface::Scalar, Surface::Simd(Isa::Scalar)];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            v.push(Surface::Simd(Isa::Avx2));
        }
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
        {
            v.push(Surface::Simd(Isa::Avx512));
        }
    }
    v
}

fn observe_surface(
    store: &ObjectStore,
    world: &World,
    start: i64,
    frames: usize,
    s: Surface,
) -> Result<Vec<i32>> {
    match s {
        Surface::Scalar => ScalarOracle::new(world.clone()).observe(store, start, frames),
        Surface::Simd(isa) => SimdOracle {
            world: world.clone(),
            isa,
        }
        .observe(store, start, frames),
    }
}

// ---------------------------------------------------------------------------
// Fact model
// ---------------------------------------------------------------------------

/// A world-plus-store the fact observes.
pub struct FactWorld {
    pub store: ObjectStore,
    pub world: World,
}

/// One fact.
pub struct Fact {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    /// The independent expectation: either observation windows over a world
    /// (verified on every host surface) or an authority-level statement that
    /// needs no world (surface-independent arithmetic/validation facts).
    pub expectation: Expectation,
}

pub enum Expectation {
    /// Windows over a world with expected buffers computed from first
    /// principles. Verified on every host surface.
    Windows {
        world: FactWorld,
        windows: Vec<(i64, usize)>,
        /// One expected interleaved buffer per window, same order.
        expected: Vec<Vec<i32>>,
    },
    /// Authority-level check (surface-independent; e.g. resolution-time
    /// arithmetic or validation).
    Authority { check: fn() -> Result<String> },
}

/// Result of running one fact on one surface.
#[derive(Debug, Clone)]
pub struct FactSurfaceResult {
    pub surface: String,
    pub passed: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FactResult {
    pub fact_id: &'static str,
    pub fact_name: &'static str,
    pub rows: Vec<FactSurfaceResult>,
}

/// Run every fact on every applicable host surface.
pub fn run_facts() -> Result<Vec<FactResult>> {
    let mut out = Vec::new();
    for fact in registry() {
        let mut rows = Vec::new();
        match &fact.expectation {
            Expectation::Windows {
                world,
                windows,
                expected,
            } => {
                if windows.len() != expected.len() {
                    return Err(crate::error::Error::malformed(format!(
                        "{}: windows/expected length mismatch",
                        fact.id
                    )));
                }
                for surface in host_surfaces() {
                    let mut passed = true;
                    let mut detail: Option<String> = None;
                    for (k, &(start, frames)) in windows.iter().enumerate() {
                        let observed = match observe_surface(
                            &world.store,
                            &world.world,
                            start,
                            frames,
                            surface,
                        ) {
                            Ok(o) => o,
                            Err(e) => {
                                passed = false;
                                detail = Some(format!("observe error: {e}"));
                                break;
                            }
                        };
                        if observed != expected[k] {
                            passed = false;
                            let idx = observed
                                .iter()
                                .zip(expected[k].iter())
                                .position(|(a, b)| a != b)
                                .unwrap_or(0);
                            detail = Some(format!(
                                "window [{start}, {}) differs at element {} (expected {}, got {})",
                                start + frames as i64,
                                idx,
                                expected[k][idx],
                                observed[idx],
                            ));
                            break;
                        }
                    }
                    rows.push(FactSurfaceResult {
                        surface: surface.label(),
                        passed,
                        detail,
                    });
                }
            }
            Expectation::Authority { check } => {
                // An authority fact runs once (no world/surface); a failed
                // check is a failing row, not an abort — the court can then
                // emit an honest FAILED_CORRECTNESS receipt.
                match (check)() {
                    Ok(detail) => rows.push(FactSurfaceResult {
                        surface: "authority".to_string(),
                        passed: true,
                        detail: Some(detail),
                    }),
                    Err(e) => rows.push(FactSurfaceResult {
                        surface: "authority".to_string(),
                        passed: false,
                        detail: Some(format!("{e}")),
                    }),
                }
            }
        }
        out.push(FactResult {
            fact_id: fact.id,
            fact_name: fact.name,
            rows,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Fact builders (independent expected math lives here, not in the evaluator)
// ---------------------------------------------------------------------------

fn env_instant() -> EnvelopeParams {
    EnvelopeParams::new(0, 0, ENV_U, 0).unwrap()
}

/// Base mono voice spec: unity gain/pan, instant envelope, linear interp.
fn voice_mono(object: ObjectId, trigger: i64) -> VoiceSpec {
    VoiceSpec {
        object,
        trigger_frame: trigger,
        note_off: None,
        start_pos_q24: 0,
        rate_q24: 1 << 24,
        object_channel: 0,
        route: Route::Mono(0),
        gain_q16: 1 << 16,
        pan_q16: 0,
        envelope: env_instant(),
        loop_mode: LoopMode::Off,
        interp: Interp::Linear,
    }
}

/// Constant (all-ones) expectation of `len` samples.
fn ones(len: usize, value: i32) -> Vec<i32> {
    vec![value; len]
}

fn insert(
    store: &mut ObjectStore,
    rep: Representation,
    extent: u64,
    layout: Layout,
    data: ObjectData,
) -> ObjectId {
    let d = ObjectDescriptor::new(rep, extent, layout, None).unwrap();
    store.insert(d.clone(), data).unwrap()
}

/// Register every fact (stable ids F01…). Add a fact when adding a
/// representation or transform — see docs/SEMANTIC_FACTS.md.
pub fn registry() -> Vec<Fact> {
    vec![
        Fact {
            id: "F01",
            name: "silence",
            description: "Silence objects observe as exact zeros over any window (liveness baseline).",
            expectation: Expectation::Windows {
                world: silence_world(),
                windows: vec![(0, 37), (5000, 64)],
                expected: vec![ones(37, 0), ones(64, 0)],
            },
        },
        Fact {
            id: "F02",
            name: "constant-unity-identity",
            description: "A Constant voice at unity gain/env observes exactly its level (identity chain).",
            expectation: Expectation::Windows {
                world: constant_world(1_234_567, 1 << 16, None, 0),
                windows: vec![(100, 16)],
                expected: vec![ones(16, 1_234_567)],
            },
        },
        Fact {
            id: "F03",
            name: "constant-half-gain",
            description: "Half gain scales a Constant exactly with round-half-away: 1234567 -> 617284.",
            expectation: Expectation::Windows {
                world: constant_world(1_234_567, 1 << 15, None, 0),
                windows: vec![(0, 8)],
                expected: vec![ones(8, 617_284)],
            },
        },
        Fact {
            id: "F04",
            name: "oscillator-exact-phase-points",
            description: "Oscillator at 12000 Hz / 48000 advances exactly a quarter turn per frame: \
                 phase-zero (0), table peak (+(2^31-2)), sin(pi) (0), negative peak, period 4.",
            expectation: Expectation::Windows {
                world: osc_quarter_world(),
                windows: vec![(0, 8), (100, 12)],
                expected: vec![
                    quarter_seq(8),
                    quarter_seq(12), // phase is t-independent of trigger offset
                ],
            },
        },
        Fact {
            id: "F05",
            name: "noise-frozen-vectors",
            description: "VOLE-SPLITMIX64-STREAM frame vectors for one seed, produced by an independent \
                 Python implementation and frozen here; observed through the world unchanged.",
            expectation: Expectation::Windows {
                world: noise_world(),
                windows: vec![(0, 16)],
                expected: vec![NOISE_VECTORS_16.to_vec()],
            },
        },
        Fact {
            id: "F06",
            name: "wavetable-fractional-interpolation",
            description: "Cycle read at rate 1.5 frames/frame: hand-computed linear-interpolation \
                 sequence (half-fraction rounding is exact on even spans).",
            expectation: Expectation::Windows {
                world: cycle_world(vec![0, 8000, 16000, 0], 3 << 23),
                windows: vec![(0, 16)],
                expected: vec![half_rate_seq_16()],
            },
        },
        Fact {
            id: "F07",
            name: "exact-repeat-periodicity",
            description: "Cycle content at unity rate observes X[t] == pattern[t mod P] (independent \
                 modulo oracle, 40 frames).",
            expectation: Expectation::Windows {
                world: cycle_world(vec![7, 13, 29, 3], 1 << 24),
                windows: vec![(0, 40)],
                expected: vec![(0..40).map(|t| [7, 13, 29, 3][t % 4]).collect()],
            },
        },
        Fact {
            id: "F08",
            name: "reverse-one-shot-sequence",
            description: "Reverse playback (rate -1, start at the last frame) reproduces the content \
                 backwards then silences exactly at the natural end frame.",
            expectation: Expectation::Windows {
                world: literal_world(vec![10, 20, 30, 40, 50], -(1 << 24), 4 << 24),
                windows: vec![(0, 10)],
                expected: vec![vec![50, 40, 30, 20, 10, 0, 0, 0, 0, 0]],
            },
        },
        Fact {
            id: "F09",
            name: "loop-boundary-enumeration",
            description: "Loop [10, 20) entered at frame 19: hand-enumerated boundary frames incl. the \
                 wrap 19 -> 10 -> 11 … 19 -> 10.",
            expectation: Expectation::Windows {
                world: literal_loop_world(0..=19i64, LoopRegion::new(10, 20).unwrap()),
                windows: vec![(0, 12)],
                expected: vec![vec![19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10]],
            },
        },
        Fact {
            id: "F10",
            name: "pan-endpoints-and-center",
            description: "Pan endpoints route analytically: hard left -> (level, 0), hard right -> \
                 (0, level), center -> equal-gain halves (exact on even level).",
            expectation: Expectation::Windows {
                world: pan_world(),
                windows: vec![(0, 4), (100, 4), (200, 4)],
                expected: vec![
                    vec![123_456, 0, 123_456, 0, 123_456, 0, 123_456, 0],
                    vec![61_728; 8],
                    vec![0, 123_456, 0, 123_456, 0, 123_456, 0, 123_456],
                ],
            },
        },
        Fact {
            id: "F11",
            name: "envelope-exact-knots",
            description: "Constant 2^16 through an A=4/D=4/S=U/2/R=4 envelope observes exactly the \
                 envelope levels (hand-computed knots incl. release from sustain).",
            expectation: Expectation::Windows {
                world: envelope_knot_world(),
                windows: vec![(0, 16)],
                expected: vec![vec![
                    0, 16_384, 32_768, 49_152, 65_536, 57_344, 49_152, 40_960, 32_768, 32_768,
                    32_768, 32_768, 32_768, 24_576, 16_384, 8_192,
                ]],
            },
        },
        Fact {
            id: "F12",
            name: "mix-sum-and-saturation-bound",
            description: "Mix is an ordinary i64 sum saturated exactly once at the output boundary: \
                 4 x 2^30 voices -> i32::MAX; 2 x +2^30 + 2 x -2^30 -> 0; a 2x-gain single \
                 voice saturates at the voice bus. The first quartet is note-offed at frame \
                 50 (instant release) so the later windows isolate the later groups.",
            expectation: Expectation::Windows {
                world: mix_world(),
                windows: vec![(0, 5), (100, 5), (200, 5)],
                expected: vec![ones(5, i32::MAX), ones(5, 0), vec![i32::MAX; 5]],
            },
        },
        Fact {
            id: "F13",
            name: "residual-closure-equals-intrinsic",
            description: "A residual-governed object (zero model + closing records over a formulaic \
                 intrinsic) observes exactly the intrinsic sequence — closure H+R == X_O.",
            expectation: Expectation::Windows {
                world: residual_world(),
                windows: vec![(0, 64)],
                expected: vec![(0..64).map(intrinsic_sample).collect()],
            },
        },
        Fact {
            id: "F14",
            name: "reference-unity-transpose",
            description: "A Referenced object at unity transpose observes exactly its target's content \
                 (formulaic intrinsic; reference resolution + transpose compose is exact).",
            expectation: Expectation::Windows {
                world: reference_world(),
                windows: vec![(0, 64)],
                expected: vec![(0..64).map(|i| i << 20).collect()],
            },
        },
        Fact {
            id: "F15",
            name: "transpose-wide-integer-oracle",
            description: "compose_transpose matches an independent i128 formula on every vector and \
                 saturates at its documented ceiling; a composed rate beyond the frozen rate \
                 domain is rejected at voice resolution (never wrapped into valid-looking \
                 garbage).",
            expectation: Expectation::Authority {
                check: transpose_oracle,
            },
        },
    ]
}

// -- world builders ---------------------------------------------------------

fn silence_world() -> FactWorld {
    let mut store = ObjectStore::new();
    let id = insert(
        &mut store,
        Representation::Silence,
        0,
        Layout::Mono,
        ObjectData::Silence,
    );
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(voice_mono(id, 0))]).unwrap();
    FactWorld { store, world }
}

fn constant_world(level: i32, gain: i32, note_off: Option<i64>, trigger: i64) -> FactWorld {
    let mut store = ObjectStore::new();
    let id = insert(
        &mut store,
        Representation::Constant,
        0,
        Layout::Mono,
        ObjectData::Constant(Constant::new(level)),
    );
    let mut spec = voice_mono(id, trigger);
    spec.gain_q16 = gain;
    spec.note_off = note_off;
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(spec)]).unwrap();
    FactWorld { store, world }
}

fn osc_quarter_world() -> FactWorld {
    let mut store = ObjectStore::new();
    let id = insert(
        &mut store,
        Representation::Oscillator,
        0,
        Layout::Mono,
        ObjectData::Oscillator(Oscillator::checked(12_000, 1 << 16).unwrap()),
    );
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(voice_mono(id, 0))]).unwrap();
    FactWorld { store, world }
}

/// Independent quarter-turn sequence: phase = t * 2^62 mod 2^64 lands exactly
/// on table indexes 0/1024/2048/3072 with zero fraction.
fn quarter_seq(len: usize) -> Vec<i32> {
    let peak = (1i64 << 31) - 2; // +(2^30 - 1) * 2 at unity Q16 amplitude
    (0..len)
        .map(|t| match t % 4 {
            0 | 2 => 0,
            1 => peak as i32,
            _ => -peak as i32,
        })
        .collect()
}

fn noise_world() -> FactWorld {
    let mut store = ObjectStore::new();
    let id = insert(
        &mut store,
        Representation::Noise,
        0,
        Layout::Mono,
        ObjectData::Noise(Noise::new(0x0BAD_5EED_2026_0D1A)),
    );
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(voice_mono(id, 0))]).unwrap();
    FactWorld { store, world }
}

/// Frozen frame vectors for `noise_world`'s seed, produced by an independent
/// Python implementation of VOLE-SPLITMIX64-STREAM (see the F05 comment);
/// frames 0..16 at unity gain/env observe the noise samples unchanged.
const NOISE_VECTORS_16: [i32; 16] = [
    -1_387_741_431,
    -1_360_973_437,
    111_364_597,
    217_214_590,
    1_968_934_142,
    -1_960_602_211,
    -526_340_721,
    -1_805_038_026,
    -1_640_702_009,
    2_096_043_865,
    285_603_315,
    -1_031_174_962,
    114_528_067,
    -1_759_495_177,
    728_987_141,
    -1_979_745_157,
];

fn cycle_world(samples: Vec<i32>, rate: i64) -> FactWorld {
    let mut store = ObjectStore::new();
    let len = samples.len() as u64;
    let d = ObjectDescriptor::new(Representation::Wavetable, len, Layout::Mono, None).unwrap();
    let id = store
        .insert(
            d.clone(),
            ObjectData::Wavetable(Cycle::new(&d, samples).unwrap()),
        )
        .unwrap();
    let mut spec = voice_mono(id, 0);
    spec.rate_q24 = rate;
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(spec)]).unwrap();
    FactWorld { store, world }
}

/// Independent half-rate sequence over the cycle [0,8000,16000,0] at 1.5
/// frames/frame: positions 0,1.5,3,0.5,2,3.5,1,2.5 mod 4 (period 8).
fn half_rate_seq_16() -> Vec<i32> {
    const PAT: [i32; 8] = [0, 12_000, 0, 4_000, 16_000, 0, 8_000, 8_000];
    (0..16).map(|t| PAT[t % 8]).collect()
}

fn literal_world(samples: Vec<i32>, rate: i64, start_pos_q24: i64) -> FactWorld {
    let mut store = ObjectStore::new();
    let len = samples.len() as u64;
    let d = ObjectDescriptor::new(Representation::Literal, len, Layout::Mono, None).unwrap();
    let id = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, samples).unwrap()),
        )
        .unwrap();
    let mut spec = voice_mono(id, 0);
    spec.rate_q24 = rate;
    spec.start_pos_q24 = start_pos_q24;
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(spec)]).unwrap();
    FactWorld { store, world }
}

/// Content whose sample at index i equals i (via `range`), looped over the
/// given region, entered at the region's last frame.
fn literal_loop_world(range: std::ops::RangeInclusive<i64>, region: LoopRegion) -> FactWorld {
    let samples: Vec<i32> = range.map(|i| i as i32).collect();
    let mut store = ObjectStore::new();
    let len = samples.len() as u64;
    let d = ObjectDescriptor::new(Representation::Literal, len, Layout::Mono, None).unwrap();
    let id = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, samples).unwrap()),
        )
        .unwrap();
    let mut spec = voice_mono(id, 0);
    spec.start_pos_q24 = ((region.end_frame - 1) << 24) as i64; // enter at the last region frame
    spec.loop_mode = LoopMode::Region(region);
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(spec)]).unwrap();
    FactWorld { store, world }
}

fn pan_world() -> FactWorld {
    // Stereo object with constant level 123456 on both channels.
    let level = 123_456i32;
    let mut store = ObjectStore::new();
    let samples = vec![level; 8 * 2];
    let d = ObjectDescriptor::new(Representation::Literal, 8, Layout::Stereo, None).unwrap();
    let id = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, samples).unwrap()),
        )
        .unwrap();
    let mut events = Vec::new();
    // hard left (frame 0), center (frame 100), hard right (frame 200).
    for (trigger, pan) in [(0i64, -(1 << 16)), (100, 0), (200, 1 << 16)] {
        let mut spec = voice_mono(id, trigger);
        spec.route = Route::StereoPair(0);
        spec.pan_q16 = pan;
        events.push(TimelineEvent::VoiceOn(spec));
    }
    let world = World::new(RATE, 2, events).unwrap();
    FactWorld { store, world }
}

fn envelope_knot_world() -> FactWorld {
    let mut store = ObjectStore::new();
    let id = insert(
        &mut store,
        Representation::Constant,
        0,
        Layout::Mono,
        ObjectData::Constant(Constant::new(65_536)),
    );
    let mut spec = voice_mono(id, 0);
    spec.envelope = EnvelopeParams::new(4, 4, ENV_U / 2, 4).unwrap();
    // Note-off reaches the voice only through a timeline event (the world
    // scheduler never reads VoiceSpec.note_off). Single voice -> id 0.
    let world = World::new(
        RATE,
        1,
        vec![
            TimelineEvent::VoiceOn(spec),
            TimelineEvent::VoiceOff {
                frame: 12,
                voice: 0,
            },
        ],
    )
    .unwrap();
    FactWorld { store, world }
}

fn mix_world() -> FactWorld {
    let mut store = ObjectStore::new();
    let pos = insert(
        &mut store,
        Representation::Constant,
        0,
        Layout::Mono,
        ObjectData::Constant(Constant::new(1 << 30)),
    );
    let neg = insert(
        &mut store,
        Representation::Constant,
        0,
        Layout::Mono,
        ObjectData::Constant(Constant::new(-(1 << 30))),
    );
    let mut events = Vec::new();
    // (a) four +2^30 voices -> sums to 2^32 -> saturates to i32::MAX.
    // Voice ids are assigned in trigger order: the quartet triggered at frame
    // 0 gets ids 0..=3; note them all off at frame 50 (instant release), so
    // the later windows isolate the later groups.
    for _ in 0..4 {
        events.push(TimelineEvent::VoiceOn(voice_mono(pos, 0)));
    }
    for voice in 0..4 {
        events.push(TimelineEvent::VoiceOff { frame: 50, voice });
    }
    // (b) two of each sign -> exactly 0 (endless, zero-sum for all time).
    for id in [pos, pos, neg, neg] {
        events.push(TimelineEvent::VoiceOn(voice_mono(id, 100)));
    }
    // (c) a single 2x-gain voice saturates at the voice bus (i32::MAX).
    let mut spec = voice_mono(pos, 200);
    spec.gain_q16 = 1 << 17;
    events.push(TimelineEvent::VoiceOn(spec));
    let world = World::new(RATE, 1, events).unwrap();
    FactWorld { store, world }
}

/// Formulaic intrinsic for the residual fact (deterministic, full-range-safe).
fn intrinsic_sample(i: i64) -> i32 {
    (((i * 1_000_003 + 12_345) % (1 << 31)) - (1 << 30)) as i32
}

fn residual_world() -> FactWorld {
    let mut store = ObjectStore::new();
    let samples: Vec<i32> = (0..64).map(intrinsic_sample).collect();
    let model = ResidualModel::Zero;
    let records = Residual::closing_residual(&samples, 1, &model).unwrap();
    let d =
        ObjectDescriptor::new(Representation::PredictorResidual, 64, Layout::Mono, None).unwrap();
    let id = store
        .insert(
            d.clone(),
            ObjectData::PredictorResidual(Residual::new(&d, model, records).unwrap()),
        )
        .unwrap();
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(voice_mono(id, 0))]).unwrap();
    FactWorld { store, world }
}

fn reference_world() -> FactWorld {
    let mut store = ObjectStore::new();
    let samples: Vec<i32> = (0..64).map(|i| i << 20).collect();
    let d = ObjectDescriptor::new(Representation::Literal, 64, Layout::Mono, None).unwrap();
    let lit = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, samples).unwrap()),
        )
        .unwrap();
    let target = store.get(lit).unwrap().content_id;
    let rd = ObjectDescriptor::new(Representation::Referenced, 64, Layout::Mono, None).unwrap();
    let rid = store
        .insert(
            rd.clone(),
            ObjectData::Referenced(
                crate::object::reference::Referenced::checked(target, 1 << 24, None).unwrap(),
            ),
        )
        .unwrap();
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(voice_mono(rid, 0))]).unwrap();
    FactWorld { store, world }
}

/// Authority-level transpose oracle: compose_transpose vs an independent
/// i128 formula, ceiling saturation, and out-of-domain rejection. Failures are
/// returned as `Err` (never panics) so `court facts` can emit an honest
/// FAILED_CORRECTNESS receipt.
fn transpose_oracle() -> Result<String> {
    use crate::error::Error;
    use crate::object::compose_transpose;
    let oracle = |a: i64, b: i64| -> i64 {
        let prod = i128::from(a) * i128::from(b);
        let half = 1i128 << 23;
        let q = if prod >= 0 {
            (prod + half) >> 24
        } else {
            -(((-prod) + half) >> 24)
        };
        q.clamp(-(1 << 47), 1 << 47) as i64
    };
    let mut rng = crate::universe::prng::XoShiro256::from_seed(0x07A1_AF15);
    let mut count = 0u64;
    for _ in 0..100_000 {
        let a = rng.next_u64() as i64;
        let b = rng.next_u64() as i64;
        let got = compose_transpose(a, b);
        if got != oracle(a, b) {
            return Err(Error::integrity(format!(
                "compose_transpose({a}, {b}) = {got}, i128 oracle says {}",
                oracle(a, b)
            )));
        }
        count += 1;
    }
    // Documented ceiling saturation on domain extremes.
    let expect = |a: i64, b: i64, want: i64, what: &str| -> Result<()> {
        let got = compose_transpose(a, b);
        if got != want {
            return Err(Error::integrity(format!(
                "{what}: compose_transpose({a}, {b}) = {got}, expected {want}"
            )));
        }
        Ok(())
    };
    expect(1 << 40, 1 << 47, 1 << 47, "ceiling saturation")?;
    expect(
        -(1 << 40),
        -(1 << 47),
        1 << 47,
        "negative*negative saturates positive ceiling",
    )?;
    // Out-of-domain composed rate must be *rejected* at resolution.
    let mut store = ObjectStore::new();
    let d = ObjectDescriptor::new(Representation::Literal, 8, Layout::Mono, None).unwrap();
    let lit = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, vec![0; 8]).unwrap()),
        )
        .unwrap();
    let rd = ObjectDescriptor::new(Representation::Referenced, 8, Layout::Mono, None).unwrap();
    let rid = store
        .insert(
            rd.clone(),
            ObjectData::Referenced(
                crate::object::reference::Referenced::checked(
                    store.get(lit).unwrap().content_id,
                    1 << 25,
                    None,
                )
                .unwrap(),
            ),
        )
        .unwrap();
    let mut spec = voice_mono(rid, 0);
    spec.rate_q24 = 1 << 40; // composed effective rate ~2^41 > MAX_RATE_Q24
    let world = World::new(RATE, 1, vec![TimelineEvent::VoiceOn(spec)]).unwrap();
    if world.observe(&store, 0, 1).is_ok() {
        return Err(Error::integrity(
            "out-of-domain composed rate must be rejected, never wrapped",
        ));
    }
    Ok(format!(
        "compose_transpose == i128 oracle on {count} random pairs; ceiling saturation and \
         out-of-domain rejection verified"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_ids_are_unique_and_windows_match_expected() {
        let facts = registry();
        let mut seen = std::collections::BTreeSet::new();
        for f in &facts {
            assert!(seen.insert(f.id), "duplicate fact id {}", f.id);
            assert!(!f.name.is_empty() && !f.description.is_empty());
            if let Expectation::Windows {
                windows, expected, ..
            } = &f.expectation
            {
                assert_eq!(
                    windows.len(),
                    expected.len(),
                    "{}: windows/expected length mismatch",
                    f.id
                );
                for (w, e) in windows.iter().zip(expected.iter()) {
                    assert!(!e.is_empty(), "{}: empty expected buffer", f.id);
                    // Every window must be a legal observation (start >= 0;
                    // frame counts under the quantum ceiling are checked by
                    // World::observe at run time).
                    assert!(w.0 >= 0, "{}: negative window start", f.id);
                }
            }
        }
        // The registry is a deliberate, closed list.
        assert!(facts.len() >= 15, "registry shrank: {}", facts.len());
    }

    /// Every fact must pass on every available host surface. This is the
    /// `cargo test` twin of `court facts` (which additionally writes an
    /// immutable receipt).
    #[test]
    fn every_fact_passes_on_every_host_surface() {
        let results = run_facts().expect("facts runner must not fail");
        assert_eq!(results.len(), registry().len());
        let mut failures = Vec::new();
        for r in &results {
            for row in &r.rows {
                if !row.passed {
                    failures.push(format!(
                        "fact {} on surface {} failed: {}",
                        r.fact_id,
                        row.surface,
                        row.detail.as_deref().unwrap_or("no detail")
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} semantic fact(s) failed:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

//! The flagship corpus manifest (Phase M) — identity-bearing, frozen, verified.
//!
//! The manifest is data (`corpus/manifest.json`, shipped inside the crate and in
//! the seal subject). It is written **once** by `vole-audio corpus freeze` from
//! the frozen membership in [`specs`], and every later run of
//! `vole-audio corpus verify` regenerates each object and proves it.
//!
//! The manifest's identity covers everything that can change a result — not just
//! the sample bytes:
//!
//! ```text
//! object identity bytes =
//!     id ∥ representation ∥ amplitude ∥ channel structure ∥ temporal ∥ entropy
//!   ∥ rate ∥ channels ∥ frames ∥ semantics ∥ repeat period
//!   ∥ generator kind ∥ generator parameters ∥ source ∥ canonical i32 hash
//! ```
//!
//! and `manifest_sha256` covers the whole object set plus the population counts.
//! A flagship result can therefore state exactly which frozen population, under
//! which generator parameters, produced it.
//!
//! `verify` fails on **any** of: a manifest that does not parse or has the wrong
//! schema; a corpus hash that does not match its objects; an object missing from
//! the manifest that the frozen membership defines (or the reverse); a
//! mis-sized, wrong-rate or wrong-channel object; a generator whose regenerated
//! samples do not match the frozen canonical hash.

use crate::error::{Error, Result};
use crate::hash::sha256::{Sha256, hex};
use serde::{Deserialize, Serialize};

pub mod generate;
pub mod specs;

use self::generate::{
    Amplitude, ChannelStructure, Entropy, Representation, Signal, Spec, Temporal,
};

/// Frozen manifest schema id.
pub const SCHEMA: &str = "vole.audio.corpus.v1";

/// The frozen manifest, embedded from the repository (shipped in the crate).
pub const MANIFEST_JSON: &str = include_str!("../../corpus/manifest.json");

/// Conversion path recorded for synthetic objects: there is none.
pub const SYNTHETIC_CONVERSION: &str =
    "none: canonical i32 generated directly (no decode, shift, dither or resampling)";

/// One object's frozen identity and generator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestObject {
    pub id: String,
    pub class: String,
    pub representation_class: String,
    pub amplitude_class: String,
    pub channel_structure: String,
    pub temporal_class: String,
    pub entropy_class: String,
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub frames: u64,
    pub duration_ms: u64,
    pub semantics: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_period_frames: Option<u32>,
    pub generator: Signal,
    pub source: String,
    pub conversion: String,
    /// Which baselines/surfaces are expected to include this object.
    pub expected_inclusion_surfaces: Vec<String>,
    /// `false` means B1 (FLAC) is `NOT_APPLICABLE_BY_FORMAT_DOMAIN` for this
    /// object and its bytes must never enter a B1-vs-VOLE aggregate.
    pub b1_comparable: bool,
    /// Canonical SHA-256 over the interleaved i32 samples (hex).
    pub canonical_i32_sha256: String,
}

/// The whole frozen manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub schema: String,
    pub universe: String,
    pub profile: String,
    pub state: String,
    pub note: String,
    pub populations: PopulationsJson,
    pub objects: Vec<ManifestObject>,
    /// SHA-256 over every object's identity bytes, in manifest order (hex).
    pub corpus_sha256: String,
}

/// Population counts, as recorded in the manifest.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct PopulationsJson {
    pub whole_corpus_objects: usize,
    pub b1_comparable_objects: usize,
    pub b1_excluded_objects: usize,
    pub high_channel_stress_objects: usize,
}

/// Parse the embedded manifest.
pub fn manifest() -> Result<Manifest> {
    serde_json::from_str(MANIFEST_JSON)
        .map_err(|e| Error::internal(format!("corpus manifest does not parse: {e}")))
}

/// The canonical identity bytes of one object.
fn identity_bytes(o: &ManifestObject) -> Vec<u8> {
    fn push_str(v: &mut Vec<u8>, x: &str) {
        v.extend_from_slice(x.as_bytes());
        v.push(0);
    }
    let mut v = Vec::new();
    push_str(&mut v, &o.id);
    push_str(&mut v, &o.representation_class);
    push_str(&mut v, &o.amplitude_class);
    push_str(&mut v, &o.channel_structure);
    push_str(&mut v, &o.temporal_class);
    push_str(&mut v, &o.entropy_class);
    v.extend_from_slice(&o.sample_rate_hz.to_le_bytes());
    v.push(o.channels);
    v.extend_from_slice(&o.frames.to_le_bytes());
    push_str(&mut v, &o.semantics);
    v.extend_from_slice(&o.repeat_period_frames.unwrap_or(u32::MAX).to_le_bytes());
    push_str(&mut v, &o.source);
    v.extend_from_slice(
        serde_json::to_string(&o.generator)
            .unwrap_or_default()
            .as_bytes(),
    );
    v.push(0);
    v.extend_from_slice(&(o.b1_comparable as u8).to_le_bytes());
    push_str(&mut v, &o.canonical_i32_sha256);
    v
}

/// SHA-256 over every object's identity bytes (the manifest's `corpus_sha256`).
pub fn corpus_sha256(objects: &[ManifestObject]) -> [u8; 32] {
    let mut h = Sha256::new();
    for o in objects {
        h.update(&identity_bytes(o));
    }
    h.finalize()
}

/// The whole-manifest identity: schema, universe, profile, population counts and
/// the corpus hash. This is the value a flagship result binds to.
pub fn manifest_sha256(m: &Manifest) -> [u8; 32] {
    let mut h = Sha256::new();
    for s in [
        &m.schema,
        &m.universe,
        &m.profile,
        &m.state,
        &m.corpus_sha256,
    ] {
        h.update(s.as_bytes());
        h.update(&[0]);
    }
    h.update(&(m.populations.whole_corpus_objects as u64).to_le_bytes());
    h.update(&(m.populations.b1_comparable_objects as u64).to_le_bytes());
    h.update(&(m.populations.b1_excluded_objects as u64).to_le_bytes());
    h.update(&(m.populations.high_channel_stress_objects as u64).to_le_bytes());
    h.finalize()
}

/// Build the manifest object for one spec (used by `corpus freeze`).
pub fn object_for(spec: &Spec, canonical: &[i32]) -> ManifestObject {
    let repr = spec.representation;
    let amp = spec.amplitude;
    let ch = spec.channel_structure;
    let temporal = spec.temporal;
    let entropy = spec.entropy;
    let mut surfaces = vec!["B0".to_string()];
    if spec.b1_comparable() {
        surfaces.push("B1".to_string());
    } else {
        surfaces.push("B1_NOT_APPLICABLE_BY_FORMAT_DOMAIN".to_string());
    }
    surfaces.extend(["B5".to_string(), "B6".to_string()]);
    ManifestObject {
        id: spec.id.clone(),
        class: "generated".to_string(),
        representation_class: repr.as_str().to_string(),
        amplitude_class: amp.as_str().to_string(),
        channel_structure: ch.as_str().to_string(),
        temporal_class: temporal.as_str().to_string(),
        entropy_class: entropy.as_str().to_string(),
        sample_rate_hz: spec.sample_rate_hz,
        channels: spec.channels,
        frames: spec.frames as u64,
        duration_ms: spec.duration_ms(),
        semantics: spec.semantics.kind().to_string(),
        repeat_period_frames: spec.semantics.period_frames(),
        generator: spec.signal,
        source: spec.source.clone(),
        conversion: SYNTHETIC_CONVERSION.to_string(),
        expected_inclusion_surfaces: surfaces,
        b1_comparable: spec.b1_comparable(),
        canonical_i32_sha256: hex(&self::generate::canonical_sha256(canonical)),
    }
}

/// Build the full manifest from the frozen membership (the freeze act).
pub fn frozen_manifest(specs: &[Spec]) -> Result<Manifest> {
    let mut objects = Vec::with_capacity(specs.len());
    for s in specs {
        let samples = self::generate::generate(s)?;
        objects.push(object_for(s, &samples));
    }
    let p = specs::populations(specs);
    let corpus = corpus_sha256(&objects);
    Ok(Manifest {
        schema: SCHEMA.to_string(),
        universe: "vole.audio.u1".to_string(),
        profile: "u1/v1".to_string(),
        state: "FROZEN".to_string(),
        note: "Frozen before any flagship result exists. Membership and class \
               assignments are not changed because a result is unfavourable \
               (contract §48). Objects are generated, never stored: the manifest \
               carries the generator plus the canonical content hash."
            .to_string(),
        populations: PopulationsJson {
            whole_corpus_objects: p.whole_corpus_objects,
            b1_comparable_objects: p.b1_comparable_objects,
            b1_excluded_objects: p.b1_excluded_objects,
            high_channel_stress_objects: p.high_channel_stress_objects,
        },
        objects,
        corpus_sha256: hex(&corpus),
    })
}

/// One verification finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub id: String,
    pub kind: FindingKind,
    pub detail: String,
}

/// The ways a frozen corpus can be wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingKind {
    MissingFromManifest,
    ExtraInManifest,
    IdentityChanged,
    MisSized,
    WrongRate,
    WrongChannels,
    HashMismatch,
    GeneratorError,
    CorpusHashMismatch,
    SchemaMismatch,
    PopulationMismatch,
}

impl FindingKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            FindingKind::MissingFromManifest => "missing_from_manifest",
            FindingKind::ExtraInManifest => "extra_in_manifest",
            FindingKind::IdentityChanged => "identity_changed",
            FindingKind::MisSized => "mis_sized",
            FindingKind::WrongRate => "wrong_rate",
            FindingKind::WrongChannels => "wrong_channels",
            FindingKind::HashMismatch => "hash_mismatch",
            FindingKind::GeneratorError => "generator_error",
            FindingKind::CorpusHashMismatch => "corpus_hash_mismatch",
            FindingKind::SchemaMismatch => "schema_mismatch",
            FindingKind::PopulationMismatch => "population_mismatch",
        }
    }
}

/// The result of verifying the corpus.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub objects: usize,
    pub verified: usize,
    pub findings: Vec<Finding>,
    pub corpus_sha256: String,
    pub manifest_sha256: String,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.findings.is_empty()
    }
}

/// Verify the embedded manifest against the frozen membership and the
/// regenerated objects.
///
/// Fails on a schema mismatch, a corpus hash that does not cover its objects,
/// membership drift in either direction, a class/size/rate/channel mutation, a
/// generator whose output no longer matches the frozen hash, and a population
/// count that disagrees with the objects.
pub fn verify() -> Result<VerifyReport> {
    verify_manifest(&manifest()?)
}

/// Verify a specific manifest (the embedded one in production; a mutated one in
/// the mutation battery). See [`verify`] for the failure surface.
pub fn verify_manifest(m: &Manifest) -> Result<VerifyReport> {
    let mut findings: Vec<Finding> = Vec::new();

    if m.schema != SCHEMA {
        findings.push(Finding {
            id: "<manifest>".into(),
            kind: FindingKind::SchemaMismatch,
            detail: format!("schema {} != {SCHEMA}", m.schema),
        });
    }

    // Corpus hash must cover exactly these objects.
    let computed_corpus = hex(&corpus_sha256(&m.objects));
    if computed_corpus != m.corpus_sha256 {
        findings.push(Finding {
            id: "<manifest>".into(),
            kind: FindingKind::CorpusHashMismatch,
            detail: format!(
                "computed {computed_corpus}, manifest declares {}",
                m.corpus_sha256
            ),
        });
    }

    // Membership drift: compare the frozen membership (code) with the manifest.
    let frozen = specs::specs();
    let frozen_by_id: std::collections::BTreeMap<&str, &Spec> =
        frozen.iter().map(|s| (s.id.as_str(), s)).collect();
    let manifest_by_id: std::collections::BTreeMap<&str, &ManifestObject> =
        m.objects.iter().map(|o| (o.id.as_str(), o)).collect();
    for id in frozen_by_id.keys() {
        if !manifest_by_id.contains_key(id) {
            findings.push(Finding {
                id: (*id).to_string(),
                kind: FindingKind::MissingFromManifest,
                detail: "the frozen membership defines this object but the manifest does not"
                    .into(),
            });
        }
    }
    for id in manifest_by_id.keys() {
        if !frozen_by_id.contains_key(id) {
            findings.push(Finding {
                id: (*id).to_string(),
                kind: FindingKind::ExtraInManifest,
                detail: "the manifest contains an object the frozen membership does not define"
                    .into(),
            });
        }
    }

    // Per-object verification.
    let mut verified = 0usize;
    for o in &m.objects {
        // Identity fields must agree with the frozen membership when present.
        if let Some(s) = frozen_by_id.get(o.id.as_str()) {
            let expected_repr = s.representation.as_str();
            let expected_amp = s.amplitude.as_str();
            let expected_ch = s.channel_structure.as_str();
            let expected_t = s.temporal.as_str();
            let expected_e = s.entropy.as_str();
            if o.representation_class != expected_repr
                || o.amplitude_class != expected_amp
                || o.channel_structure != expected_ch
                || o.temporal_class != expected_t
                || o.entropy_class != expected_e
                || o.sample_rate_hz != s.sample_rate_hz
                || o.channels != s.channels
                || o.frames != s.frames as u64
                || o.generator != s.signal
            {
                findings.push(Finding {
                    id: o.id.clone(),
                    kind: FindingKind::IdentityChanged,
                    detail: "manifest identity differs from the frozen membership".into(),
                });
                continue;
            }
        }

        // Structural checks that do not need regeneration.
        if o.sample_rate_hz == 0 {
            findings.push(Finding {
                id: o.id.clone(),
                kind: FindingKind::WrongRate,
                detail: "zero sample rate".into(),
            });
            continue;
        }
        if o.channels == 0 {
            findings.push(Finding {
                id: o.id.clone(),
                kind: FindingKind::WrongChannels,
                detail: "zero channels".into(),
            });
            continue;
        }
        if o.frames == 0 || o.frames > crate::limits::MAX_OBJECT_FRAMES {
            findings.push(Finding {
                id: o.id.clone(),
                kind: FindingKind::MisSized,
                detail: format!("frames {} out of domain", o.frames),
            });
            continue;
        }

        // Regenerate and check the content hash and structure.
        let (
            Some(representation),
            Some(amplitude),
            Some(channel_structure),
            Some(temporal),
            Some(entropy),
        ) = (
            parse_repr(&o.representation_class),
            parse_amp(&o.amplitude_class),
            parse_channels(&o.channel_structure),
            parse_temporal(&o.temporal_class),
            parse_entropy(&o.entropy_class),
        )
        else {
            findings.push(Finding {
                id: o.id.clone(),
                kind: FindingKind::IdentityChanged,
                detail: "the manifest carries an unknown class label".into(),
            });
            continue;
        };
        let spec = Spec {
            id: o.id.clone(),
            representation,
            amplitude,
            channel_structure,
            temporal,
            entropy,
            sample_rate_hz: o.sample_rate_hz,
            channels: o.channels,
            frames: o.frames as usize,
            semantics: parse_semantics(&o.semantics, o.repeat_period_frames),
            signal: o.generator,
            source: o.source.clone(),
        };
        match self::generate::generate(&spec) {
            Ok(samples) => {
                if samples.len() != o.frames as usize * usize::from(o.channels) {
                    findings.push(Finding {
                        id: o.id.clone(),
                        kind: FindingKind::MisSized,
                        detail: format!(
                            "regenerated {} samples, manifest declares {}",
                            samples.len(),
                            o.frames as usize * usize::from(o.channels)
                        ),
                    });
                    continue;
                }
                let got = hex(&self::generate::canonical_sha256(&samples));
                if got != o.canonical_i32_sha256 {
                    findings.push(Finding {
                        id: o.id.clone(),
                        kind: FindingKind::HashMismatch,
                        detail: format!("regenerated {got}, frozen {}", o.canonical_i32_sha256),
                    });
                    continue;
                }
                verified += 1;
            }
            Err(e) => findings.push(Finding {
                id: o.id.clone(),
                kind: FindingKind::GeneratorError,
                detail: format!("{e}"),
            }),
        }
    }

    // Population accounting must agree with the objects.
    let b1_ok = m.objects.iter().filter(|o| o.b1_comparable).count();
    let b1_no = m.objects.len() - b1_ok;
    if m.populations.whole_corpus_objects != m.objects.len()
        || m.populations.b1_comparable_objects != b1_ok
        || m.populations.b1_excluded_objects != b1_no
        || m.populations.high_channel_stress_objects != b1_no
    {
        findings.push(Finding {
            id: "<manifest>".into(),
            kind: FindingKind::PopulationMismatch,
            detail: format!(
                "declared whole/b1_ok/b1_excluded/high_channel = {}/{}/{}/{}, objects say {}/{}/{}/{}",
                m.populations.whole_corpus_objects,
                m.populations.b1_comparable_objects,
                m.populations.b1_excluded_objects,
                m.populations.high_channel_stress_objects,
                m.objects.len(),
                b1_ok,
                b1_no,
                b1_no
            ),
        });
    }

    Ok(VerifyReport {
        objects: m.objects.len(),
        verified,
        findings,
        corpus_sha256: m.corpus_sha256.clone(),
        manifest_sha256: hex(&manifest_sha256(m)),
    })
}

// Class parsing (the manifest is data; an unknown label is a verification
// finding, never a silent default).
fn parse_repr(s: &str) -> Option<Representation> {
    Some(match s {
        "literal" => Representation::Literal,
        "exact_repetition" => Representation::ExactRepetition,
        "oscillator" => Representation::Oscillator,
        "wavetable" => Representation::Wavetable,
        "residual" => Representation::Residual,
        "compound" => Representation::Compound,
        "noise" => Representation::Noise,
        _ => return None,
    })
}

fn parse_amp(s: &str) -> Option<Amplitude> {
    Some(match s {
        "low_byte" => Amplitude::LowByte,
        "s16_like" => Amplitude::S16,
        "s24_like" => Amplitude::S24,
        "full_i32" => Amplitude::Full,
        _ => return None,
    })
}

fn parse_channels(s: &str) -> Option<ChannelStructure> {
    Some(match s {
        "mono" => ChannelStructure::Mono,
        "identical_stereo" => ChannelStructure::IdenticalStereo,
        "correlated_stereo" => ChannelStructure::CorrelatedStereo,
        "anticorrelated_stereo" => ChannelStructure::AnticorrelatedStereo,
        "independent_stereo" => ChannelStructure::IndependentStereo,
        "multichannel" => ChannelStructure::Multichannel,
        _ => return None,
    })
}

fn parse_temporal(s: &str) -> Option<Temporal> {
    Some(match s {
        "stationary" => Temporal::Stationary,
        "transient" => Temporal::Transient,
        "loop" => Temporal::Loop,
        "one_shot" => Temporal::OneShot,
        "slowly_varying" => Temporal::SlowlyVarying,
        "strongly_modulated" => Temporal::StronglyModulated,
        _ => return None,
    })
}

fn parse_entropy(s: &str) -> Option<Entropy> {
    Some(match s {
        "highly_predictable" => Entropy::HighlyPredictable,
        "locally_predictable" => Entropy::LocallyPredictable,
        "globally_periodic" => Entropy::GloballyPeriodic,
        "sparse_residual" => Entropy::SparseResidual,
        "spectrally_structured" => Entropy::SpectrallyStructured,
        "full_width_random" => Entropy::FullWidthRandom,
        "scrambled" => Entropy::Scrambled,
        _ => return None,
    })
}

fn parse_semantics(kind: &str, period: Option<u32>) -> self::generate::Semantics {
    match (kind, period) {
        ("loop", Some(p)) => self::generate::Semantics::Loop { period_frames: p },
        _ => self::generate::Semantics::OneShot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_verifies() {
        let r = verify().expect("verify");
        assert!(
            r.ok(),
            "embedded manifest must verify; findings: {:?}",
            r.findings
        );
        assert!(r.objects >= 95, "corpus too small: {}", r.objects);
        assert!(r.verified >= 95);
    }

    #[test]
    fn manifest_hash_is_stable_and_covers_identity() {
        let m = manifest().unwrap();
        let a = manifest_sha256(&m);
        let b = manifest_sha256(&m);
        assert_eq!(a, b, "manifest hash must be deterministic");

        // Mutating any identity field changes the corpus hash.
        let mut m2 = m.clone();
        m2.objects[0].sample_rate_hz = 96_000;
        assert_ne!(corpus_sha256(&m.objects), corpus_sha256(&m2.objects));

        let mut m3 = m.clone();
        m3.objects[0].canonical_i32_sha256 = "00".repeat(32);
        assert_ne!(corpus_sha256(&m.objects), corpus_sha256(&m3.objects));

        let mut m4 = m.clone();
        m4.populations.b1_comparable_objects += 1;
        assert_ne!(manifest_sha256(&m), manifest_sha256(&m4));
    }

    #[test]
    fn corpus_hash_is_order_sensitive() {
        let m = manifest().unwrap();
        let mut swapped = m.objects.clone();
        if swapped.len() >= 2 {
            swapped.swap(0, 1);
            assert_ne!(corpus_sha256(&m.objects), corpus_sha256(&swapped));
        }
    }

    /// The gate must fail on every mutation class a reviewer would worry about.
    #[test]
    fn verification_fails_on_every_mutation_class() {
        let base = manifest().unwrap();
        assert!(
            verify_manifest(&base).unwrap().ok(),
            "the unmutated manifest must verify"
        );
        // A three-object manifest for the mutation classes that do not depend on
        // the full frozen membership (regenerating the whole corpus 13 times
        // would dominate the suite for no extra signal).
        let small = frozen_manifest(&specs::specs()[..3]).unwrap();
        let kinds = |m: &Manifest| -> Vec<FindingKind> {
            let mut v: Vec<FindingKind> = verify_manifest(m)
                .unwrap()
                .findings
                .into_iter()
                .map(|f| f.kind)
                .collect();
            v.sort_by_key(|k| k.as_str());
            v.dedup();
            v
        };
        assert!(!base.objects.is_empty());

        // Schema mutation.
        let mut m = small.clone();
        m.schema = "vole.audio.corpus.v2".into();
        assert!(kinds(&m).contains(&FindingKind::SchemaMismatch));

        // Corpus-hash mutation (the objects no longer match their declared hash).
        let mut m = small.clone();
        m.corpus_sha256 = "00".repeat(32);
        assert!(kinds(&m).contains(&FindingKind::CorpusHashMismatch));

        // A missing object.
        let mut m = base.clone();
        m.objects.remove(0);
        assert!(kinds(&m).contains(&FindingKind::MissingFromManifest));

        // An extra object (an id the frozen membership does not define).
        let mut m = small.clone();
        let mut extra = m.objects[0].clone();
        extra.id = "smuggled-object".into();
        m.objects.push(extra);
        assert!(kinds(&m).contains(&FindingKind::ExtraInManifest));

        // A class mutation against the frozen membership.
        let mut m = base.clone();
        m.objects[0].entropy_class = "scrambled".into();
        assert!(kinds(&m).contains(&FindingKind::IdentityChanged));

        // A rate mutation against the frozen membership.
        let mut m = base.clone();
        m.objects[0].sample_rate_hz = 192_000;
        assert!(kinds(&m).contains(&FindingKind::IdentityChanged));

        // A size mutation: frames changed but the id kept.
        let mut m = base.clone();
        m.objects[0].frames += 1;
        assert!(kinds(&m).contains(&FindingKind::IdentityChanged));

        // A content mutation: the frozen samples no longer regenerate.
        let mut m = small.clone();
        m.objects[0].canonical_i32_sha256 = "11".repeat(32);
        let k = kinds(&m);
        assert!(
            k.contains(&FindingKind::HashMismatch) || k.contains(&FindingKind::CorpusHashMismatch),
            "a mutated content hash must fail: {k:?}"
        );

        // Structural mutations that do not need the membership cross-check.
        let mut m = small.clone();
        m.objects[0].sample_rate_hz = 0;
        m.objects[0].id = "zero-rate".into();
        assert!(kinds(&m).contains(&FindingKind::WrongRate));

        let mut m = small.clone();
        m.objects[0].channels = 0;
        m.objects[0].id = "zero-channels".into();
        assert!(kinds(&m).contains(&FindingKind::WrongChannels));

        let mut m = small.clone();
        m.objects[0].frames = 0;
        m.objects[0].id = "zero-frames".into();
        assert!(kinds(&m).contains(&FindingKind::MisSized));

        // Population accounting drift.
        let mut m = small.clone();
        m.populations.b1_comparable_objects += 1;
        assert!(kinds(&m).contains(&FindingKind::PopulationMismatch));

        // Generator mutation: a different recipe for the same id.
        let mut m = small.clone();
        if let Some(o) = m.objects.first_mut() {
            o.generator = Signal::Uniform { seed: 0xdead_beef };
        }
        let k = kinds(&m);
        assert!(
            k.contains(&FindingKind::HashMismatch)
                || k.contains(&FindingKind::IdentityChanged)
                || k.contains(&FindingKind::CorpusHashMismatch),
            "a mutated generator must fail: {k:?}"
        );
    }
}

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
//!     id ∥ class ∥ source structure ∥ amplitude ∥ channel structure ∥ temporal
//!   ∥ entropy ∥ rate ∥ channels ∥ frames ∥ semantics ∥ repeat period
//!   ∥ generator kind ∥ generator parameters ∥ source ∥ conversion
//!   ∥ b1 comparable ∥ canonical i32 hash
//! ```
//!
//! `duration_ms` is *derived* and recomputed+verified rather than hashed, and
//! `expected_inclusion_surfaces` is *derived policy* recomputed during
//! verification, not independent authority. `manifest_sha256` covers the whole
//! object set plus the population counts. A flagship result can therefore state
//! exactly which frozen population, under which generator parameters, produced
//! it.
//!
//! `verify` fails on **any** of: a manifest that does not parse, has the wrong
//! schema, universe, profile or state; a corpus hash that does not match its
//! objects; a duplicate object id; a missing or extra object versus the frozen
//! membership; a manifest order that differs from the frozen membership
//! (benchmark order is experimental state); a mis-sized, wrong-rate,
//! wrong-channel or wrong-`b1_comparable` object; a generator whose regenerated
//! samples do not match the frozen canonical hash; any identity field that
//! differs from the canonical regenerated object; and population-count drift
//! from the **derived** format-domain counts.

use crate::error::{Error, Result};
use crate::hash::sha256::{Sha256, hex};
use serde::{Deserialize, Serialize};

pub mod generate;
pub mod specs;

use self::generate::{Signal, Spec};

/// Frozen manifest schema id.
pub const SCHEMA: &str = "vole.audio.corpus.v1";

/// Frozen manifest universe id.
pub const UNIVERSE: &str = "vole.audio.u1";

/// Frozen manifest profile id.
pub const PROFILE: &str = "u1/v1";

/// The only admissible state of a frozen corpus.
pub const STATE_FROZEN: &str = "FROZEN";

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
    /// The frozen **source (generative) structure class** of the object. This is
    /// what the material was designed as, not the representation the inverse
    /// compiler later selects; measured results record `selected_representation`
    /// separately.
    pub source_structure_class: String,
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
    push_str(&mut v, &o.class);
    push_str(&mut v, &o.source_structure_class);
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
    // The conversion path is provenance-critical for future real recordings:
    // changing how a canonical i32 was derived must change the identity.
    push_str(&mut v, &o.conversion);
    v.extend_from_slice(
        serde_json::to_string(&o.generator)
            .unwrap_or_default()
            .as_bytes(),
    );
    v.push(0);
    v.push(o.b1_comparable as u8);
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
///
/// This is also the **canonical** object the verifier compares the manifest
/// against, field for field: given a frozen `Spec` and its regenerated samples,
/// `object_for` is the single source of truth for every derived field (`class`,
/// the inclusion surfaces, `b1_comparable`, `duration_ms`, the content hash).
pub fn object_for(spec: &Spec, canonical: &[i32]) -> ManifestObject {
    let b1 = spec.b1_comparable();
    let mut surfaces = vec!["B0".to_string()];
    if b1 {
        surfaces.push("B1".to_string());
    } else {
        surfaces.push("B1_NOT_APPLICABLE_BY_FORMAT_DOMAIN".to_string());
    }
    surfaces.extend(["B5".to_string(), "B6".to_string()]);
    ManifestObject {
        id: spec.id.clone(),
        class: "generated".to_string(),
        source_structure_class: spec.source_structure.as_str().to_string(),
        amplitude_class: spec.amplitude.as_str().to_string(),
        channel_structure: spec.channel_structure.as_str().to_string(),
        temporal_class: spec.temporal.as_str().to_string(),
        entropy_class: spec.entropy.as_str().to_string(),
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
        b1_comparable: b1,
        canonical_i32_sha256: hex(&self::generate::canonical_sha256(canonical)),
    }
}

/// Derive the B1 population counts from the **format domain** (channel count),
/// never from the manifest's audit field. This is what courts must use for the
/// B1-vs-VOLE denominator.
pub fn derived_b1_counts(objects: &[ManifestObject]) -> (usize, usize) {
    let mut comparable = 0usize;
    for o in objects {
        if self::generate::b1_comparable(o.channels) {
            comparable += 1;
        }
    }
    (comparable, objects.len() - comparable)
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
        universe: UNIVERSE.to_string(),
        profile: PROFILE.to_string(),
        state: STATE_FROZEN.to_string(),
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
    /// The manifest's universe, profile or state is not the frozen one.
    RootMismatch,
    /// An object id appears more than once.
    DuplicateId,
    /// The manifest's object order differs from the frozen membership order.
    OrderMismatch,
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
            FindingKind::RootMismatch => "root_mismatch",
            FindingKind::DuplicateId => "duplicate_id",
            FindingKind::OrderMismatch => "order_mismatch",
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
/// Fails on a schema, universe, profile or state mismatch; a corpus hash that
/// does not cover its objects; a duplicate object id; membership drift in either
/// direction; a manifest order that differs from the frozen membership; a
/// size/rate/channel mutation; any identity field (including `class` and
/// `conversion`) that differs from the canonical regenerated object; a generator
/// whose output no longer matches the frozen hash; and a population count that
/// disagrees with the **derived** format-domain counts.
pub fn verify() -> Result<VerifyReport> {
    verify_manifest(&manifest()?)
}

/// One finding, constructed uniformly.
fn finding(id: &str, kind: FindingKind, detail: impl Into<String>) -> Finding {
    Finding {
        id: id.to_string(),
        kind,
        detail: detail.into(),
    }
}

/// Verify a specific manifest (the embedded one in production; a mutated one in
/// the mutation battery). See [`verify`] for the failure surface.
pub fn verify_manifest(m: &Manifest) -> Result<VerifyReport> {
    let mut findings: Vec<Finding> = Vec::new();

    // Root identity and frozen state: a changed universe, profile or state is a
    // different corpus identity, not a compatible one.
    if m.schema != SCHEMA {
        findings.push(finding(
            "<manifest>",
            FindingKind::SchemaMismatch,
            format!("schema {} != {SCHEMA}", m.schema),
        ));
    }
    for (what, got, want) in [
        ("universe", m.universe.as_str(), UNIVERSE),
        ("profile", m.profile.as_str(), PROFILE),
        ("state", m.state.as_str(), STATE_FROZEN),
    ] {
        if got != want {
            findings.push(finding(
                "<manifest>",
                FindingKind::RootMismatch,
                format!("{what} {got} != {want}"),
            ));
        }
    }

    // Corpus hash must cover exactly these objects.
    let computed_corpus = hex(&corpus_sha256(&m.objects));
    if computed_corpus != m.corpus_sha256 {
        findings.push(finding(
            "<manifest>",
            FindingKind::CorpusHashMismatch,
            format!(
                "computed {computed_corpus}, manifest declares {}",
                m.corpus_sha256
            ),
        ));
    }

    // Duplicate ids: a set-based membership comparison would silently collapse
    // them, so an appended copy of an existing object must fail here rather than
    // pass as the same id set.
    let mut seen_ids: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for o in &m.objects {
        if !seen_ids.insert(o.id.as_str()) {
            findings.push(finding(
                &o.id,
                FindingKind::DuplicateId,
                "the manifest lists this object id more than once",
            ));
        }
    }

    // Membership drift in either direction, plus exact frozen order: benchmark
    // order is experimental state (cache, thermal and GPU-clock history), so the
    // frozen *sequence*, not merely the set, must be preserved.
    let frozen = specs::specs();
    let frozen_ids: Vec<&str> = frozen.iter().map(|s| s.id.as_str()).collect();
    let manifest_ids: Vec<&str> = m.objects.iter().map(|o| o.id.as_str()).collect();
    let frozen_set: std::collections::BTreeSet<&str> = frozen_ids.iter().copied().collect();
    let manifest_set: std::collections::BTreeSet<&str> = manifest_ids.iter().copied().collect();
    let manifest_by_id: std::collections::BTreeMap<&str, &ManifestObject> =
        m.objects.iter().map(|o| (o.id.as_str(), o)).collect();
    for id in &frozen_set {
        if !manifest_set.contains(id) {
            findings.push(finding(
                id,
                FindingKind::MissingFromManifest,
                "the frozen membership defines this object but the manifest does not",
            ));
        }
    }
    for id in &manifest_set {
        if !frozen_set.contains(id) {
            findings.push(finding(
                id,
                FindingKind::ExtraInManifest,
                "the manifest contains an object the frozen membership does not define",
            ));
        }
    }
    if manifest_ids != frozen_ids {
        findings.push(finding(
            "<manifest>",
            FindingKind::OrderMismatch,
            format!(
                "manifest id order ({} entries) != frozen membership order ({} entries)",
                manifest_ids.len(),
                frozen_ids.len()
            ),
        ));
    }

    // Structural domain checks apply to every listed object, even one that is
    // not part of the frozen membership.
    for o in &m.objects {
        if o.sample_rate_hz == 0 {
            findings.push(finding(&o.id, FindingKind::WrongRate, "zero sample rate"));
        }
        if o.channels == 0 || o.channels > crate::limits::MAX_CHANNELS as u8 {
            findings.push(finding(
                &o.id,
                FindingKind::WrongChannels,
                format!("channels {} out of domain", o.channels),
            ));
        }
        if o.frames == 0 || o.frames > crate::limits::MAX_OBJECT_FRAMES {
            findings.push(finding(
                &o.id,
                FindingKind::MisSized,
                format!("frames {} out of domain", o.frames),
            ));
        }
    }

    // Canonical per-object verification: regenerate each frozen object and
    // compare the manifest entry against the canonical object field for field.
    // This covers every derived field (class, conversion, duration, inclusion
    // surfaces, b1 flag, content hash) without a hand-maintained subset, and it
    // is fail-closed: an unknown or mutated label is a mismatch, never a silent
    // default.
    let mut verified = 0usize;
    for s in &frozen {
        let Some(o) = manifest_by_id.get(s.id.as_str()) else {
            continue; // already reported as MissingFromManifest
        };
        match self::generate::generate(s) {
            Ok(samples) => {
                let expected = object_for(s, &samples);
                if o.canonical_i32_sha256 != expected.canonical_i32_sha256 {
                    findings.push(finding(
                        &s.id,
                        FindingKind::HashMismatch,
                        format!(
                            "regenerated {}, frozen {}",
                            expected.canonical_i32_sha256, o.canonical_i32_sha256
                        ),
                    ));
                    continue;
                }
                if **o != expected {
                    findings.push(finding(
                        &s.id,
                        FindingKind::IdentityChanged,
                        format!(
                            "manifest identity differs from the canonical object; \
                             differing fields: {}",
                            diff_fields(o, &expected)
                        ),
                    ));
                    continue;
                }
                verified += 1;
            }
            Err(e) => findings.push(finding(&s.id, FindingKind::GeneratorError, format!("{e}"))),
        }
    }

    // B1 eligibility is a format-domain fact, derived from the channel count:
    // the denominator can never be changed by editing a manifest flag. The
    // manifest's field remains audited metadata that must agree with the
    // derivation.
    let mut b1_ok = 0usize;
    for o in &m.objects {
        let derived = self::generate::b1_comparable(o.channels);
        if o.b1_comparable != derived {
            findings.push(finding(
                &o.id,
                FindingKind::PopulationMismatch,
                format!(
                    "b1_comparable {} != format-domain value {derived} for {} channels",
                    o.b1_comparable, o.channels
                ),
            ));
        }
        if derived {
            b1_ok += 1;
        }
    }
    let b1_no = m.objects.len() - b1_ok;
    if m.populations.whole_corpus_objects != m.objects.len()
        || m.populations.b1_comparable_objects != b1_ok
        || m.populations.b1_excluded_objects != b1_no
        || m.populations.high_channel_stress_objects != b1_no
    {
        findings.push(finding(
            "<manifest>",
            FindingKind::PopulationMismatch,
            format!(
                "declared whole/b1_ok/b1_excluded/high_channel = {}/{}/{}/{}, derived {}/{}/{}/{}",
                m.populations.whole_corpus_objects,
                m.populations.b1_comparable_objects,
                m.populations.b1_excluded_objects,
                m.populations.high_channel_stress_objects,
                m.objects.len(),
                b1_ok,
                b1_no,
                b1_no
            ),
        ));
    }

    Ok(VerifyReport {
        objects: m.objects.len(),
        verified,
        findings,
        corpus_sha256: m.corpus_sha256.clone(),
        manifest_sha256: hex(&manifest_sha256(m)),
    })
}

/// The names of the identity fields that differ, for a truthful finding detail.
/// `canonical_i32_sha256` is reported separately as [`FindingKind::HashMismatch`].
fn diff_fields(got: &ManifestObject, expected: &ManifestObject) -> String {
    let mut names: Vec<&str> = Vec::new();
    macro_rules! check {
        ($f:ident) => {
            if got.$f != expected.$f {
                names.push(stringify!($f));
            }
        };
    }
    check!(id);
    check!(class);
    check!(source_structure_class);
    check!(amplitude_class);
    check!(channel_structure);
    check!(temporal_class);
    check!(entropy_class);
    check!(sample_rate_hz);
    check!(channels);
    check!(frames);
    check!(duration_ms);
    check!(semantics);
    check!(repeat_period_frames);
    check!(generator);
    check!(source);
    check!(conversion);
    check!(expected_inclusion_surfaces);
    check!(b1_comparable);
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
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

        // `class` and `conversion` are provenance-critical and must be covered.
        let mut m5 = m.clone();
        m5.objects[0].class = "imported".into();
        assert_ne!(corpus_sha256(&m.objects), corpus_sha256(&m5.objects));
        let mut m6 = m.clone();
        m6.objects[0].conversion = "shifted 8 bits".into();
        assert_ne!(corpus_sha256(&m.objects), corpus_sha256(&m6.objects));
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

        // Manifest root/state must be the frozen one.
        for mutate in [
            (|m: &mut Manifest| m.universe = "vole.audio.u2".into()) as fn(&mut Manifest),
            |m: &mut Manifest| m.profile = "u1/v2".into(),
            |m: &mut Manifest| m.state = "DRAFT".into(),
        ] {
            let mut m = small.clone();
            mutate(&mut m);
            assert!(kinds(&m).contains(&FindingKind::RootMismatch));
        }

        // Duplicate ids: a set comparison would collapse them, so an appended
        // copy of an existing object must fail explicitly.
        let mut m = small.clone();
        let dup = m.objects[0].clone();
        m.objects.push(dup);
        m.corpus_sha256 = hex(&corpus_sha256(&m.objects));
        assert!(kinds(&m).contains(&FindingKind::DuplicateId));

        // Frozen order: benchmark order is experimental state, so a reordering
        // must fail even when the corpus hash is recomputed to match.
        let mut m = small.clone();
        m.objects.swap(0, 1);
        m.corpus_sha256 = hex(&corpus_sha256(&m.objects));
        assert!(kinds(&m).contains(&FindingKind::OrderMismatch));

        // B1 eligibility is derived from the format domain, so editing the
        // manifest audit flag is a failure even when the hash is recomputed.
        let mut m = small.clone();
        m.objects[0].b1_comparable = !m.objects[0].b1_comparable;
        m.corpus_sha256 = hex(&corpus_sha256(&m.objects));
        assert!(kinds(&m).contains(&FindingKind::PopulationMismatch));

        // `duration_ms` is derived: it is recomputed and verified, not trusted.
        let mut m = small.clone();
        m.objects[0].duration_ms += 1;
        m.corpus_sha256 = hex(&corpus_sha256(&m.objects));
        assert!(kinds(&m).contains(&FindingKind::IdentityChanged));

        // `class` and `conversion` are part of the identity the canonical
        // comparison checks (a mutated value must be an identity failure).
        let mut m = small.clone();
        m.objects[0].conversion = "shifted 8 bits".into();
        m.corpus_sha256 = hex(&corpus_sha256(&m.objects));
        assert!(kinds(&m).contains(&FindingKind::IdentityChanged));
    }
}

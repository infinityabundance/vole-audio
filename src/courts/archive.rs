//! `court archive` — the canonical `.volea` archive (Phase N).
//!
//! Builds one archive over the **frozen corpus**: each object's full-object
//! container is an opaque, integrity-bound payload, and the manifest binds its
//! name, representation, geometry and content id. The court proves the archive
//! is canonical (decode re-encodes identically, a second encode is
//! byte-identical), records a reproducible manifest digest, and runs a hostile
//! battery in which mutations are **resealed** so they reach the content
//! validators rather than failing the outer digest.

use crate::corpus;
use crate::corpus::generate::{self, Spec};
use crate::error::{Error, Result};
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::format::archive::{
    ArchiveEntry, ArchiveManifest, ArchiveSession, decode_archive, encode_archive,
};
use crate::format::manifest::Manifest;
use crate::fullobj::{self, FullSemantics};
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::SearchBudget;
use crate::object::descriptor::Representation;
use crate::object::id::ContentId;
use crate::status::Verdict;
use crate::transport::{Checkpoint, encode_checkpoint, encode_event};
use crate::universe::event::{Event, EventClass};
use crate::universe::time::MediaFrame;
use std::collections::BTreeMap;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const ARCHIVE_RESULT_SHA256: &str =
    "81db84de01f23d6ab498075f49b2bd0b326e6feb87707dce81d1b726ac9b22ba";

/// The frozen archive-court protocol identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.archive.protocol.v1";

/// The profile/universe tags the flagship archive carries.
pub const ARCHIVE_PROFILE: &str = "u1/v1";
pub const ARCHIVE_UNIVERSE: &str = "vole.audio.u1";

fn budget() -> SearchBudget {
    SearchBudget::default()
}

fn semantics_of(spec: &Spec) -> FullSemantics {
    match spec.semantics.period_frames() {
        Some(period_frames) => FullSemantics::Loop { period_frames },
        None => FullSemantics::OneShot,
    }
}

/// Everything the archive court's static projection binds.
struct Projection {
    manifest_digest: [u8; 32],
    archive_digest: [u8; 32],
    object_bytes: u64,
    entries: Vec<(String, [u8; 32])>,
    events: u64,
    checkpoints: u64,
    dependencies: u64,
    integrity_rejected: u64,
    structural_rejected: u64,
}

fn static_projection(p: &Projection) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(PROTOCOL_SCHEMA.as_bytes());
    out.push(0);
    out.extend_from_slice(&p.manifest_digest);
    out.extend_from_slice(&p.archive_digest);
    out.extend_from_slice(&p.object_bytes.to_le_bytes());
    out.extend_from_slice(&(p.entries.len() as u64).to_le_bytes());
    for (name, cid) in &p.entries {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(cid);
    }
    out.extend_from_slice(&p.events.to_le_bytes());
    out.extend_from_slice(&p.checkpoints.to_le_bytes());
    out.extend_from_slice(&p.dependencies.to_le_bytes());
    out.extend_from_slice(&p.integrity_rejected.to_le_bytes());
    out.extend_from_slice(&p.structural_rejected.to_le_bytes());
    out
}

/// Run the court; writes an immutable receipt under `receipts/archive/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("archive");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("archive court failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court archive: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let manifest_in = corpus::manifest()?;
    let report = corpus::verify_manifest(&manifest_in)?;
    if !report.ok() {
        return fail("the frozen flagship corpus does not verify");
    }
    let specs = corpus::specs::specs();
    let spec_by_id: BTreeMap<&str, &Spec> = specs.iter().map(|s| (s.id.as_str(), s)).collect();

    // ---- authoring: compile each frozen object's container ----
    let mut entries: Vec<ArchiveEntry> = Vec::with_capacity(manifest_in.objects.len());
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(manifest_in.objects.len());
    for o in &manifest_in.objects {
        let spec = spec_by_id
            .get(o.id.as_str())
            .ok_or_else(|| Error::internal(format!("{}: not in the frozen membership", o.id)))?;
        let samples = generate::generate(spec)?;
        if hex(&generate::canonical_sha256(&samples)) != o.canonical_i32_sha256 {
            return fail(&format!("{}: regenerated content changed", o.id));
        }
        let obj = fullobj::compile_full_object(
            &o.id,
            o.sample_rate_hz,
            o.channels,
            o.frames,
            semantics_of(spec),
            &samples,
            budget(),
        )?;
        let bytes = obj.bytes.clone();
        let content_id = ContentId::from_bytes(obj.sha256());
        entries.push(ArchiveEntry {
            name: o.id.clone(),
            // The payload is a full-object container: a composition of U1
            // segments, recorded as such. The archive treats it as opaque.
            representation: Representation::Compound.tag(),
            channels: o.channels,
            sample_rate_hz: o.sample_rate_hz,
            frames: o.frames,
            content_id,
        });
        payloads.push(bytes);
    }
    // Canonical order: the manifest requires strictly ascending names.
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| entries[a].name.cmp(&entries[b].name));
    let entries: Vec<ArchiveEntry> = order.iter().map(|&i| entries[i].clone()).collect();
    let payloads: Vec<Vec<u8>> = order.iter().map(|&i| payloads[i].clone()).collect();

    // A deterministic session beside the objects: one timed event per object in
    // frozen (name) order, one checkpoint anchoring the end, and one declared
    // external dependency. Events and checkpoints use the transport's canonical
    // payload encodings, so an archive is a complete bundle.
    let mut session = ArchiveSession::default();
    for i in 0..entries.len() {
        let at = (i as i64) * 100;
        let event = Event::new(
            MediaFrame::new(at),
            EventClass::Param,
            i as u64,
            (i & 0xFFFF) as u16,
        );
        session.events.push(encode_event(&event));
    }
    let anchor = (entries.len() as i64) * 100;
    let anchor_state = Sha256::digest(format!("archive-session-anchor:{anchor}").as_bytes());
    session.checkpoints.push(encode_checkpoint(&Checkpoint::new(
        0,
        anchor,
        anchor_state.to_vec(),
    ))?);
    session
        .dependencies
        .push(ContentId::from_bytes(Sha256::digest(
            b"external-dependency",
        )));

    let manifest = ArchiveManifest {
        profile: ARCHIVE_PROFILE.into(),
        universe: ARCHIVE_UNIVERSE.into(),
        entries: entries.clone(),
        session: session.clone(),
    };
    let first = encode_archive(&manifest, &payloads)?;

    // Canonicality: encoding is a pure function of the content.
    let second = encode_archive(&manifest, &payloads)?;
    if first != second {
        return fail("archive encoding is not deterministic");
    }
    let decoded = match decode_archive(&first) {
        Ok(d) => d,
        Err(e) => return fail(&format!("the encoded archive does not decode: {e}")),
    };
    if decoded.objects() != payloads.len() {
        return fail("archive object count mismatch");
    }
    for (i, want) in payloads.iter().enumerate() {
        match decoded.object_bytes(&first, i) {
            Ok(got) if got == want.as_slice() => {}
            _ => return fail("archive object payload did not round-trip"),
        }
    }
    // The session sections must round-trip exactly too.
    if decoded.session() != &session {
        return fail("archive session sections did not round-trip");
    }
    // Decode -> re-encode must be byte-identical (canonical form).
    let reencoded = encode_archive(decoded.manifest(), &payloads)?;
    if reencoded != first {
        return fail("archive is not canonical: re-encoding differs");
    }
    let reproducible = Manifest::from_archive(&decoded);

    // ---- hostile battery: mutations are resealed to reach the validators ----
    let (integrity_rejected, structural_rejected) = hostile_battery(&first)?;

    let object_bytes: u64 = payloads.iter().map(|p| p.len() as u64).sum();
    let projection = Projection {
        manifest_digest: reproducible.digest(),
        archive_digest: decoded.archive_digest(),
        object_bytes,
        entries: entries
            .iter()
            .map(|e| (e.name.clone(), e.content_id.to_bytes()))
            .collect(),
        events: session.events.len() as u64,
        checkpoints: session.checkpoints.len() as u64,
        dependencies: session.dependencies.len() as u64,
        integrity_rejected,
        structural_rejected,
    };
    let result_hex = hex(&Sha256::digest(&static_projection(&projection)));
    if ARCHIVE_RESULT_SHA256.is_empty() {
        eprintln!("court archive: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != ARCHIVE_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {ARCHIVE_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    let verdict = Verdict::Supported;
    let mut builder = ReceiptBuilder::new("archive");
    builder
        .result(verdict)
        .result_detail(format!(
            "canonical archive over {} frozen objects ({} events, {} checkpoints, {} dependencies): {} \
             payload bytes, manifest digest {}, archive digest {}; encoding is deterministic and \
             decode re-encodes byte-identically; hostile battery rejected {integrity_rejected} \
             integrity and {structural_rejected} resealed structural mutations; result sha256 {result_hex}",
            entries.len(),
            session.events.len(),
            session.checkpoints.len(),
            session.dependencies.len(),
            object_bytes,
            hex(&projection.manifest_digest),
            hex(&projection.archive_digest),
        ))
        .params(CourtParams {
            universe: Some(manifest_in.universe.clone()),
            profile: Some(format!("{} + archive.v1", manifest_in.profile)),
            backend: Some("canonical .volea archive container".into()),
            content_kind: Some("flagship corpus (frozen before results): archive".into()),
            ..Default::default()
        })
        .provenance(Provenance {
            reference_hash: Some(result_hex.clone()),
            corpus_hash: Some(report.corpus_sha256.clone()),
            ..Default::default()
        })
        .extra(
            "protocol",
            serde_json::json!({
                "schema": PROTOCOL_SCHEMA,
                "magic": "vole.archive",
                "version": crate::format::archive::VERSION,
                "profile": ARCHIVE_PROFILE,
                "universe": ARCHIVE_UNIVERSE,
                "payload": "one full-object container per frozen object, opaque and integrity-bound",
                "session": "event/checkpoint/dependency sections carrying the transport's canonical \
                            payloads beside the objects, counts frozen in the manifest",
                "canonical": "decode -> re-encode is byte-identical; a second encode equals the first",
                "manifest": "reproducible canonical text (format::manifest) with its own digest",
                "hostile": "truncation, bit flip, reserved byte, unknown kind, offset gap, length lie, \
                            manifest count lie, session kind reorder, malformed event payload, \
                            trailing bytes and an allocation bomb; structural mutations are resealed \
                            so they reach the content validators",
            }),
        )
        .extra(
            "bytes",
            serde_json::json!({
                "archive_bytes": first.len(),
                "object_bytes": object_bytes,
                "object_count": entries.len(),
                "events": session.events.len(),
                "checkpoints": session.checkpoints.len(),
                "dependencies": session.dependencies.len(),
            }),
        )
        .extra(
            "manifest_text",
            serde_json::json!(reproducible.canonical_text()),
        )
        .limitation(
            "OBJECT payloads are opaque bytes whose identity is their SHA-256; the archive does not \
             reinterpret them. A payload that is a full-object container keeps its own semantics",
        )
        .limitation(
            "a DEPENDENCY section declares a required content id and is integrity-bound; it carries \
             no byte accounting and makes no amortization claim (contract §31) — an external \
             dependency's bytes are never counted as zero here",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court archive: {verdict}");
    println!(
        "  objects: {} | archive {} B | manifest digest {}",
        entries.len(),
        first.len(),
        hex(&projection.manifest_digest)
    );
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

/// Recompute every section digest and the trailing archive digest so a mutation
/// reaches the content validators rather than failing the outer digest.
fn reseal(bytes: &mut [u8]) {
    use crate::format::archive::{DIGEST_BYTES, MAGIC, SECTION_RECORD_BYTES};
    let mut p = MAGIC.len() + 1;
    p += 1 + usize::from(bytes[p]);
    p += 1 + usize::from(bytes[p]);
    let count = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
    let ts = p + 4;
    let body_end = bytes.len() - DIGEST_BYTES;
    for i in 0..count {
        let rec = ts + i * SECTION_RECORD_BYTES;
        let off = u64::from_le_bytes(bytes[rec + 2..rec + 10].try_into().unwrap()) as usize;
        let len = u64::from_le_bytes(bytes[rec + 10..rec + 18].try_into().unwrap()) as usize;
        let digest = Sha256::digest(&bytes[off..off + len]);
        bytes[rec + 18..rec + 50].copy_from_slice(&digest);
    }
    let digest = Sha256::digest(&bytes[..body_end]);
    bytes[body_end..].copy_from_slice(&digest);
}

fn must_reject(candidate: &[u8]) -> Result<()> {
    if decode_archive(candidate).is_err() {
        Ok(())
    } else {
        Err(Error::internal("a hostile archive decoded"))
    }
}

/// Returns `(integrity_hostile, structural_hostile)` rejections.
fn hostile_battery(valid: &[u8]) -> Result<(u64, u64)> {
    use crate::format::archive::{DIGEST_BYTES, MAGIC, SECTION_RECORD_BYTES, SectionKind};
    let decoded = decode_archive(valid)?; // control: the fixture is valid
    let object_count = decoded.objects();
    let event_count = decoded.events().len();

    let mut integrity = 0u64;
    // Truncations and bit flips without resealing.
    for cut in [1usize, 16, 64] {
        if cut < valid.len() {
            must_reject(&valid[..valid.len() - cut])?;
            integrity += 1;
        }
    }
    for at in [
        0usize,
        MAGIC.len(),
        MAGIC.len() + 1,
        valid.len() - DIGEST_BYTES - 1,
    ] {
        if at < valid.len() {
            let mut b = valid.to_vec();
            b[at] ^= 0xFF;
            must_reject(&b)?;
            integrity += 1;
        }
    }

    // Structural mutations, resealed so the content validators must catch them.
    let mut p = MAGIC.len() + 1;
    p += 1 + usize::from(valid[p]);
    p += 1 + usize::from(valid[p]);
    let count = u32::from_le_bytes(valid[p..p + 4].try_into().unwrap()) as usize;
    let ts = p + 4;
    let mut structural = 0u64;

    // Reserved byte.
    let mut r = valid.to_vec();
    r[ts + 1] = 1;
    reseal(&mut r);
    must_reject(&r)?;
    structural += 1;
    // Unknown section kind.
    let mut k = valid.to_vec();
    k[ts] = 0x7F;
    reseal(&mut k);
    must_reject(&k)?;
    structural += 1;
    // Swapped kinds (manifest/object order violated).
    if count >= 2 {
        let mut s = valid.to_vec();
        s[ts] = SectionKind::Object as u8;
        s[ts + SECTION_RECORD_BYTES] = SectionKind::Manifest as u8;
        reseal(&mut s);
        must_reject(&s)?;
        structural += 1;
    }
    // Offset gap.
    if count >= 2 {
        let at = ts + SECTION_RECORD_BYTES + 2;
        let off = u64::from_le_bytes(valid[at..at + 8].try_into().unwrap());
        let mut g = valid.to_vec();
        g[at..at + 8].copy_from_slice(&(off + 1).to_le_bytes());
        reseal(&mut g);
        must_reject(&g)?;
        structural += 1;
    }
    // Object content-id forgery: flip a byte in section 1's payload and reseal.
    if count >= 2 {
        let rec = ts + SECTION_RECORD_BYTES;
        let off = u64::from_le_bytes(valid[rec + 2..rec + 10].try_into().unwrap()) as usize;
        let mut f = valid.to_vec();
        f[off] ^= 0x01;
        reseal(&mut f);
        must_reject(&f)?;
        structural += 1;
    }
    // Manifest entry-count lie.
    let manifest_off = u64::from_le_bytes(valid[ts + 2..ts + 10].try_into().unwrap()) as usize;
    let mut c = valid.to_vec();
    c[manifest_off..manifest_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    reseal(&mut c);
    must_reject(&c)?;
    structural += 1;
    // Session kind reorder: retag the first EVENT section as an OBJECT. The
    // counts live in the manifest, so the canonical kind order must be enforced
    // even though every payload digest has been recomputed.
    if event_count > 0 {
        let rec = ts + (1 + object_count) * SECTION_RECORD_BYTES;
        let mut s = valid.to_vec();
        s[rec] = SectionKind::Object as u8;
        reseal(&mut s);
        must_reject(&s)?;
        structural += 1;
        // A resealed malformed event payload (unknown class byte) must be
        // rejected by the session semantic validator, not merely by integrity.
        let off = u64::from_le_bytes(valid[rec + 2..rec + 10].try_into().unwrap()) as usize;
        let mut e = valid.to_vec();
        e[off + 8] = 0x7F;
        reseal(&mut e);
        must_reject(&e)?;
        structural += 1;
    }
    // A resealed checkpoint whose state length lies must be rejected too.
    let checkpoint_at = 1 + object_count + event_count;
    if !decoded.checkpoints().is_empty() {
        let rec = ts + checkpoint_at * SECTION_RECORD_BYTES;
        let off = u64::from_le_bytes(valid[rec + 2..rec + 10].try_into().unwrap()) as usize;
        let mut cp = valid.to_vec();
        // STATE_LEN is the four bytes at offset 12 in the checkpoint payload.
        cp[off + 12..off + 16].copy_from_slice(&u32::MAX.to_le_bytes());
        reseal(&mut cp);
        must_reject(&cp)?;
        structural += 1;
    }
    // Trailing byte before the digest, resealed.
    let mut t = valid.to_vec();
    let dig = t.len() - DIGEST_BYTES;
    t.insert(dig, 0xAB);
    reseal(&mut t);
    must_reject(&t)?;
    structural += 1;

    // Garbage.
    for candidate in [Vec::new(), vec![0u8; 4], (0..=255u8).collect::<Vec<u8>>()] {
        must_reject(&candidate)?;
        integrity += 1;
    }
    Ok((integrity, structural))
}

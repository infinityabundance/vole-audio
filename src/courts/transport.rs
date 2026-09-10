//! `court transport` — deterministic transport framing, integrity and recovery
//! (Phase N, contract §36/§37).
//!
//! Builds one frame stream over the frozen corpus — dependencies, object
//! containers, timed events, a checkpoint, a clock epoch advance and a closing
//! integrity attestation — then proves:
//!
//! * the stream round-trips and its integrity frame attests its body;
//! * the receiver classifies sequencing, duplicates, stale epochs, gaps, late
//!   events and checkpoint resync deterministically, and resolves dependencies;
//! * malformed streams (unknown kinds, oversized payloads, corruption, forged
//!   attestations, truncation, allocation bombs) are rejected;
//! * each xrun policy yields a distinct, reproducible recovery outcome.

use crate::corpus;
use crate::corpus::generate::{self, Spec};
use crate::error::{Error, Result};
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::fullobj::{self, FullSemantics};
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::SearchBudget;
use crate::object::id::ContentId;
use crate::status::Verdict;
use crate::transport::{
    Checkpoint, Frame, FrameKind, IntegrityReport, NO_MEDIA_FRAME, Outcome, TransportReceiver,
    decode_stream, encode_checkpoint, encode_event, encode_frame, integrity_frame, stream_report,
    verify_integrity_frame,
};
use crate::universe::clock::XrunPolicy;
use crate::universe::event::{Event, EventClass};
use crate::universe::time::MediaFrame;
use std::collections::BTreeMap;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const TRANSPORT_RESULT_SHA256: &str =
    "782a46b86d00f56b0bd2fe2b7fa4c38ba043e3e233a088177ddf843c951c926d";

/// The frozen transport-court protocol identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.transport.protocol.v1";

fn budget() -> SearchBudget {
    SearchBudget::default()
}

fn semantics_of(spec: &Spec) -> FullSemantics {
    match spec.semantics.period_frames() {
        Some(period_frames) => FullSemantics::Loop { period_frames },
        None => FullSemantics::OneShot,
    }
}

/// Everything the transport court's static projection binds.
struct Projection {
    integrity: IntegrityReport,
    outcomes: Vec<(&'static str, u64)>,
    objects: u64,
    dependencies: u64,
    late_events: u64,
    duplicates: u64,
    stale_epochs: u64,
    gaps: u64,
    resyncs: u64,
    unresolved_after: u64,
    rejected: u64,
    recovery: Vec<(&'static str, u32, i64, u64, u64)>,
}

fn static_projection(p: &Projection) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(PROTOCOL_SCHEMA.as_bytes());
    out.push(0);
    out.extend_from_slice(&p.integrity.digest);
    for v in [
        p.integrity.frames as u64,
        p.integrity.total_bytes,
        p.objects,
        p.dependencies,
        p.late_events,
        p.duplicates,
        p.stale_epochs,
        p.gaps,
        p.resyncs,
        p.unresolved_after,
        p.rejected,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for (name, count) in &p.outcomes {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(&count.to_le_bytes());
    }
    for (label, epoch, frame, skipped, inserted) in &p.recovery {
        out.extend_from_slice(label.as_bytes());
        out.push(0);
        out.extend_from_slice(&epoch.to_le_bytes());
        out.extend_from_slice(&frame.to_le_bytes());
        out.extend_from_slice(&skipped.to_le_bytes());
        out.extend_from_slice(&inserted.to_le_bytes());
    }
    out
}

fn outcome_label(o: &Outcome) -> &'static str {
    match o {
        Outcome::Object { .. } => "object",
        Outcome::Dependency { .. } => "dependency",
        Outcome::Event { late: false } => "event",
        Outcome::Event { late: true } => "late_event",
        Outcome::State => "state",
        Outcome::Clock { .. } => "clock",
        Outcome::Integrity => "integrity",
        Outcome::Checkpoint { resync: false } => "checkpoint",
        Outcome::Checkpoint { resync: true } => "checkpoint_resync",
        Outcome::StaleEpoch { .. } => "stale_epoch",
        Outcome::Duplicate { .. } => "duplicate",
        Outcome::SequenceGap { .. } => "sequence_gap",
    }
}

/// Run the court; writes an immutable receipt under `receipts/transport/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("transport");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("transport court failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court transport: FAILED_CORRECTNESS ({why})");
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

    // ---- authoring: a deterministic stream over the frozen corpus ----
    let mut seq = 0u64;
    let mut frames: Vec<Frame> = Vec::new();
    frames.push(Frame::new(FrameKind::Clock, 0, seq, NO_MEDIA_FRAME));
    seq += 1;

    let mut containers: Vec<(String, Vec<u8>, ContentId)> = Vec::new();
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
        containers.push((
            o.id.clone(),
            obj.bytes.clone(),
            ContentId::from_bytes(obj.sha256()),
        ));
    }

    // Declare every dependency first, so they are unresolved until the objects.
    for (_, _, cid) in &containers {
        frames.push(
            Frame::new(FrameKind::Dependency, 0, seq, NO_MEDIA_FRAME)
                .with_payload(cid.to_bytes().to_vec()),
        );
        seq += 1;
    }
    // Then the objects that satisfy them.
    for (i, (_, bytes, _)) in containers.iter().enumerate() {
        let media = (i as i64) * 100;
        frames.push(Frame::new(FrameKind::Object, 0, seq, media).with_payload(bytes.clone()));
        seq += 1;
    }
    // Timed events after the objects.
    for (i, _) in containers.iter().enumerate() {
        let at = (i as i64) * 100 + 50;
        let e = Event::new(MediaFrame::new(at), EventClass::Param, seq, 1);
        frames.push(Frame::new(FrameKind::Event, 0, seq, at).with_payload(encode_event(&e)));
        seq += 1;
    }
    // A checkpoint anchoring the end of the stream.
    let anchor = (containers.len() as i64) * 100;
    let state = Sha256::digest(format!("anchor:{anchor}").as_bytes()).to_vec();
    let cp = Checkpoint::new(0, anchor, state);
    frames.push(
        Frame::new(FrameKind::Checkpoint, 0, seq, anchor).with_payload(encode_checkpoint(&cp)?),
    );
    seq += 1;

    // Encode the body, then append the integrity attestation.
    let mut body = Vec::new();
    for f in &frames {
        body.extend_from_slice(&encode_frame(f)?);
    }
    let attestation = integrity_frame(0, seq, &body);
    let mut stream = body.clone();
    stream.extend_from_slice(&encode_frame(&attestation)?);

    // Integrity: decode + attestation + report.
    let decoded = match decode_stream(&stream) {
        Ok(d) => d,
        Err(e) => return fail(&format!("the encoded stream does not decode: {e}")),
    };
    if decoded.len() != frames.len() + 1 {
        return fail("stream frame count mismatch after round-trip");
    }
    if let Err(e) = verify_integrity_frame(&stream) {
        return fail(&format!("the stream integrity frame does not verify: {e}"));
    }
    let integrity = match stream_report(&stream) {
        Ok(r) => r,
        Err(e) => return fail(&format!("stream report failed: {e}")),
    };

    // ---- receiver: ordered, bounded processing ----
    let mut receiver = TransportReceiver::new();
    let mut counts: BTreeMap<&'static str, u64> = BTreeMap::new();
    for f in &decoded {
        let outcome = receiver.push(f)?;
        *counts.entry(outcome_label(&outcome)).or_insert(0) += 1;
    }
    let stats = receiver.stats();
    let unresolved_after = receiver.unresolved_dependencies().len() as u64;
    if unresolved_after != 0 {
        return fail("a declared dependency was never resolved");
    }

    // ---- hostile classification battery ----
    let rejected = hostile_battery(&frames, &stream)?;

    // ---- recovery policies ----
    let recovery = recovery_evidence();

    let projection = Projection {
        integrity: integrity.clone(),
        outcomes: counts.iter().map(|(k, v)| (*k, *v)).collect(),
        objects: stats.objects,
        dependencies: stats.dependencies,
        late_events: stats.late_events,
        duplicates: stats.duplicates,
        stale_epochs: stats.stale_epochs,
        gaps: stats.gaps,
        resyncs: stats.resyncs,
        unresolved_after,
        rejected,
        recovery: recovery.clone(),
    };
    let result_hex = hex(&Sha256::digest(&static_projection(&projection)));
    if TRANSPORT_RESULT_SHA256.is_empty() {
        eprintln!("court transport: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != TRANSPORT_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {TRANSPORT_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    let verdict = Verdict::Supported;
    let mut builder = ReceiptBuilder::new("transport");
    builder
        .result(verdict)
        .result_detail(format!(
            "{} frames over {} objects: {} objects resolved, {} dependencies declared, {} events \
             ({} late), {} duplicates, {} stale epochs, {} sequence gaps, {} resyncs; integrity \
             digest {}; {} hostile candidates rejected; recovery outcomes recorded for all policies; \
             result sha256 {result_hex}",
            integrity.frames,
            containers.len(),
            stats.objects,
            stats.dependencies,
            stats.events,
            stats.late_events,
            stats.duplicates,
            stats.stale_epochs,
            stats.gaps,
            stats.resyncs,
            hex(&integrity.digest),
            rejected,
        ))
        .params(CourtParams {
            universe: Some(manifest_in.universe.clone()),
            profile: Some(format!("{} + transport.v1", manifest_in.profile)),
            backend: Some("deterministic transport framing + receiver".into()),
            content_kind: Some("flagship corpus (frozen before results): transport".into()),
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
                "kinds": ["object", "event", "state", "checkpoint", "dependency", "clock", "integrity"],
                "payloads": "object containers, canonical event payloads, checkpoints; procedural/state \
                             information, never mandatory PCM",
                "ordering": "sequence monotonic within an epoch; only a CLOCK frame may advance the epoch",
                "integrity": "every frame carries a digest; the final integrity frame attests the body",
                "hostile": "unknown kind, unknown flags, oversized payload, corruption, truncation, \
                            forged attestation, duplicate, stale epoch, gap and a late event",
            }),
        )
        .extra(
            "integrity",
            serde_json::json!({
                "frames": integrity.frames,
                "total_bytes": integrity.total_bytes,
                "digest": hex(&integrity.digest),
                "epochs": integrity.epochs,
                "has_integrity_frame": integrity.has_integrity_frame,
            }),
        )
        .extra(
            "receiver",
            serde_json::json!({
                "outcomes": counts,
                "objects": stats.objects,
                "dependencies": stats.dependencies,
                "late_events": stats.late_events,
                "duplicates": stats.duplicates,
                "stale_epochs": stats.stale_epochs,
                "gaps": stats.gaps,
                "missing_sequence_frames": stats.missing_sequence_frames,
                "resyncs": stats.resyncs,
                "unresolved_dependencies_after": unresolved_after,
            }),
        )
        .extra(
            "recovery",
            serde_json::json!(recovery
                .iter()
                .map(|(label, epoch, frame, skipped, inserted)| serde_json::json!({
                    "policy": label,
                    "epoch": epoch,
                    "logical_frame": frame,
                    "skipped_frames": skipped,
                    "inserted_frames": inserted,
                }))
                .collect::<Vec<_>>()),
        )
        .limitation(
            "this is a deterministic framing/state-machine court, not a network stack. It carries \
             procedural/state information; the literal fallback rides as an object payload",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court transport: {verdict}");
    println!(
        "  frames: {} | objects: {} | late: {} | rejected: {rejected}",
        integrity.frames, stats.objects, stats.late_events
    );
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

/// Deterministic classification battery. Returns the number of candidates that
/// were correctly rejected or classified.
fn hostile_battery(frames: &[Frame], stream: &[u8]) -> Result<u64> {
    let mut handled = 0u64;
    let expect = |cond: bool, what: &str| -> Result<()> {
        if cond {
            Ok(())
        } else {
            Err(Error::internal(format!(
                "hostile classification failed: {what}"
            )))
        }
    };

    // Unknown frame kind.
    let mut unknown = encode_frame(&Frame::new(FrameKind::State, 0, 0, NO_MEDIA_FRAME))?;
    unknown[0] = 0x7F;
    expect(decode_stream(&unknown).is_err(), "unknown kind")?;
    handled += 1;
    // Unknown flags.
    let mut flags = encode_frame(&Frame::new(FrameKind::State, 0, 0, NO_MEDIA_FRAME))?;
    flags[1] = 0x80;
    expect(decode_stream(&flags).is_err(), "unknown flags")?;
    handled += 1;
    // Oversized payload length (allocation bomb).
    let mut bomb = encode_frame(&Frame::new(FrameKind::State, 0, 0, NO_MEDIA_FRAME))?;
    bomb[22..26].copy_from_slice(&u32::MAX.to_le_bytes());
    expect(decode_stream(&bomb).is_err(), "oversized payload")?;
    handled += 1;
    // Corruption of the stream body.
    let mut corrupt = stream.to_vec();
    corrupt[0] ^= 0xFF;
    expect(decode_stream(&corrupt).is_err(), "corruption")?;
    handled += 1;
    // Truncation.
    expect(
        decode_stream(&stream[..stream.len() - 1]).is_err(),
        "truncation",
    )?;
    handled += 1;
    // Forged integrity attestation.
    let mut forged_body = Vec::new();
    for f in frames {
        forged_body.extend_from_slice(&encode_frame(f)?);
    }
    let mut bad = forged_body.clone();
    let mut forged = integrity_frame(0, 999, &forged_body);
    forged.payload = vec![0u8; 32];
    bad.extend_from_slice(&encode_frame(&forged)?);
    expect(
        crate::transport::verify_integrity_frame(&bad).is_err(),
        "forged attestation",
    )?;
    handled += 1;
    // Duplicate, stale epoch, gap and late event are classified by the receiver.
    let mut r = TransportReceiver::new();
    let f0 = Frame::new(FrameKind::Clock, 1, 0, NO_MEDIA_FRAME);
    r.push(&f0)?;
    expect(
        matches!(
            r.push(&Frame::new(FrameKind::State, 0, 1, NO_MEDIA_FRAME))?,
            Outcome::StaleEpoch { .. }
        ),
        "stale epoch",
    )?;
    handled += 1;
    let mut r2 = TransportReceiver::new();
    r2.push(&Frame::new(FrameKind::State, 0, 0, NO_MEDIA_FRAME))?;
    expect(
        matches!(
            r2.push(&Frame::new(FrameKind::State, 0, 0, NO_MEDIA_FRAME))?,
            Outcome::Duplicate { .. }
        ),
        "duplicate",
    )?;
    handled += 1;
    let mut r3 = TransportReceiver::new();
    expect(
        matches!(
            r3.push(&Frame::new(FrameKind::State, 0, 5, NO_MEDIA_FRAME))?,
            Outcome::SequenceGap { .. }
        ),
        "sequence gap",
    )?;
    handled += 1;
    let mut r4 = TransportReceiver::new();
    r4.advance_cursor(1_000);
    let e = Event::new(MediaFrame::new(10), EventClass::Param, 0, 1);
    let late = Frame::new(FrameKind::Event, 0, 0, 10).with_payload(encode_event(&e));
    expect(
        matches!(r4.push(&late)?, Outcome::Event { late: true }),
        "late event",
    )?;
    handled += 1;
    Ok(handled)
}

/// The three recovery policies, on identical clocks, for the receipt.
fn recovery_evidence() -> Vec<(&'static str, u32, i64, u64, u64)> {
    let mut out = Vec::new();
    for policy in [
        XrunPolicy::PreserveTimeline,
        XrunPolicy::Discontinuity,
        XrunPolicy::RestartEpoch,
    ] {
        let mut r = TransportReceiver::new();
        r.advance_cursor(1_000);
        let o = r.on_xrun(policy, 128);
        out.push((
            policy.label(),
            o.epoch,
            o.logical_frame,
            o.skipped_frames,
            o.inserted_frames,
        ));
    }
    out
}

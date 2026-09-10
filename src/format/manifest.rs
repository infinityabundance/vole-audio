//! Reproducible manifests (Phase N, contract §52).
//!
//! The archive's binary MANIFEST is the normative catalogue. This module adds a
//! **canonical text rendering** of the same content so a receipt or a human can
//! compare two archives byte-for-byte: the text is deterministic, sorted and
//! newline-delimited, and its SHA-256 is the manifest digest.
//!
//! ```text
//! vole.audio.manifest.v1
//! profile <tag>
//! universe <tag>
//! entry <name> <representation> <channels> <rate> <frames> <content_id_hex>
//! events <count>
//! event <index> <payload_sha256_hex>
//! checkpoints <count>
//! checkpoint <index> <payload_sha256_hex>
//! dependencies <count>
//! dependency <content_id_hex>
//! ```

use crate::hash::sha256::{Sha256, hex};

use super::archive::{ArchiveManifest, DecodedArchive};

/// One manifest row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestEntry {
    pub name: String,
    pub representation: u8,
    pub channels: u8,
    pub sample_rate_hz: u32,
    pub frames: u64,
    pub content_id: [u8; 32],
}

/// A reproducible manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub profile: String,
    pub universe: String,
    pub entries: Vec<ManifestEntry>,
    /// SHA-256 of each canonical event payload, in frozen section order.
    pub events: Vec<[u8; 32]>,
    /// SHA-256 of each canonical checkpoint payload, in frozen section order.
    pub checkpoints: Vec<[u8; 32]>,
    /// Dependency content ids, in frozen section order.
    pub dependencies: Vec<[u8; 32]>,
}

impl Manifest {
    /// Build from a validated archive.
    pub fn from_archive(archive: &DecodedArchive) -> Manifest {
        Manifest {
            profile: archive.profile().to_string(),
            universe: archive.universe().to_string(),
            entries: archive
                .manifest()
                .entries
                .iter()
                .map(|e| ManifestEntry {
                    name: e.name.clone(),
                    representation: e.representation,
                    channels: e.channels,
                    sample_rate_hz: e.sample_rate_hz,
                    frames: e.frames,
                    content_id: e.content_id.to_bytes(),
                })
                .collect(),
            events: archive
                .session()
                .events
                .iter()
                .map(|p| Sha256::digest(p))
                .collect(),
            checkpoints: archive
                .session()
                .checkpoints
                .iter()
                .map(|p| Sha256::digest(p))
                .collect(),
            dependencies: archive
                .session()
                .dependencies
                .iter()
                .map(|c| c.to_bytes())
                .collect(),
        }
    }

    /// Build from an archive manifest.
    pub fn from_manifest(m: &ArchiveManifest) -> Manifest {
        Manifest {
            profile: m.profile.clone(),
            universe: m.universe.clone(),
            entries: m
                .entries
                .iter()
                .map(|e| ManifestEntry {
                    name: e.name.clone(),
                    representation: e.representation,
                    channels: e.channels,
                    sample_rate_hz: e.sample_rate_hz,
                    frames: e.frames,
                    content_id: e.content_id.to_bytes(),
                })
                .collect(),
            events: m.session.events.iter().map(|p| Sha256::digest(p)).collect(),
            checkpoints: m
                .session
                .checkpoints
                .iter()
                .map(|p| Sha256::digest(p))
                .collect(),
            dependencies: m
                .session
                .dependencies
                .iter()
                .map(|c| c.to_bytes())
                .collect(),
        }
    }

    /// Canonical, sorted, deterministic text form.
    pub fn canonical_text(&self) -> String {
        let mut entries = self.entries.clone();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        let mut out = String::from("vole.audio.manifest.v1\n");
        out.push_str(&format!("profile {}\n", self.profile));
        out.push_str(&format!("universe {}\n", self.universe));
        for e in &entries {
            out.push_str(&format!(
                "entry {} {} {} {} {} {hex}\n",
                e.name,
                e.representation,
                e.channels,
                e.sample_rate_hz,
                e.frames,
                hex = hex(&e.content_id),
            ));
        }
        out.push_str(&format!("events {}\n", self.events.len()));
        for (i, d) in self.events.iter().enumerate() {
            out.push_str(&format!("event {i} {}\n", hex(d)));
        }
        out.push_str(&format!("checkpoints {}\n", self.checkpoints.len()));
        for (i, d) in self.checkpoints.iter().enumerate() {
            out.push_str(&format!("checkpoint {i} {}\n", hex(d)));
        }
        out.push_str(&format!("dependencies {}\n", self.dependencies.len()));
        for d in &self.dependencies {
            out.push_str(&format!("dependency {}\n", hex(d)));
        }
        out
    }

    /// SHA-256 of the canonical text — the reproducible manifest digest.
    pub fn digest(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_text().as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::descriptor::{ObjectDescriptor, Representation};
    use crate::object::simple::Constant;
    use crate::object::{ObjectData, canonical_content_id, canonical_object_bytes};
    use crate::universe::layout::Layout;

    fn archive() -> Vec<u8> {
        let make = |name: &str, level: i32| {
            let d =
                ObjectDescriptor::new(Representation::Constant, 64, Layout::Mono, None).unwrap();
            let data = ObjectData::Constant(Constant { level });
            let bytes = canonical_object_bytes(&d, &data);
            let cid = canonical_content_id(&d, &data);
            (
                super::super::archive::ArchiveEntry {
                    name: name.to_string(),
                    representation: d.representation.tag(),
                    channels: 1,
                    sample_rate_hz: 48_000,
                    frames: 64,
                    content_id: cid,
                },
                bytes,
            )
        };
        let (a, ab) = make("beta", 1);
        let (b, bb) = make("alpha", 2);
        let event = crate::transport::encode_event(&crate::universe::event::Event::new(
            crate::universe::time::MediaFrame::new(64),
            crate::universe::event::EventClass::Param,
            3,
            0x0011,
        ));
        let checkpoint = crate::transport::encode_checkpoint(&crate::transport::Checkpoint::new(
            0,
            128,
            vec![0xAA, 0xBB],
        ))
        .unwrap();
        let manifest = ArchiveManifest {
            profile: "u1/v1".into(),
            universe: "vole.audio.u1".into(),
            entries: vec![b, a], // already sorted ("alpha" < "beta")
            session: super::super::archive::ArchiveSession {
                events: vec![event],
                checkpoints: vec![checkpoint],
                dependencies: vec![crate::object::id::ContentId::from_bytes([9u8; 32])],
            },
        };
        super::super::archive::encode_archive(&manifest, &[bb, ab]).unwrap()
    }

    #[test]
    fn manifest_text_is_sorted_and_reproducible() {
        let bytes = archive();
        let decoded = super::super::archive::decode_archive(&bytes).unwrap();
        let m = Manifest::from_archive(&decoded);
        let text = m.canonical_text();
        let alpha = text.find("entry alpha").unwrap();
        let beta = text.find("entry beta").unwrap();
        assert!(alpha < beta);
        assert!(text.starts_with("vole.audio.manifest.v1\n"));
        // The session sections are covered by the manifest text too.
        assert!(text.contains("events 1\nevent 0 "));
        assert!(text.contains("checkpoints 1\ncheckpoint 0 "));
        assert!(text.contains(&format!("dependencies 1\ndependency {}", hex(&[9u8; 32]))));
        // Same content -> same text -> same digest.
        let again = Manifest::from_archive(&decoded);
        assert_eq!(text, again.canonical_text());
        assert_eq!(m.digest(), again.digest());
        // Shuffling input entries does not change the canonical text.
        let mut shuffled = m.clone();
        shuffled.entries.reverse();
        assert_eq!(shuffled.canonical_text(), text);
    }
}

//! Canonical archive container (`.volea`) — Phase N.
//!
//! The media format is explicit binary, little-endian and canonical. It is
//! **not** serde/bincode-derived layout: every field is written and read by
//! hand, every parse is length-checked, and no length is converted to `usize`
//! without a checked conversion.
//!
//! ```text
//! Archive :=
//!     MAGIC            12 bytes   b"vole.archive"
//!     VERSION          u8         = 1
//!     PROFILE_LEN      u8         1..=32
//!     PROFILE          bytes      e.g. b"u1/v1"
//!     UNIVERSE_LEN     u8         1..=32
//!     UNIVERSE         bytes      e.g. b"vole.audio.u1"
//!     SECTION_COUNT    u32        1..=MAX_SECTIONS
//!     SECTION_TABLE    SECTION_COUNT × SectionRecord
//!     SECTIONS         payloads, contiguous, in table order
//!     ARCHIVE_DIGEST   32 bytes   SHA-256 over everything before it
//!
//! SectionRecord :=
//!     KIND      u8        SectionKind
//!     RESERVED  u8        = 0 (must be zero)
//!     OFFSET    u64       absolute offset of the payload
//!     LENGTH    u64       payload length
//!     SHA256    32        digest of the payload
//! ```
//!
//! Canonical layout rules the decoder enforces: section 0 is the MANIFEST, the
//! remaining sections are OBJECTs in manifest-entry order, payload offsets are
//! contiguous and start immediately after the table, there are no trailing
//! bytes, and no unknown section kind is accepted in v1.
//!
//! ```text
//! Manifest :=
//!     ENTRY_COUNT  u32        1..=MAX_OBJECTS
//!     ENTRY_COUNT × Entry     ordered by name, strictly ascending
//!
//! Entry :=
//!     NAME_LEN         u8      1..=MAX_NAME
//!     NAME             bytes
//!     REPRESENTATION   u8      the U1 representation tag
//!     CHANNELS         u8
//!     SAMPLE_RATE_HZ   u32
//!     FRAMES           u64
//!     CONTENT_ID       32      SHA-256 of the canonical object bytes
//! ```
//!
//! The OBJECT payload **is** the canonical U1 object byte form, so an archive
//! entry's `CONTENT_ID` is exactly [`crate::object::canonical_content_id`] and
//! the decoder verifies it against the payload rather than trusting the table.

use crate::error::{Error, Result};
use crate::hash::sha256::{Sha256, hex};
use crate::object::id::ContentId;

/// Fixed archive magic.
pub const MAGIC: [u8; 12] = *b"vole.archive";
/// Archive format version.
pub const VERSION: u8 = 1;
/// Trailing archive digest length.
pub const DIGEST_BYTES: usize = 32;
/// Encoded size of one section record.
pub const SECTION_RECORD_BYTES: usize = 1 + 1 + 8 + 8 + 32;
/// Upper bound on sections (manifest + objects) in one archive.
pub const MAX_SECTIONS: u32 = 65_536;
/// Upper bound on catalogue names.
pub const MAX_NAME: usize = 64;
/// Upper bound on the profile/universe tags.
pub const MAX_TAG: usize = 32;

/// Section kinds understood by v1. Unknown kinds are rejected, never skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SectionKind {
    Manifest = 1,
    Object = 2,
    Integrity = 3,
    Event = 4,
    State = 5,
    Checkpoint = 6,
    Dependency = 7,
    Clock = 8,
}

impl SectionKind {
    fn from_u8(v: u8) -> Result<SectionKind> {
        Ok(match v {
            1 => SectionKind::Manifest,
            2 => SectionKind::Object,
            3 => SectionKind::Integrity,
            4 => SectionKind::Event,
            5 => SectionKind::State,
            6 => SectionKind::Checkpoint,
            7 => SectionKind::Dependency,
            8 => SectionKind::Clock,
            other => {
                return Err(Error::new(
                    crate::error::Kind::Unsupported,
                    format!("unknown archive section kind {other}"),
                ));
            }
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            SectionKind::Manifest => "manifest",
            SectionKind::Object => "object",
            SectionKind::Integrity => "integrity",
            SectionKind::Event => "event",
            SectionKind::State => "state",
            SectionKind::Checkpoint => "checkpoint",
            SectionKind::Dependency => "dependency",
            SectionKind::Clock => "clock",
        }
    }
}

/// One catalogue entry: identity and geometry of an archived object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveEntry {
    pub name: String,
    pub representation: u8,
    pub channels: u8,
    pub sample_rate_hz: u32,
    pub frames: u64,
    pub content_id: ContentId,
}

impl ArchiveEntry {
    /// Validate the entry in isolation (bounded, non-degenerate geometry).
    fn validate(&self) -> Result<()> {
        if self.name.is_empty() || self.name.len() > MAX_NAME {
            return Err(Error::malformed("archive entry name length out of range"));
        }
        if !self
            .name
            .bytes()
            .all(|b| b.is_ascii_graphic() && b != b'/' && b != b'\\')
        {
            return Err(Error::malformed(
                "archive entry name must be printable ASCII without separators",
            ));
        }
        if self.channels == 0 {
            return Err(Error::malformed("archive entry has zero channels"));
        }
        if self.sample_rate_hz == 0 {
            return Err(Error::malformed("archive entry has zero sample rate"));
        }
        Ok(())
    }
}

/// The archive catalogue (the MANIFEST section).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveManifest {
    pub profile: String,
    pub universe: String,
    pub entries: Vec<ArchiveEntry>,
}

impl ArchiveManifest {
    /// Entry-level validation, independent of the header tags (so a parsed
    /// manifest can be validated before its profile/universe are attached).
    fn validate_entries(&self) -> Result<()> {
        if self.entries.is_empty() {
            return Err(Error::malformed(
                "an archive must contain at least one object",
            ));
        }
        if self.entries.len() > MAX_SECTIONS as usize {
            return Err(Error::limit("archive entry count exceeds the bound"));
        }
        // Canonical order: strictly ascending by name, so the byte form is unique.
        for w in self.entries.windows(2) {
            if w[0].name >= w[1].name {
                return Err(Error::malformed(
                    "archive entries must be strictly ascending by name",
                ));
            }
        }
        for e in &self.entries {
            e.validate()?;
        }
        Ok(())
    }

    fn canonical_payload(&self) -> Result<Vec<u8>> {
        if self.profile.is_empty() || self.profile.len() > MAX_TAG {
            return Err(Error::malformed("archive profile tag length out of range"));
        }
        if self.universe.is_empty() || self.universe.len() > MAX_TAG {
            return Err(Error::malformed("archive universe tag length out of range"));
        }
        self.validate_entries()?;
        let mut out = Vec::new();
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            out.push(e.name.len() as u8);
            out.extend_from_slice(e.name.as_bytes());
            out.push(e.representation);
            out.push(e.channels);
            out.extend_from_slice(&e.sample_rate_hz.to_le_bytes());
            out.extend_from_slice(&e.frames.to_le_bytes());
            out.extend_from_slice(&e.content_id.to_bytes());
        }
        Ok(out)
    }

    fn parse(bytes: &[u8]) -> Result<ArchiveManifest> {
        let mut r = Reader::new(bytes);
        let count = r.u32()? as usize;
        if count == 0 {
            return Err(Error::malformed("empty archive manifest"));
        }
        if count > MAX_SECTIONS as usize {
            return Err(Error::limit(
                "archive manifest entry count exceeds the bound",
            ));
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let name_len = r.u8()? as usize;
            if name_len == 0 || name_len > MAX_NAME {
                return Err(Error::malformed("archive entry name length out of range"));
            }
            let name = r.bytes(name_len)?;
            let name = std::str::from_utf8(name)
                .map_err(|_| Error::malformed("archive entry name is not UTF-8"))?
                .to_string();
            let representation = r.u8()?;
            let channels = r.u8()?;
            let sample_rate_hz = r.u32()?;
            let frames = r.u64()?;
            let content_id = ContentId::from_bytes(r.array32()?);
            entries.push(ArchiveEntry {
                name,
                representation,
                channels,
                sample_rate_hz,
                frames,
                content_id,
            });
        }
        if r.remaining() != 0 {
            return Err(Error::malformed("trailing bytes in the archive manifest"));
        }
        let manifest = ArchiveManifest {
            profile: String::new(),
            universe: String::new(),
            entries,
        };
        manifest.validate_entries()?;
        Ok(manifest)
    }
}

/// A decoded, fully validated archive.
#[derive(Debug, Clone)]
pub struct DecodedArchive {
    profile: String,
    universe: String,
    manifest: ArchiveManifest,
    section_kinds: Vec<SectionKind>,
    payloads: Vec<std::ops::Range<usize>>,
    archive_digest: [u8; 32],
}

impl DecodedArchive {
    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn universe(&self) -> &str {
        &self.universe
    }

    pub fn manifest(&self) -> &ArchiveManifest {
        &self.manifest
    }

    pub fn archive_digest(&self) -> [u8; 32] {
        self.archive_digest
    }

    pub fn objects(&self) -> usize {
        self.manifest.entries.len()
    }

    pub fn section_kinds(&self) -> &[SectionKind] {
        &self.section_kinds
    }

    /// Canonical object bytes of entry `i` (already integrity-verified).
    pub fn object_bytes<'a>(&self, bytes: &'a [u8], i: usize) -> Result<&'a [u8]> {
        let range = self
            .payloads
            .get(i + 1)
            .ok_or_else(|| Error::limit("archive object index out of range"))?
            .clone();
        Ok(&bytes[range])
    }

    pub fn manifest_bytes<'a>(&self, bytes: &'a [u8]) -> Result<&'a [u8]> {
        let range = self
            .payloads
            .first()
            .ok_or_else(|| Error::malformed("archive has no manifest section"))?
            .clone();
        Ok(&bytes[range])
    }
}

/// Encode a canonical archive. `objects[i]` must be the canonical U1 object
/// bytes for `manifest.entries[i]`.
pub fn encode_archive(manifest: &ArchiveManifest, objects: &[Vec<u8>]) -> Result<Vec<u8>> {
    if objects.len() != manifest.entries.len() {
        return Err(Error::malformed(
            "archive object count does not match the manifest",
        ));
    }
    let manifest_payload = manifest.canonical_payload()?;

    // Bind every entry's content id to the bytes actually stored.
    for (entry, bytes) in manifest.entries.iter().zip(objects) {
        let digest = ContentId::from_bytes(Sha256::digest(bytes));
        if digest != entry.content_id {
            return Err(Error::new(
                crate::error::Kind::Integrity,
                format!(
                    "archive entry '{}' content id does not match its bytes",
                    entry.name
                ),
            ));
        }
    }

    let payloads: Vec<&[u8]> = std::iter::once(manifest_payload.as_slice())
        .chain(objects.iter().map(Vec::as_slice))
        .collect();
    let section_count = u32::try_from(payloads.len())
        .map_err(|_| Error::limit("archive section count overflows"))?;
    if section_count > MAX_SECTIONS {
        return Err(Error::limit("archive section count exceeds the bound"));
    }

    let header_len = MAGIC.len() + 1 + 1 + manifest.profile.len() + 1 + manifest.universe.len() + 4;
    let table_len = SECTION_RECORD_BYTES * payloads.len();

    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(manifest.profile.len() as u8);
    out.extend_from_slice(manifest.profile.as_bytes());
    out.push(manifest.universe.len() as u8);
    out.extend_from_slice(manifest.universe.as_bytes());
    out.extend_from_slice(&section_count.to_le_bytes());

    let mut offset = header_len + table_len;
    for (i, payload) in payloads.iter().enumerate() {
        let kind = if i == 0 {
            SectionKind::Manifest
        } else {
            SectionKind::Object
        };
        out.push(kind as u8);
        out.push(0);
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&Sha256::digest(payload));
        offset = offset
            .checked_add(payload.len())
            .ok_or_else(|| Error::limit("archive size overflows"))?;
    }
    for payload in &payloads {
        out.extend_from_slice(payload);
    }
    let digest = Sha256::digest(&out);
    out.extend_from_slice(&digest);
    Ok(out)
}

/// Decode and fully validate an archive.
pub fn decode_archive(bytes: &[u8]) -> Result<DecodedArchive> {
    if bytes.len() < MAGIC.len() + 1 + 1 + 1 + 1 + 4 + DIGEST_BYTES {
        return Err(Error::malformed("archive is too short"));
    }
    if bytes[..MAGIC.len()] != MAGIC {
        return Err(Error::malformed("archive magic mismatch"));
    }
    let version = bytes[MAGIC.len()];
    if version != VERSION {
        return Err(Error::new(
            crate::error::Kind::Unsupported,
            format!("unsupported archive version {version}"),
        ));
    }
    let mut p = MAGIC.len() + 1;

    let profile_len = bytes[p] as usize;
    p += 1;
    if profile_len == 0 || profile_len > MAX_TAG {
        return Err(Error::malformed("archive profile tag length out of range"));
    }
    let profile = read_tag(bytes, p, profile_len)?;
    p += profile_len;

    let universe_len = usize::from(bytes[p]);
    p += 1;
    if universe_len == 0 || universe_len > MAX_TAG {
        return Err(Error::malformed("archive universe tag length out of range"));
    }
    let universe = read_tag(bytes, p, universe_len)?;
    p += universe_len;

    let section_count = u32::from_le_bytes(
        bytes[p..p + 4]
            .try_into()
            .map_err(|_| Error::malformed("archive section count is truncated"))?,
    ) as usize;
    p += 4;
    if section_count == 0 || section_count > MAX_SECTIONS as usize {
        return Err(Error::limit("archive section count out of range"));
    }
    let table_bytes = section_count
        .checked_mul(SECTION_RECORD_BYTES)
        .ok_or_else(|| Error::limit("archive section table overflows"))?;
    if p.checked_add(table_bytes).is_none() || p + table_bytes > bytes.len() {
        return Err(Error::malformed("archive section table is truncated"));
    }

    let mut kinds = Vec::with_capacity(section_count);
    let mut ranges = Vec::with_capacity(section_count);
    let mut expected_offset = p + table_bytes;
    for i in 0..section_count {
        let rec = p + i * SECTION_RECORD_BYTES;
        let kind = SectionKind::from_u8(bytes[rec])?;
        if bytes[rec + 1] != 0 {
            return Err(Error::malformed(
                "archive section reserved byte is not zero",
            ));
        }
        let offset = u64::from_le_bytes(bytes[rec + 2..rec + 10].try_into().unwrap());
        let length = u64::from_le_bytes(bytes[rec + 10..rec + 18].try_into().unwrap());
        let digest: [u8; 32] = bytes[rec + 18..rec + 50].try_into().unwrap();

        // Canonical layout: contiguous, table order, no gaps or aliases.
        if offset != expected_offset as u64 {
            return Err(Error::malformed(
                "archive payloads must be contiguous and follow the table",
            ));
        }
        let length = usize::try_from(length)
            .map_err(|_| Error::limit("archive payload length exceeds host usize"))?;
        let offset = usize::try_from(offset)
            .map_err(|_| Error::limit("archive payload offset exceeds host usize"))?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| Error::limit("archive payload range overflows"))?;
        if end > bytes.len() - DIGEST_BYTES {
            return Err(Error::malformed("archive payload runs past the body"));
        }
        if Sha256::digest(&bytes[offset..end]) != digest {
            return Err(Error::new(
                crate::error::Kind::Integrity,
                format!("archive {} payload digest mismatch", kind.name()),
            ));
        }
        // Section kinds: manifest first, objects thereafter, in v1.
        let expected_kind = if i == 0 {
            SectionKind::Manifest
        } else {
            SectionKind::Object
        };
        if kind != expected_kind {
            return Err(Error::malformed(
                "archive sections must be the manifest followed by objects",
            ));
        }
        kinds.push(kind);
        ranges.push(offset..end);
        expected_offset = end;
    }
    // No trailing bytes between the last payload and the archive digest.
    if expected_offset != bytes.len() - DIGEST_BYTES {
        return Err(Error::malformed("archive has unreferenced trailing bytes"));
    }
    let body = &bytes[..bytes.len() - DIGEST_BYTES];
    let stated: [u8; 32] = bytes[bytes.len() - DIGEST_BYTES..].try_into().unwrap();
    if Sha256::digest(body) != stated {
        return Err(Error::new(
            crate::error::Kind::Integrity,
            "archive digest mismatch",
        ));
    }

    let manifest_payload = &bytes[ranges[0].clone()];
    let parsed = ArchiveManifest::parse(manifest_payload)?;
    if parsed.entries.len() + 1 != section_count {
        return Err(Error::malformed(
            "archive section count does not match the manifest entry count",
        ));
    }
    // The manifest's identity column must agree with the stored payloads.
    for (i, entry) in parsed.entries.iter().enumerate() {
        let payload = &bytes[ranges[i + 1].clone()];
        if ContentId::from_bytes(Sha256::digest(payload)) != entry.content_id {
            return Err(Error::new(
                crate::error::Kind::Integrity,
                format!(
                    "archive entry '{}' content id does not match its bytes",
                    entry.name
                ),
            ));
        }
    }
    let manifest = ArchiveManifest {
        profile,
        universe,
        entries: parsed.entries,
    };
    Ok(DecodedArchive {
        profile: manifest.profile.clone(),
        universe: manifest.universe.clone(),
        manifest,
        section_kinds: kinds,
        payloads: ranges,
        archive_digest: stated,
    })
}

/// Human-readable digest of an archive, for receipts.
pub fn archive_digest_hex(bytes: &[u8]) -> String {
    hex(&decode_archive(bytes)
        .map(|d| d.archive_digest())
        .unwrap_or([0u8; 32]))
}

fn read_tag(bytes: &[u8], at: usize, len: usize) -> Result<String> {
    let end = at
        .checked_add(len)
        .ok_or_else(|| Error::limit("archive tag range overflows"))?;
    let slice = bytes
        .get(at..end)
        .ok_or_else(|| Error::malformed("archive tag is truncated"))?;
    let s = std::str::from_utf8(slice).map_err(|_| Error::malformed("archive tag is not UTF-8"))?;
    Ok(s.to_string())
}

/// A checked little-endian reader: every advance is bounds-checked and every
/// length is converted with `try_from`, never cast.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::limit("archive manifest read overflows"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::malformed("archive manifest is truncated"))?;
        self.pos = end;
        Ok(slice)
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn array32(&mut self) -> Result<[u8; 32]> {
        Ok(self.take(32)?.try_into().unwrap())
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectData;
    use crate::object::descriptor::{ObjectDescriptor, Representation};
    use crate::object::simple::Constant;
    use crate::universe::layout::Layout;

    fn constant(name: &str, level: i32, frames: u64) -> (ArchiveEntry, Vec<u8>) {
        let d = ObjectDescriptor::new(Representation::Constant, frames, Layout::Mono, None)
            .expect("descriptor");
        let data = ObjectData::Constant(Constant { level });
        let bytes = crate::object::canonical_object_bytes(&d, &data);
        let content_id = crate::object::canonical_content_id(&d, &data);
        (
            ArchiveEntry {
                name: name.to_string(),
                representation: d.representation.tag(),
                channels: 1,
                sample_rate_hz: 48_000,
                frames,
                content_id,
            },
            bytes,
        )
    }

    fn valid_archive() -> (ArchiveManifest, Vec<Vec<u8>>, Vec<u8>) {
        let (a, ab) = constant("alpha", 1, 1024);
        let (b, bb) = constant("beta", -7, 2048);
        let manifest = ArchiveManifest {
            profile: "u1/v1".into(),
            universe: "vole.audio.u1".into(),
            entries: vec![a, b],
        };
        let objects = vec![ab, bb];
        let bytes = encode_archive(&manifest, &objects).expect("encode");
        (manifest, objects, bytes)
    }

    /// Offset of the section table.
    fn table_start(bytes: &[u8]) -> usize {
        let mut p = MAGIC.len() + 1;
        let plen = usize::from(bytes[p]);
        p += 1 + plen;
        let ulen = usize::from(bytes[p]);
        p += 1 + ulen;
        p + 4
    }

    /// Recompute every section digest and the trailing archive digest, so a
    /// mutation reaches the *content* validators rather than failing integrity.
    fn reseal(bytes: &mut [u8]) {
        let ts = table_start(bytes);
        let count = u32::from_le_bytes(bytes[ts - 4..ts].try_into().unwrap()) as usize;
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

    #[test]
    fn round_trip_is_exact_and_canonical() {
        let (manifest, objects, bytes) = valid_archive();
        let decoded = decode_archive(&bytes).expect("decode");
        assert_eq!(decoded.profile(), "u1/v1");
        assert_eq!(decoded.universe(), "vole.audio.u1");
        assert_eq!(decoded.objects(), 2);
        assert_eq!(decoded.manifest(), &manifest);
        for (i, want) in objects.iter().enumerate() {
            assert_eq!(decoded.object_bytes(&bytes, i).unwrap(), want.as_slice());
        }
        // The format is canonical: decode -> re-encode is byte-identical.
        let rebuilt = encode_archive(decoded.manifest(), &objects).expect("re-encode");
        assert_eq!(rebuilt, bytes);
    }

    #[test]
    fn encode_refuses_a_content_id_that_does_not_match_its_bytes() {
        let (mut a, ab) = constant("alpha", 1, 1024);
        a.content_id = ContentId::from_bytes([0u8; 32]);
        let manifest = ArchiveManifest {
            profile: "u1/v1".into(),
            universe: "vole.audio.u1".into(),
            entries: vec![a],
        };
        let err = encode_archive(&manifest, &[ab]).unwrap_err();
        assert_eq!(err.kind(), crate::error::Kind::Integrity);
    }

    #[test]
    fn encode_refuses_non_canonical_manifests() {
        let (a, ab) = constant("beta", 1, 1024);
        let (b, bb) = constant("alpha", 2, 1024);
        let unsorted = ArchiveManifest {
            profile: "u1/v1".into(),
            universe: "vole.audio.u1".into(),
            entries: vec![a.clone(), b.clone()],
        };
        assert!(encode_archive(&unsorted, &[ab.clone(), bb.clone()]).is_err());

        let mut bad_name = a.clone();
        bad_name.name = "a/b".into();
        let manifest = ArchiveManifest {
            profile: "u1/v1".into(),
            universe: "vole.audio.u1".into(),
            entries: vec![bad_name],
        };
        assert!(encode_archive(&manifest, &[ab]).is_err());

        let empty = ArchiveManifest {
            profile: "u1/v1".into(),
            universe: "vole.audio.u1".into(),
            entries: Vec::new(),
        };
        assert!(encode_archive(&empty, &[]).is_err());
    }

    #[test]
    fn corruption_without_resealing_is_rejected() {
        let (_, _, bytes) = valid_archive();
        // Truncation.
        assert!(decode_archive(&bytes[..bytes.len() - 1]).is_err());
        // Magic.
        let mut m = bytes.clone();
        m[0] ^= 0xFF;
        assert!(decode_archive(&m).is_err());
        // Payload bit flip (fails the section digest).
        let ts = table_start(&bytes);
        let off = u64::from_le_bytes(bytes[ts + 2..ts + 10].try_into().unwrap()) as usize;
        let mut f = bytes.clone();
        f[off] ^= 0xFF;
        assert!(decode_archive(&f).is_err());
    }

    #[test]
    fn unsupported_version_is_a_typed_error() {
        let (_, _, bytes) = valid_archive();
        let mut v = bytes.clone();
        v[MAGIC.len()] = VERSION.wrapping_add(1);
        let err = decode_archive(&v).unwrap_err();
        assert_eq!(err.kind(), crate::error::Kind::Unsupported);
    }

    #[test]
    fn resealed_structural_mutations_are_rejected() {
        let (_, _, base) = valid_archive();
        let ts = table_start(&base);

        // A forged content id: flip a byte in an OBJECT payload (section 1) and
        // reseal every digest, including the archive digest. The manifest's
        // identity column still names the original bytes, so the decoder must
        // catch the mismatch rather than trusting the table.
        let rec1 = ts + SECTION_RECORD_BYTES;
        let off = u64::from_le_bytes(base[rec1 + 2..rec1 + 10].try_into().unwrap()) as usize;
        let mut f = base.clone();
        f[off] ^= 0x01;
        reseal(&mut f);
        let err = decode_archive(&f).unwrap_err();
        assert_eq!(err.kind(), crate::error::Kind::Integrity);

        // Corrupting the manifest payload itself is rejected too (a different
        // validator, but never accepted).
        let man = u64::from_le_bytes(base[ts + 2..ts + 10].try_into().unwrap()) as usize;
        let mut mp = base.clone();
        mp[man] ^= 0x01;
        reseal(&mut mp);
        assert!(decode_archive(&mp).is_err());

        // Reserved byte must be zero.
        let mut r = base.clone();
        r[ts + 1] = 1;
        reseal(&mut r);
        assert!(decode_archive(&r).is_err());

        // Unknown section kind (the object row becomes kind 0x7F).
        let mut k = base.clone();
        k[ts + SECTION_RECORD_BYTES] = 0x7F;
        reseal(&mut k);
        assert!(decode_archive(&k).is_err());

        // Swapping the two section kinds breaks the manifest/objects order.
        let mut s = base.clone();
        s[ts] = SectionKind::Object as u8;
        s[ts + SECTION_RECORD_BYTES] = SectionKind::Manifest as u8;
        reseal(&mut s);
        assert!(decode_archive(&s).is_err());

        // A non-contiguous payload offset.
        let mut g = base.clone();
        let second = ts + SECTION_RECORD_BYTES + 2;
        let off2 = u64::from_le_bytes(base[second..second + 8].try_into().unwrap());
        g[second..second + 8].copy_from_slice(&(off2 + 1).to_le_bytes());
        reseal(&mut g);
        assert!(decode_archive(&g).is_err());

        // A corrupted manifest entry count (rejects the section/entry mismatch).
        let manifest_off = u64::from_le_bytes(base[ts + 2..ts + 10].try_into().unwrap()) as usize;
        let mut c = base.clone();
        c[manifest_off..manifest_off + 4].copy_from_slice(&99u32.to_le_bytes());
        reseal(&mut c);
        assert!(decode_archive(&c).is_err());
    }

    #[test]
    fn trailing_bytes_are_rejected_even_when_resealed() {
        let (_, _, base) = valid_archive();
        let mut t = base.clone();
        let dig = t.len() - DIGEST_BYTES;
        t.insert(dig, 0xAB); // an unreferenced byte before the digest
        reseal(&mut t);
        assert!(decode_archive(&t).is_err());
    }

    #[test]
    fn allocation_bombs_are_rejected_before_allocating() {
        // A manifest entry count that claims far more entries than the payload
        // can hold must be refused without allocating for it.
        let (_, _, base) = valid_archive();
        let ts = table_start(&base);
        let manifest_off = u64::from_le_bytes(base[ts + 2..ts + 10].try_into().unwrap()) as usize;
        let mut b = base.clone();
        b[manifest_off..manifest_off + 4].copy_from_slice(&MAX_SECTIONS.to_le_bytes());
        reseal(&mut b);
        assert!(decode_archive(&b).is_err());
    }
}

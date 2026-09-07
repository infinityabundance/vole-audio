//! Object identity.
//!
//! Two identity layers:
//!
//! * `ObjectId` — the *corpus/archive-local* handle (u64) by which voices and
//!   dependencies refer to objects. It is assigned by the loader; two loads of
//!   the same content may assign different `ObjectId`s.
//! * `ContentId` — SHA-256 of the object's canonical serialization bytes
//!   (descriptor + payload). Same content ⇒ same `ContentId`, anywhere.

use core::fmt;

/// Corpus/archive-local object handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(pub u64);

impl ObjectId {
    pub const ZERO: ObjectId = ObjectId(0);
    pub const fn new(v: u64) -> Self {
        Self(v)
    }
    pub const fn to_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "obj#{}", self.0)
    }
}

/// Content identity: SHA-256 digest of canonical object bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentId(pub [u8; 32]);

impl ContentId {
    pub const fn from_bytes(b: [u8; 32]) -> Self {
        Self(b)
    }

    pub const fn to_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Display for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

/// Reference from one object to another (dependency edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Dependency {
    pub from: ObjectId,
    pub to: ObjectId,
    /// Role tag (0 = plain reference; future roles: residual-source, table, ...).
    pub role: u8,
}

impl fmt::Display for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} --[{}]--> {}", self.from, self.role, self.to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_id_display_is_64_hex() {
        let id = ContentId([0xAB; 32]);
        let s = id.to_string();
        assert_eq!(s.len(), 64);
        assert!(s.starts_with("abababab"));
    }

    #[test]
    fn ids_are_plain_data() {
        assert_eq!(ObjectId::ZERO.to_u64(), 0);
        assert_eq!(
            Dependency {
                from: ObjectId(1),
                to: ObjectId(2),
                role: 0
            }
            .to_string(),
            "obj#1 --[0]--> obj#2"
        );
    }
}

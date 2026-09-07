//! Host-side error taxonomy.
//!
//! Errors are structured so that a failure can always be (a) attributed to a
//! phase of the pipeline, (b) rendered without panicking, and (c) recorded in
//! evidence receipts. Normative evaluator arithmetic never returns `Err`; the
//! exact domain is infallible by construction and the status vocabulary for
//! courts/device words lives in `evidence::receipt::Verdict`.

use core::fmt;

/// Category of failure. Coarse enough to aggregate, precise enough to debug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Input violated a declared hard limit (`limits.rs`).
    LimitExceeded,
    /// Input was structurally malformed (lengths, chunks, framing, padding).
    Malformed,
    /// An unsupported format/feature was requested and was *not* silently
    /// converted (exact profile).
    Unsupported,
    /// A declared integrity/identity hash did not match.
    Integrity,
    /// A dependency is missing, cyclic, or unbound.
    Dependency,
    /// An external API (CUDA, HIP/HSA, ALSA, OS) returned a failure.
    External,
    /// A backend/path is not available in this environment.
    Unavailable,
    /// An internal invariant was violated (bug; not caused by input).
    Internal,
    /// I/O failure.
    Io,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Kind::LimitExceeded => "limit exceeded",
            Kind::Malformed => "malformed input",
            Kind::Unsupported => "unsupported",
            Kind::Integrity => "integrity mismatch",
            Kind::Dependency => "dependency error",
            Kind::External => "external api error",
            Kind::Unavailable => "unavailable",
            Kind::Internal => "internal error",
            Kind::Io => "i/o error",
        };
        f.write_str(s)
    }
}

/// Structured error with a category and a free-form, alloc-free message spine.
///
/// The message is a `String` only on the std host build; this type is never
/// compiled for device targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    kind: Kind,
    message: String,
}

impl Error {
    pub fn new(kind: Kind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn limit(msg: impl Into<String>) -> Self {
        Self::new(Kind::LimitExceeded, msg)
    }

    pub fn malformed(msg: impl Into<String>) -> Self {
        Self::new(Kind::Malformed, msg)
    }

    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::new(Kind::Unsupported, msg)
    }

    pub fn integrity(msg: impl Into<String>) -> Self {
        Self::new(Kind::Integrity, msg)
    }

    pub fn dependency(msg: impl Into<String>) -> Self {
        Self::new(Kind::Dependency, msg)
    }

    pub fn external(msg: impl Into<String>) -> Self {
        Self::new(Kind::External, msg)
    }

    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self::new(Kind::Unavailable, msg)
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::new(Kind::Internal, msg)
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::new(Kind::Io, e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::new(Kind::Malformed, format!("json: {e}"))
    }
}

/// Convenience alias matching the crate convention `Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_informative() {
        let e = Error::limit("objects > MAX_OBJECTS_PER_CORPUS");
        let s = e.to_string();
        assert!(s.contains("limit exceeded"));
        assert!(s.contains("MAX_OBJECTS_PER_CORPUS"));
        assert_eq!(e.kind(), Kind::LimitExceeded);
    }

    #[test]
    fn io_conversion_preserves_kind() {
        let e = Error::from(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        assert_eq!(e.kind(), Kind::Io);
    }
}

//! Event transport payloads (Phase N, contract §36).
//!
//! An event frame's payload is the canonical byte form of a
//! [`crate::universe::event::Event`], so the frozen total order
//! `(media_frame, class_priority, sequence)` survives the wire unchanged.
//!
//! ```text
//! EventPayload :=
//!     MEDIA_FRAME  i64
//!     CLASS        u8      EventClass
//!     SEQUENCE     u64
//!     KIND         u16
//! ```

use crate::error::{Error, Result};
use crate::universe::event::{Event, EventClass};
use crate::universe::time::MediaFrame;

/// Encoded size of one event payload.
pub const EVENT_PAYLOAD_BYTES: usize = 8 + 1 + 8 + 2;

fn event_class_from_u8(v: u8) -> Result<EventClass> {
    Ok(match v {
        0 => EventClass::Start,
        1 => EventClass::Stop,
        2 => EventClass::Param,
        3 => EventClass::World,
        4 => EventClass::Clock,
        5 => EventClass::Diagnostic,
        other => {
            return Err(Error::new(
                crate::error::Kind::Unsupported,
                format!("unknown event class {other}"),
            ));
        }
    })
}

/// Canonical payload bytes for one event.
pub fn encode_event(e: &Event) -> Vec<u8> {
    let mut out = Vec::with_capacity(EVENT_PAYLOAD_BYTES);
    out.extend_from_slice(&e.media_frame.to_i64().to_le_bytes());
    out.push(e.class as u8);
    out.extend_from_slice(&e.sequence.to_le_bytes());
    out.extend_from_slice(&e.kind.to_le_bytes());
    out
}

/// Parse one event payload (exactly `EVENT_PAYLOAD_BYTES`; no trailing bytes).
pub fn decode_event(bytes: &[u8]) -> Result<Event> {
    if bytes.len() != EVENT_PAYLOAD_BYTES {
        return Err(Error::malformed("event payload has the wrong length"));
    }
    let media_frame = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let class = event_class_from_u8(bytes[8])?;
    let sequence = u64::from_le_bytes(bytes[9..17].try_into().unwrap());
    let kind = u16::from_le_bytes(bytes[17..19].try_into().unwrap());
    Ok(Event::new(
        MediaFrame::new(media_frame),
        class,
        sequence,
        kind,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_round_trip_is_exact() {
        let e = Event::new(MediaFrame::new(12_345), EventClass::Param, 7, 0x0302);
        let bytes = encode_event(&e);
        assert_eq!(bytes.len(), EVENT_PAYLOAD_BYTES);
        assert_eq!(decode_event(&bytes).unwrap(), e);
        // The wire form preserves the frozen order keys.
        let other = Event::new(MediaFrame::new(12_345), EventClass::Diagnostic, 7, 0);
        assert!(e < other, "class priority must survive the wire");
    }

    #[test]
    fn malformed_event_payloads_are_rejected() {
        assert!(decode_event(&[]).is_err());
        let mut bytes = encode_event(&Event::new(MediaFrame::new(0), EventClass::Start, 0, 0));
        bytes.push(0);
        assert!(decode_event(&bytes).is_err()); // trailing byte
        let mut bad = encode_event(&Event::new(MediaFrame::new(0), EventClass::Start, 0, 0));
        bad[8] = 0x7F;
        assert!(decode_event(&bad).is_err()); // unknown class
    }
}

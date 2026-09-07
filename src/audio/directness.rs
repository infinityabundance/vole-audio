//! Path directness classes D0..D3.
//!
//! The path class and the physical topology are separate axes and must never
//! be conflated: an mmap'd ALSA region does not by itself make a path D1, and
//! a PCIe peer link does not by itself make a path D2. Receipts record both.

use serde::{Deserialize, Serialize};

/// Directness of the sample-materialization path (paper §"D0/D1/D2/D3").
///
/// The enum exists for the first implementation, but D3 must always report
/// `NotImplemented`/future-conceptual until an endpoint-native evaluator
/// exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Directness {
    /// Conventional buffered path: host PCM staging and/or device->host copy.
    D0Buffered,
    /// GPU writes final sample codes into the actual endpoint-mapped region.
    D1EndpointMapped,
    /// Peer-device / endpoint-DMA materialization across PCIe/fabric.
    D2PeerDevice,
    /// Endpoint-native evaluation (FPGA/DSP/ASIC) — future conceptual only.
    D3EndpointNative,
}

impl Directness {
    pub const fn label(self) -> &'static str {
        match self {
            Directness::D0Buffered => "D0_BUFFERED",
            Directness::D1EndpointMapped => "D1_ENDPOINT_MAPPED",
            Directness::D2PeerDevice => "D2_PEER_DEVICE",
            Directness::D3EndpointNative => "D3_ENDPOINT_NATIVE",
        }
    }
}

impl std::fmt::Display for Directness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directness_labels() {
        assert_eq!(Directness::D0Buffered.label(), "D0_BUFFERED");
        assert_eq!(Directness::D3EndpointNative.label(), "D3_ENDPOINT_NATIVE");
    }

    #[test]
    fn serde_shape() {
        let j = serde_json::to_string(&Directness::D1EndpointMapped).unwrap();
        assert_eq!(j, "\"D1_ENDPOINT_MAPPED\"");
    }
}

//! Physical memory/endpoint topology classes.
//!
//! Topology describes how an endpoint's observation memory relates to the
//! computing device. Directness is never inferred from topology alone; both
//! are measured and recorded independently (see `directness.rs`).

use serde::{Deserialize, Serialize};

/// Physical topology class of the observation memory relative to the executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Topology {
    /// Unified memory architecture (single address space, coherent).
    Uma,
    /// Host memory mapped into the device address space (PCIe/soC path).
    HostMapped,
    /// Two peer devices over PCIe/fabric with peer access.
    PciePeer,
    /// Endpoint memory behind a device BAR (e.g. some FPGA/DSP regions).
    DeviceBar,
    /// Custom endpoint memory class (vendor-specific).
    CustomEndpoint,
    /// Network/remote endpoint memory.
    NetworkEndpoint,
    /// Not yet classified.
    Unknown,
}

impl Topology {
    pub const fn label(self) -> &'static str {
        match self {
            Topology::Uma => "UMA",
            Topology::HostMapped => "HOST_MAPPED",
            Topology::PciePeer => "PCIE_PEER",
            Topology::DeviceBar => "DEVICE_BAR",
            Topology::CustomEndpoint => "CUSTOM_ENDPOINT",
            Topology::NetworkEndpoint => "NETWORK_ENDPOINT",
            Topology::Unknown => "UNKNOWN",
        }
    }
}

impl std::fmt::Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        assert_eq!(Topology::HostMapped.label(), "HOST_MAPPED");
        assert_eq!(Topology::DeviceBar.label(), "DEVICE_BAR");
    }
}

//! Energy measurement adapters.
//!
//! Energy claims are optional and only made when a real measurement source is
//! available. Sources, in priority order per platform:
//!   * NVIDIA: NVML (`libnvidia-ml.so`)
//!   * AMD: ROCm SMI / AMDSMI (`librocm_smi64.so`) or hwmon/sysfs
//!   * generic: hwmon power readings under /sys/class/hwmon
//!   * external meter input (recorded manually)
//!
//! No energy figure is ever invented from TDP. Until a source reports real
//! readings, `energy` in receipts is `None`.

use serde::{Deserialize, Serialize};

/// One measured power/energy sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergySample {
    /// Machine-readable source id, e.g. "nvml", "rocm_smi", "hwmon", "meter".
    pub source: String,
    /// Instantaneous power in watts, if the source reports it.
    pub watts: Option<f64>,
    /// Energy delta in joules for the sample interval, if the source reports it.
    pub joules_delta: Option<f64>,
    /// Source resolution in watts if declared by the source.
    pub resolution_watts: Option<f64>,
    /// Sample interval seconds.
    pub interval_secs: f64,
    /// Monotonic timestamp (ns) of the sample.
    pub at_ns: i64,
}

/// Declared uncertainty and method for a whole energy measurement run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EnergyMethod {
    pub source: String,
    pub idle_watts_subtracted: Option<f64>,
    pub idle_watts: Option<f64>,
    pub load_watts_mean: Option<f64>,
    pub uncertainty_note: String,
}

/// A recorded energy measurement with provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyReport {
    pub method: EnergyMethod,
    pub samples: Vec<EnergySample>,
}

/// Adapter trait for energy sources. Implementations live with the backend
/// that owns the measurement (CUDA backend -> NVML, ROCm backend -> AMDSMI).
pub trait EnergySource {
    fn source_id(&self) -> &'static str;

    /// Read current power in watts.
    fn read_watts(&self) -> Option<f64>;

    /// Human-readable capability note.
    fn describe(&self) -> String;
}

/// Adapter that reports nothing — the honest default.
pub struct NoEnergy;

impl EnergySource for NoEnergy {
    fn source_id(&self) -> &'static str {
        "none"
    }

    fn read_watts(&self) -> Option<f64> {
        None
    }

    fn describe(&self) -> String {
        "no energy measurement source available".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_energy_is_honest() {
        let e = NoEnergy;
        assert_eq!(e.source_id(), "none");
        assert!(e.read_watts().is_none());
    }
}

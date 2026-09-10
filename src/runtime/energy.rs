//! Host power/energy measurement for the interference court (Phase M, "energy
//! where measurable").
//!
//! Two distinct things are probed, and neither is inferred:
//!
//! * **instantaneous power** from Linux hwmon (`/sys/class/hwmon/*/power*_input`
//!   or `*_average`, microwatts) — useful as a spot reading, but two spot
//!   readings are *not* a workload-energy measurement;
//! * **cumulative energy** from Linux powercap (`/sys/class/powercap/*/energy_uj`,
//!   microjoules), which is the right instrument for joules over an interval. A
//!   counter that wraps is handled against its declared `max_energy_range_uj`.
//!
//! NVML/AMDSMI cumulative GPU energy would be a further source on hardware that
//! exposes it; it is **not** probed here, and GPU-only energy would not stand in
//! for CPU/system energy for the scalar B2–B5 court anyway. When no source
//! exists, energy is `NOT_AVAILABLE` — never estimated from TDP or a model.

use std::path::PathBuf;

/// A probed instantaneous power source.
#[derive(Debug, Clone)]
pub struct PowerSource {
    path: PathBuf,
    scale_watts: f64,
    id: String,
}

impl PowerSource {
    /// Instantaneous power in watts, if the source currently reports a value.
    pub fn read_watts(&self) -> Option<f64> {
        let text = std::fs::read_to_string(&self.path).ok()?;
        let raw: f64 = text.trim().parse().ok()?;
        Some(raw * self.scale_watts)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn describe(&self) -> String {
        format!("{} ({})", self.id, self.path.display())
    }
}

/// A probed cumulative energy counter (joules over an interval).
#[derive(Debug, Clone)]
pub struct EnergyCounter {
    path: PathBuf,
    max_range_uj: Option<u64>,
    id: String,
}

impl EnergyCounter {
    /// Current raw counter value in microjoules.
    pub fn read_uj(&self) -> Option<u64> {
        let text = std::fs::read_to_string(&self.path).ok()?;
        text.trim().parse().ok()
    }

    /// Joules consumed between two readings of this counter, handling wraparound
    /// against the declared range when one is available.
    ///
    /// `None` means the interval cannot be determined (a wrapped counter with no
    /// declared range) — never a fabricated `0.0`, which would be
    /// indistinguishable from a genuine zero-energy interval.
    pub fn joules_between(&self, start_uj: u64, end_uj: u64) -> Option<f64> {
        let delta_uj = if end_uj >= start_uj {
            end_uj - start_uj
        } else {
            match self.max_range_uj {
                Some(max) if max > start_uj => (max - start_uj) + end_uj,
                // Unknown range and a wrapped counter: report unavailable.
                _ => return None,
            }
        };
        Some(delta_uj as f64 * 1e-6)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn describe(&self) -> String {
        format!("{} ({})", self.id, self.path.display())
    }
}

/// Find a host instantaneous power source, if one exists.
pub fn probe_power() -> Option<PowerSource> {
    let hwmons = std::fs::read_dir("/sys/class/hwmon").ok()?;
    for hw in hwmons.flatten() {
        let base = hw.path();
        let files = match std::fs::read_dir(&base) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let hw_name = std::fs::read_to_string(base.join("name"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".into());
        for file in files.flatten() {
            let fname = file.file_name().to_string_lossy().to_string();
            let is_power = fname.starts_with("power")
                && (fname.ends_with("_input") || fname.ends_with("_average"));
            if !is_power {
                continue;
            }
            let path = file.path();
            if std::fs::read_to_string(&path).is_ok() {
                return Some(PowerSource {
                    path,
                    // hwmon power attributes are microwatts.
                    scale_watts: 1e-6,
                    id: format!("hwmon:{hw_name}:{fname}"),
                });
            }
        }
    }
    None
}

/// Find a cumulative energy counter (Linux powercap), if one exists **and is
/// readable**.
///
/// A present-but-unreadable counter (powercap `energy_uj` is commonly root-only)
/// is not a usable source, so it is skipped rather than reported as available.
pub fn probe_energy_counter() -> Option<EnergyCounter> {
    let entries = std::fs::read_dir("/sys/class/powercap").ok()?;
    for e in entries.flatten() {
        let base = e.path();
        let path = base.join("energy_uj");
        if !path.is_file() {
            continue;
        }
        // Must be readable to be a usable source: skip an unreadable candidate
        // and keep looking, rather than abandoning the whole probe.
        match std::fs::read_to_string(&path) {
            Ok(text) if text.trim().parse::<u64>().is_ok() => {}
            _ => continue,
        }
        let max_range_uj = std::fs::read_to_string(base.join("max_energy_range_uj"))
            .ok()
            .and_then(|s| s.trim().parse().ok());
        let name = std::fs::read_to_string(base.join("name"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| e.file_name().to_string_lossy().to_string());
        return Some(EnergyCounter {
            path,
            max_range_uj,
            id: format!("powercap:{name}"),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_are_honest() {
        // Either a real source exists and reads finite values, or it does not.
        if let Some(p) = probe_power() {
            assert!(!p.id().is_empty());
            if let Some(w) = p.read_watts() {
                assert!(w.is_finite() && w >= 0.0, "power must be finite: {w}");
            }
        }
        if let Some(c) = probe_energy_counter() {
            assert!(!c.id().is_empty());
        }
    }

    #[test]
    fn counter_delta_handles_plain_and_wrapped_intervals() {
        let counter = EnergyCounter {
            path: PathBuf::from("/nonexistent"),
            max_range_uj: Some(1_000_000),
            id: "test".into(),
        };
        assert!((counter.joules_between(250_000, 750_000).unwrap() - 0.5).abs() < 1e-12);
        // Wrapped: 900k -> 100k over a 1M range is a 200k uJ (0.2 J) interval.
        assert!((counter.joules_between(900_000, 100_000).unwrap() - 0.2).abs() < 1e-12);
        // A genuine zero interval is Some(0.0), distinct from an unknown one.
        assert_eq!(counter.joules_between(5, 5), Some(0.0));
        let unknown = EnergyCounter {
            path: PathBuf::from("/nonexistent"),
            max_range_uj: None,
            id: "test".into(),
        };
        // A wrapped counter with no declared range reports unavailable, not zero.
        assert_eq!(unknown.joules_between(900_000, 100_000), None);
    }
}

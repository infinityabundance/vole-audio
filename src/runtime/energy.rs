//! Host power measurement for the interference court (Phase M, "energy where
//! measurable").
//!
//! A real reading requires a real source. This probes Linux hwmon power inputs
//! (`/sys/class/hwmon/*/power*_input`, microwatts). When none exists — as on the
//! reference host — energy is reported `NOT_AVAILABLE`, never estimated from TDP
//! or from a model.

use std::path::{Path, PathBuf};

/// A probed power source.
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

/// Find a host power source, if one exists.
pub fn probe_power() -> Option<PowerSource> {
    let dir = Path::new("/sys/class/hwmon");
    let hwmons = std::fs::read_dir(dir).ok()?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_is_honest() {
        // Either a real source exists and reads a finite non-negative value, or
        // there is no source. No source is invented.
        if let Some(p) = probe_power() {
            assert!(!p.id().is_empty());
            if let Some(w) = p.read_watts() {
                assert!(w.is_finite() && w >= 0.0, "power must be finite: {w}");
            }
        }
    }
}

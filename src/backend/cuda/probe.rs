//! CUDA driver/device probe evidence (Phase G).
//!
//! Everything a receipt needs about the CUDA environment: driver version,
//! device identity/compute capability, and the capability attributes that
//! gate later courts (stream priorities, unified addressing, host-register
//! support, concurrent managed access, L1 caching, kernel-exec timeout).
//! Probing never panics and never implies support that is absent.

use crate::backend::cuda::driver::{Cuda, DeviceInfo};
use serde::{Deserialize, Serialize};

/// Serialize-able probe snapshot (receipt extras).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaProbe {
    pub driver_version: i32,
    /// "major.minor" of the CUDA driver (e.g. "13.3").
    pub driver_version_label: String,
    pub device: DeviceInfoProbe,
    /// True when the context was created successfully (the probe itself ran).
    pub context_created: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfoProbe {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: String,
    pub sm: String,
    pub pci: String,
    pub multiprocessor_count: i32,
    pub clock_rate_khz: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_multiprocessor: i32,
    pub unified_addressing: bool,
    pub stream_priorities_supported: bool,
    pub stream_priority_range: (i32, i32),
    pub concurrent_managed_access: bool,
    pub host_register_supported: bool,
    pub global_l1_cache_supported: bool,
    pub kernel_exec_timeout: bool,
}

impl From<&DeviceInfo> for DeviceInfoProbe {
    fn from(d: &DeviceInfo) -> Self {
        DeviceInfoProbe {
            ordinal: d.ordinal,
            name: d.name.clone(),
            compute_capability: format!("{}.{}", d.major, d.minor),
            sm: d.sm.clone(),
            pci: format!(
                "{:04x}:{:02x}:{:02x}.0",
                d.pci_domain_id, d.pci_bus_id, d.pci_device_id
            ),
            multiprocessor_count: d.multiprocessor_count,
            clock_rate_khz: d.clock_rate_khz,
            max_threads_per_block: d.max_threads_per_block,
            max_threads_per_multiprocessor: d.max_threads_per_multiprocessor,
            unified_addressing: d.unified_addressing != 0,
            stream_priorities_supported: d.stream_priorities_supported != 0,
            stream_priority_range: (d.min_stream_priority, d.max_stream_priority),
            concurrent_managed_access: d.concurrent_managed_access != 0,
            host_register_supported: d.host_register_supported != 0,
            global_l1_cache_supported: d.global_l1_cache_supported != 0,
            kernel_exec_timeout: d.kernel_exec_timeout != 0,
        }
    }
}

impl CudaProbe {
    /// Probe device `ordinal` (default 0). `Ok(None)` when no driver/device
    /// is available (never an error — evidence, not failure).
    pub fn capture(ordinal: i32) -> crate::error::Result<Option<CudaProbe>> {
        let cuda = match Cuda::open(ordinal) {
            Ok(c) => c,
            Err(_) => return Ok(None),
        };
        let probe = CudaProbe {
            driver_version: cuda.driver_version,
            driver_version_label: format!(
                "{}.{}",
                cuda.driver_version / 1000,
                (cuda.driver_version / 10) % 100
            ),
            device: (&cuda.device).into(),
            context_created: true,
        };
        drop(cuda);
        Ok(Some(probe))
    }
}

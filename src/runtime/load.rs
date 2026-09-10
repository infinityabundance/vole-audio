//! Deterministic background load for the interference court (Phase M, §49).
//!
//! Each load is a bounded set of worker threads that run until the guard is
//! dropped. Loads are deliberately simple and reproducible: the point is to
//! create *pressure* (CPU scheduler, memory bandwidth, storage) so tail latency
//! under contention can be measured against the idle baseline, not to model any
//! particular application.
//!
//! Conditions this host **cannot** control are never faked: GPU clock state,
//! DVFS, thermal steady state, PCIe power saving and compositor/display load are
//! reported as `NOT_CONTROLLED` rather than claimed.

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A controllable interference condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadKind {
    /// One spinner per logical CPU, saturating the scheduler.
    CpuBurn,
    /// Streaming reads/writes over a large buffer, pressuring memory bandwidth.
    MemoryBandwidth,
    /// Write + `fsync` + read cycles on a block-backed file.
    StorageIo,
}

impl LoadKind {
    pub fn name(self) -> &'static str {
        match self {
            LoadKind::CpuBurn => "cpu_burn",
            LoadKind::MemoryBandwidth => "memory_bandwidth",
            LoadKind::StorageIo => "storage_io",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            LoadKind::CpuBurn => "one CPU-bound spinner per logical CPU",
            LoadKind::MemoryBandwidth => "streaming access over a large resident buffer",
            LoadKind::StorageIo => "write + fsync + read cycles on a block-backed file",
        }
    }
}

/// A running load; dropping it stops the workers.
pub struct LoadGuard {
    kind: LoadKind,
    stop: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
    scratch: Option<PathBuf>,
}

impl LoadGuard {
    /// Start `kind` with `threads` workers.
    pub fn start(kind: LoadKind, threads: usize, dir: &Path) -> Result<LoadGuard> {
        let threads = threads.max(1);
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(threads);
        let mut scratch = None;
        match kind {
            LoadKind::CpuBurn => {
                for t in 0..threads {
                    let stop = stop.clone();
                    handles.push(
                        std::thread::Builder::new()
                            .name(format!("vole-load-cpu-{t}"))
                            .spawn(move || cpu_burn(&stop))
                            .map_err(|e| {
                                Error::new(crate::error::Kind::Io, format!("load worker: {e}"))
                            })?,
                    );
                }
            }
            LoadKind::MemoryBandwidth => {
                for t in 0..threads {
                    let stop = stop.clone();
                    handles.push(
                        std::thread::Builder::new()
                            .name(format!("vole-load-mem-{t}"))
                            .spawn(move || memory_bandwidth(&stop))
                            .map_err(|e| {
                                Error::new(crate::error::Kind::Io, format!("load worker: {e}"))
                            })?,
                    );
                }
            }
            LoadKind::StorageIo => {
                std::fs::create_dir_all(dir)?;
                let path = dir.join(format!("load-{}.bin", std::process::id()));
                std::fs::write(&path, vec![0u8; 1 << 20])?;
                scratch = Some(path.clone());
                for t in 0..threads {
                    let stop = stop.clone();
                    let path = path.clone();
                    handles.push(
                        std::thread::Builder::new()
                            .name(format!("vole-load-io-{t}"))
                            .spawn(move || storage_io(&stop, &path))
                            .map_err(|e| {
                                Error::new(crate::error::Kind::Io, format!("load worker: {e}"))
                            })?,
                    );
                }
            }
        }
        Ok(LoadGuard {
            kind,
            stop,
            handles,
            scratch,
        })
    }

    pub fn kind(&self) -> LoadKind {
        self.kind
    }

    pub fn threads(&self) -> usize {
        self.handles.len()
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
        if let Some(p) = &self.scratch {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn cpu_burn(stop: &AtomicBool) {
    let mut x = 0x9E3779B97F4A7C15u64;
    while !stop.load(Ordering::Relaxed) {
        for _ in 0..4096 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
        }
        std::hint::black_box(x);
    }
}

fn memory_bandwidth(stop: &AtomicBool) {
    // ~64 MiB per worker; streamed so the traffic is real but bounded.
    let mut buf = vec![0u8; 64 << 20];
    let mut seed = 0x2545F4914F6CDD1Du64;
    while !stop.load(Ordering::Relaxed) {
        for chunk in buf.chunks_mut(64) {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            chunk.fill((seed >> 56) as u8);
        }
        std::hint::black_box(&buf);
    }
}

fn storage_io(stop: &AtomicBool, path: &Path) {
    use std::io::{Read, Seek, SeekFrom, Write};
    while !stop.load(Ordering::Relaxed) {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
        {
            let mut buf = vec![0u8; 1 << 16];
            let _ = f.seek(SeekFrom::Start(0));
            let _ = f.write_all(&buf);
            let _ = f.sync_all();
            let _ = f.seek(SeekFrom::Start(0));
            let _ = f.read_exact(&mut buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_load_starts_and_stops() {
        let dir = std::env::temp_dir().join(format!("vole-load-test-{}", std::process::id()));
        let guard = LoadGuard::start(LoadKind::CpuBurn, 1, &dir).unwrap();
        assert_eq!(guard.threads(), 1);
        assert_eq!(guard.kind().name(), "cpu_burn");
        std::thread::sleep(std::time::Duration::from_millis(20));
        drop(guard);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn storage_load_uses_the_scratch_dir_and_cleans_up() {
        let dir = std::env::temp_dir().join(format!("vole-load-io-test-{}", std::process::id()));
        let path = {
            let guard = LoadGuard::start(LoadKind::StorageIo, 1, &dir).unwrap();
            let p = guard.scratch.clone().unwrap();
            assert!(p.exists());
            std::thread::sleep(std::time::Duration::from_millis(20));
            p
        };
        assert!(!path.exists(), "scratch file is removed on drop");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

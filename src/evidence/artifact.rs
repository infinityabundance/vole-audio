//! Artifact evidence helper: bind a GPU artifact's bytes AND its provenance
//! sidecar into one receipt value.
//!
//! The build scripts write `<artifact>.json` sidecars next to the PTX and
//! AMDGPU artifacts (source-tree identity, toolchain, gfx/ISA, dirty flag).
//! Courts must consume the sidecar into the receipt rather than recording
//! only the artifact SHA-256: the hash binds the bytes, the sidecar binds
//! the artifact back to the source tree and toolchain that produced it.

use serde_json::{Value, json};

/// The provenance sidecar for an artifact is `<artifact>.json`
/// (vole_audio.ptx -> vole_audio.ptx.json; vole_audio.amdgcn.elf ->
/// vole_audio.amdgcn.json).
fn sidecar_path(path: &std::path::Path) -> std::path::PathBuf {
    let file = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(format!("{file}.json"))
}

/// Read an artifact (bytes + sha256 + sidecar when present) into a JSON
/// value for a receipt extra.
pub fn artifact_evidence(path: &std::path::Path) -> Value {
    let mut v = match std::fs::read(path) {
        Ok(bytes) => {
            let sha = crate::hash::sha256::hex(&crate::hash::sha256::Sha256::digest(&bytes));
            json!({
                "path": path.display().to_string(),
                "present": true,
                "bytes": bytes.len(),
                "sha256": sha,
            })
        }
        Err(_) => json!({
            "path": path.display().to_string(),
            "present": false,
        }),
    };
    match std::fs::read_to_string(sidecar_path(path)) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(sidecar) => {
                v["sidecar_present"] = json!(true);
                // Surface the provenance-critical fields at the top level of
                // the extra (sidecar stays available whole underneath).
                for (key, dst) in [
                    ("source_tree_sha", "build_source_tree"),
                    ("source_dirty", "build_dirty"),
                    ("rustc", "build_toolchain"),
                    ("rustc_commit_hash", "build_toolchain_commit"),
                    ("target_cpu", "target_cpu"),
                    ("entries", "entries"),
                    ("determinism", "determinism"),
                ] {
                    if let Some(val) = sidecar.get(key) {
                        v[dst] = val.clone();
                    }
                }
                v["sidecar"] = sidecar;
            }
            Err(_) => {
                v["sidecar_present"] = json!(false);
                v["sidecar_error"] = json!("unparseable sidecar");
            }
        },
        Err(_) => {
            v["sidecar_present"] = json!(false);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_artifact_is_absent() {
        let v = artifact_evidence(std::path::Path::new("/nonexistent/vole.ptx"));
        assert_eq!(v["present"], json!(false));
        assert_eq!(v["sidecar_present"], json!(false));
    }

    #[test]
    fn artifact_with_sidecar_merges_provenance() {
        let dir = std::env::temp_dir().join(format!("vole-art-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let art = dir.join("vole_audio.amdgcn.elf");
        std::fs::write(&art, b"\x7fELFfake").unwrap();
        std::fs::write(
            dir.join("vole_audio.amdgcn.elf.json"),
            r#"{"sha256":"x","source_tree_sha":"abc123","source_dirty":false,"rustc":"r1","target_cpu":"gfx906"}"#,
        )
        .unwrap();
        let v = artifact_evidence(&art);
        assert_eq!(v["present"], json!(true));
        assert_eq!(v["sidecar_present"], json!(true));
        assert_eq!(v["build_source_tree"], json!("abc123"));
        assert_eq!(v["build_dirty"], json!(false));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

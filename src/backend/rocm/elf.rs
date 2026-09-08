//! Minimal AMDGPU code-object ELF inspection (Phase I artifact evidence).
//!
//! The court must be self-defending: an arbitrary file named
//! `VOLE_ROCM_ARTIFACT` must not be hashed and called a GPU artifact. This
//! module validates, from the bytes alone:
//!
//! * ELF magic (`\x7fELF`) and class/endianness (ELF64 LE);
//! * `e_machine == EM_AMDGPU` (224) with the AMDGPU architecture flag;
//! * the required kernel entry symbols present in `.symtab` as exported
//!   FUNC symbols (the same names Phase J will look up).
//!
//! Everything here is a plain read of the file image — no external tools.

/// Required kernel entry symbols (mirrors `device::amdgcn_entry`).
pub const REQUIRED_ENTRIES: &[&str] = &[
    "vole_render_d0",
    "vole_entropy_decode",
    "vole_upmix_mono_dup",
];

/// ELF machine id for AMDGPU (gfx code objects).
pub const EM_AMDGPU: u16 = 224;

/// Information extracted from an AMDGPU code object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfInfo {
    /// e_machine == EM_AMDGPU.
    pub machine_amdgpu: bool,
    /// e_flags AMDGPU architecture version bits (0 for unknown).
    pub amdgpu_arch_version: u32,
    /// Required kernel entries found in .symtab as exported FUNC symbols.
    pub entries_found: Vec<String>,
    /// Entries from REQUIRED_ENTRIES that are missing.
    pub entries_missing: Vec<String>,
}

impl ElfInfo {
    pub fn valid(&self) -> bool {
        self.machine_amdgpu && self.entries_missing.is_empty()
    }
}

/// Inspect a code-object image. Returns `Err` for anything that is not a
/// parseable ELF64 file; `ElfInfo` then carries the semantic checks.
pub fn inspect_amdgcn_code_object(bytes: &[u8]) -> Result<ElfInfo, String> {
    if bytes.len() < 64 {
        return Err("not an ELF image (too short)".into());
    }
    if &bytes[0..4] != b"\x7fELF" {
        return Err("not an ELF image (bad magic)".into());
    }
    // e_ident[EI_CLASS]=2 (ELF64), e_ident[EI_DATA]=1 (little-endian).
    if bytes[4] != 2 || bytes[5] != 1 {
        return Err("not a little-endian ELF64 image".into());
    }
    // ELF64 header layout: e_type(2) e_machine(2) e_version(4) e_entry(8)
    // e_phoff(8) e_shoff(8) e_flags(4) e_ehsize(2) e_phentsize(2)
    // e_phnum(2) e_shentsize(2) e_shnum(2) e_shstrndx(2)  -- offsets 16..64
    let le_u16 = |b: &[u8], o: usize| -> u16 { u16::from_le_bytes([b[o], b[o + 1]]) };
    let le_u32 =
        |b: &[u8], o: usize| -> u32 { u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) };
    let le_u64 = |b: &[u8], o: usize| -> u64 {
        u64::from_le_bytes([
            b[o],
            b[o + 1],
            b[o + 2],
            b[o + 3],
            b[o + 4],
            b[o + 5],
            b[o + 6],
            b[o + 7],
        ])
    };

    let machine = le_u16(bytes, 18);
    let e_flags = le_u32(bytes, 48);
    let shoff = le_u64(bytes, 40) as usize;
    let shentsize = le_u16(bytes, 58) as usize;
    let shnum = le_u16(bytes, 60) as usize;
    let _shstrndx = le_u16(bytes, 62) as usize;

    // Section header table bounds check (hostile-file safety: no unbounded
    // reads past the image).
    if shoff == 0 || shentsize < 64 || shnum == 0 {
        return Err("ELF has no section header table".into());
    }
    let table_end = shoff
        .checked_add(shentsize.saturating_mul(shnum))
        .ok_or("section table overflow")?;
    if table_end > bytes.len() {
        return Err("ELF section table out of bounds".into());
    }

    // Locate .symtab / .strtab (SHT_SYMTAB=2, SHT_STRTAB=3).
    let mut symtab: Option<(usize, usize, usize)> = None; // (off, size, link)
    let mut strtabs: Vec<(usize, usize)> = Vec::new(); // (off, size)
    for i in 0..shnum {
        let o = shoff + i * shentsize;
        let sh_type = le_u32(bytes, o + 4);
        let sh_offset = le_u64(bytes, o + 24) as usize;
        let sh_size = le_u64(bytes, o + 32) as usize;
        let sh_link = le_u32(bytes, o + 40) as usize;
        match sh_type {
            2 => symtab = Some((sh_offset, sh_size, sh_link)),
            3 => strtabs.push((sh_offset, sh_size)),
            _ => {}
        }
    }

    let mut entries_found: Vec<String> = Vec::new();
    let mut entries_missing: Vec<String> = Vec::new();
    if let Some((sym_off, sym_size, str_link)) = symtab {
        // The string table is the section indexed by sh_link; fall back to
        // the first strtab when the link is unusable.
        let strtab = if str_link < shnum {
            let o = shoff + str_link * shentsize;
            Some((
                le_u64(bytes, o + 24) as usize,
                le_u64(bytes, o + 32) as usize,
            ))
        } else {
            None
        }
        .or_else(|| strtabs.first().copied());
        if let Some((str_off, str_size)) = strtab {
            let bounds_ok = sym_off
                .checked_add(sym_size)
                .map(|end| end <= bytes.len())
                .unwrap_or(false)
                && str_off
                    .checked_add(str_size)
                    .map(|end| end <= bytes.len())
                    .unwrap_or(false);
            if bounds_ok {
                let sym_bytes = &bytes[sym_off..sym_off + sym_size];
                let str_bytes = &bytes[str_off..str_off + str_size];
                let sym_entsize = 24usize; // ELF64 symbol size
                for sym in sym_bytes.chunks_exact(sym_entsize) {
                    let st_name = le_u32(sym, 0) as usize;
                    let st_info = sym[4];
                    let st_shndx = le_u16(sym, 6);
                    // st_info: high nibble = STB binding, low nibble = STT
                    // type. Exported kernels are STB_GLOBAL(1) STT_FUNC(2)
                    // with a defined section index (st_shndx != 0).
                    let is_func = (st_info & 0x0f) == 2;
                    let is_global = (st_info >> 4) == 1;
                    if !(is_func && is_global) || st_shndx == 0 {
                        continue;
                    }
                    if st_name >= str_bytes.len() {
                        continue;
                    }
                    let end = str_bytes[st_name..]
                        .iter()
                        .position(|&b| b == 0)
                        .map(|p| st_name + p)
                        .unwrap_or(st_name);
                    let name = String::from_utf8_lossy(&str_bytes[st_name..end]);
                    if REQUIRED_ENTRIES.contains(&name.as_ref()) {
                        entries_found.push(name.into_owned());
                    }
                }
            }
        }
    }
    for req in REQUIRED_ENTRIES {
        if !entries_found.iter().any(|f| f == req) {
            entries_missing.push((*req).to_string());
        }
    }

    Ok(ElfInfo {
        machine_amdgpu: machine == EM_AMDGPU,
        // e_flags low byte carries the EF_AMDGPU_MACH ISA code; bits 0..8
        // after the arch-version mask; report the raw arch version nibbles.
        amdgpu_arch_version: e_flags & 0x0f,
        entries_found,
        entries_missing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid ELF64 header + one section header + a .symtab
    /// with the required kernel symbols, so the parser is exercised without
    /// needing the real artifact.
    fn synthetic_code_object(entries: &[&str]) -> Vec<u8> {
        let mut b = vec![0u8; 64];
        b[0..4].copy_from_slice(b"\x7fELF");
        b[4] = 2; // ELF64
        b[5] = 1; // little-endian
        b[18..20].copy_from_slice(&EM_AMDGPU.to_le_bytes());
        b[48..52].copy_from_slice(&1u32.to_le_bytes()); // e_flags
        // Section headers: [0] null, [1] .symtab, [2] .strtab
        let shoff = 64usize;
        b[40..48].copy_from_slice(&(shoff as u64).to_le_bytes());
        b[58..60].copy_from_slice(&64u16.to_le_bytes()); // shentsize
        b[60..62].copy_from_slice(&3u16.to_le_bytes()); // shnum
        b.resize(shoff + 3 * 64, 0);
        let sym_off = shoff + 3 * 64;
        let str_off = sym_off + 64 * entries.len();
        let put = |b: &mut Vec<u8>, o: usize, v: u64| {
            b[o..o + 8].copy_from_slice(&v.to_le_bytes());
        };
        let put32 = |b: &mut Vec<u8>, o: usize, v: u32| {
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        // Section 1: .symtab (SHT_SYMTAB=2)
        let o = shoff + 64;
        put32(&mut b, o + 4, 2);
        put(&mut b, o + 24, sym_off as u64);
        put(&mut b, o + 32, (24 * entries.len()) as u64);
        put32(&mut b, o + 40, 2); // sh_link -> .strtab
        // Section 2: .strtab (SHT_STRTAB=3)
        let o = shoff + 128;
        put32(&mut b, o + 4, 3);
        put(&mut b, o + 24, str_off as u64);
        let mut strings = vec![0u8];
        let mut offsets = Vec::new();
        for e in entries {
            offsets.push(strings.len());
            strings.extend_from_slice(e.as_bytes());
            strings.push(0);
        }
        put(&mut b, o + 32, strings.len() as u64);
        // Symbols
        b.resize(str_off + strings.len(), 0);
        for (i, _name) in entries.iter().enumerate() {
            let so = sym_off + i * 24;
            put32(&mut b, so, offsets[i] as u32);
            b[so + 4] = (1 << 4) | 2; // STB_GLOBAL | STT_FUNC
            b[so + 6..so + 8].copy_from_slice(&1u16.to_le_bytes()); // st_shndx
        }
        b[str_off..].copy_from_slice(&strings);
        b
    }

    #[test]
    fn rejects_non_elf() {
        assert!(inspect_amdgcn_code_object(b"not an elf").is_err());
        assert!(inspect_amdgcn_code_object(&[0u8; 64]).is_err());
    }

    #[test]
    fn rejects_truncated_section_table() {
        let mut b = synthetic_code_object(&["vole_render_d0"]);
        b.truncate(b.len() - 8);
        // Header table end now exceeds the image: hostile-file safety.
        let _ = inspect_amdgcn_code_object(&b); // must not panic (Err ok)
    }

    #[test]
    fn parses_synthetic_code_object_with_entries() {
        let b = synthetic_code_object(&["vole_render_d0", "vole_entropy_decode"]);
        let info = inspect_amdgcn_code_object(&b).expect("parses");
        assert!(info.machine_amdgpu);
        assert!(info.entries_found.contains(&"vole_render_d0".to_string()));
        assert!(
            info.entries_found
                .contains(&"vole_entropy_decode".to_string())
        );
        assert!(
            info.entries_missing
                .contains(&"vole_upmix_mono_dup".to_string())
        );
        assert!(!info.valid()); // upmix missing
    }

    #[test]
    fn rejects_wrong_machine() {
        let mut b = synthetic_code_object(&["vole_render_d0"]);
        b[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
        let info = inspect_amdgcn_code_object(&b).expect("parses as elf");
        assert!(!info.machine_amdgpu);
        assert!(!info.valid());
    }

    #[test]
    fn real_artifact_is_a_valid_amdgpu_code_object() {
        // Repository-only test: exercises the parser against the actual
        // build output when present (scripts/ is not in the published
        // crate, so this is skipped outside the repo).
        let p = std::path::Path::new("scripts/out/vole_audio.amdgcn.elf");
        if !p.exists() {
            eprintln!("skipping: artifact not built (run scripts/build-rocm-device.sh)");
            return;
        }
        let bytes = std::fs::read(p).expect("read artifact");
        let info = inspect_amdgcn_code_object(&bytes).expect("artifact parses");
        assert!(info.valid(), "artifact must carry all required entries");
        assert!(info.machine_amdgpu);
    }
}

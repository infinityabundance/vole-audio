//! VOLE-Audio command-line interface.
//!
//! The CLI only ever reports what this build actually supports. Courts and
//! backends land in later phases; until then their commands exit with an
//! explicit `NOT_IMPLEMENTED` status (exit code 3) and a message — the CLI
//! never implies support that is not present.

use std::process::ExitCode;
use vole_audio::error::{Error, Kind, Result};
use vole_audio::evidence::receipt::ReceiptEnvelope;

const USAGE: &str = "\
vole-audio — procedural sampling and direct audio materialization

USAGE:
    vole-audio <command> [args]

COMMANDS (current build):
    probe                 Capture environment + hardware evidence summary
    court <name>          Run an executable court (semantic, authored, simd, facts,
                          inverse, inverse-search, conventional, corpus, fullobj,
                          flattening,
                          cuda, d1, rocm,
                          rocm-d0, rocm-d1, entropy-rans, entropy-literal,
                          entropy-residual, entropy-pages, entropy-partial,
                          entropy-simd, entropy-cuda, entropy-d1, entropyfs,
                          dsfb-entropy, h2)
                          [--receipts DIR]; court d1 accepts --emit-audio (court
                          entropy-d1 honors VOLE_ENTROPY_D1_EMIT_AUDIO=1)
    receipt show <file>   Verify and print an evidence receipt
    receipt perf <file>   Render a receipt's throughput_cells as Markdown
    seal verify           Executable phase-seal gate: validates an explicit
                          expected-verdict matrix over the newest receipts
                          (source_binding == bound, single seal subject +
                          battery tree, frozen semantic/authored hashes, rocm
                          compile surface; verifier subject == receipt
                          subject unless --historical)
                          [--receipts DIR]
                          [--expect court=SUPPORTED|ANY|LABEL|LABEL|...]
                          [--historical]
    seal subject          Print the current seal-subject hash: SHA-256 of the
                          tracked source excluding the evidence/governance
                          trees (receipts/ target/ scripts/out/ docs/ .git/);
                          the identity a seal compares
    version               Print version and build identity
    corpus list           List the frozen flagship corpus objects and classes
    corpus verify         Regenerate every object and prove the frozen manifest
                          (missing/extra/duplicate/reordered/mutated/mis-sized/
                          wrong-rate/wrong-hash/root all fail; exits nonzero on
                          failure) [--receipts DIR]
    corpus freeze         Write the manifest from the frozen membership (the
                          freeze act; run once, deliberately). Refuses to
                          overwrite an existing FROZEN manifest unless the new
                          corpus identity is declared with an explicit reason:
                          [--out PATH] [--amend-frozen REASON]
    help                  Show this help

Planned commands arrive with their phases (inspect/verify/encode/observe/play,
bench, court d2|depth|random-access|negative|interference, and probe
cuda|rocm|alsa|d1|d2). Until implemented they exit with
NOT_IMPLEMENTED (3); the CLI never implies support that is absent.

example:
    vole-audio court simd          # Phase F SIMD parity battery
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<u8> {
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    match cmd {
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(0)
        }
        "version" | "--version" | "-V" => {
            println!("vole-audio {}", env!("CARGO_PKG_VERSION"));
            println!("evidence schema: vole.audio.evidence.v1");
            let e = vole_audio::evidence::environment::Environment::capture();
            if let Some(c) = e.git_identity() {
                println!("executed in worktree: {c}");
            }
            if let Some(b) = &e.build_git_commit {
                println!(
                    "compiled from commit:   {}{}",
                    b,
                    if e.build_dirty == Some(true) {
                        "-dirty"
                    } else {
                        ""
                    }
                );
            }
            println!("source_binding: {}", e.source_binding().label());
            if let Some(s) = &e.seal_subject_hash {
                println!("seal subject: {s}");
            }
            Ok(0)
        }
        "probe" => cmd_probe(&args[2..]),
        "receipt" => cmd_receipt(&args[2..]),
        "court" => cmd_court(&args[2..]),
        "corpus" => cmd_corpus(&args[2..]),
        "seal" => cmd_seal(&args[2..]),
        "inspect" | "verify" | "encode" | "observe" | "play" | "bench" => {
            // Declared-but-not-yet-implemented surface: exit 3 (NOT_IMPLEMENTED);
            // the CLI never implies support that is absent.
            eprintln!(
                "not implemented: command '{cmd}' arrives with its phase; \
                 see docs/PROJECT_STATE.md"
            );
            Ok(3)
        }
        other => {
            eprintln!("error: unknown command '{other}'");
            eprint!("{USAGE}");
            Ok(2)
        }
    }
}

fn cmd_probe(args: &[String]) -> Result<u8> {
    // `probe rocm`: ROCm/AMD presence evidence (Phase I). Other probe
    // targets (cuda/alsa/d1/d2) arrive with their phases.
    if args.first().map(String::as_str) == Some("rocm") {
        let json = args.iter().any(|a| a == "--json");
        let p = vole_audio::backend::rocm::probe::RocmProbe::capture()?;
        let (v, detail) = p.classify();
        let kfd_label = match p.kfd {
            vole_audio::backend::rocm::KfdState::Absent => "absent",
            vole_audio::backend::rocm::KfdState::PresentNotAccessible => "present_not_accessible",
            vole_audio::backend::rocm::KfdState::Accessible => "accessible",
        };
        if json {
            let doc = serde_json::json!({
                "schema": "vole.audio.evidence.v1",
                "kind": "probe-rocm",
                "verdict": v.label(),
                "detail": detail,
                "amd_gpus": p.amd_gpus.iter().map(|g| serde_json::json!({
                    "bdf": g.bdf, "vendor": g.vendor, "device": g.device, "driver": g.driver,
                })).collect::<Vec<_>>(),
                "kfd": kfd_label,
                "compute_runtime": p.compute.iter().map(|a| serde_json::json!({
                    "soname": a.soname,
                    "d0_ready": a.d0_ready(),
                    "d1_ready": a.d1_ready(),
                    "detail": match (&a.d0_missing, &a.d1_missing) {
                        (Ok(m0), Ok(m1)) if m0.is_empty() && m1.is_empty() => None,
                        (Ok(m0), Ok(m1)) if m0.is_empty() => {
                            Some(format!("D0 ready; D1 missing: {}", m1.join(",")))
                        }
                        (Ok(m0), Ok(_)) => Some(format!("D0 missing: {}", m0.join(","))),
                        (Err(e), _) | (_, Err(e)) => Some(e.clone()),
                    },
                })).collect::<Vec<_>>(),
                "d0_readiness": p.compute.iter().any(|a| a.d0_ready()),
                "d1_readiness": p.compute.iter().any(|a| a.d1_ready()),
                "telemetry_rocmsmi": p.telemetry.iter().any(|(_, f)| *f),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&doc)
                    .map_err(|e| Error::new(Kind::Io, e.to_string()))?
            );
        } else {
            println!("vole-audio probe rocm (Phase I evidence)");
            println!("------------------------------------------------");
            println!("verdict:        {}", v.label());
            println!("detail:         {detail}");
            for g in &p.amd_gpus {
                println!(
                    "amd gpu pci:    {} vendor={} device={} driver={}",
                    g.bdf,
                    g.vendor.clone().unwrap_or_default(),
                    g.device.clone().unwrap_or_default(),
                    g.driver.clone().unwrap_or_default(),
                );
            }
            if p.amd_gpus.is_empty() {
                println!("amd gpu pci:    (none found in sysfs)");
            }
            println!("kfd:            {kfd_label}");
            for a in &p.compute {
                match (&a.d0_missing, &a.d1_missing) {
                    (Ok(m0), Ok(m1)) if m0.is_empty() && m1.is_empty() => println!(
                        "compute:        {} loaded (D0 + D1 surfaces resolve)",
                        a.soname
                    ),
                    (Ok(m0), Ok(m1)) if m0.is_empty() => println!(
                        "compute:        {} loaded (D0 ready; D1 missing: {})",
                        a.soname,
                        m1.join(",")
                    ),
                    (Ok(m0), Ok(_)) => println!(
                        "compute:        {} loaded (D0 missing: {})",
                        a.soname,
                        m0.join(",")
                    ),
                    (Err(e), _) | (_, Err(e)) => {
                        println!("compute:        {} not loadable ({e})", a.soname)
                    }
                }
            }
            for (n, found) in &p.telemetry {
                println!(
                    "telemetry:      {n:<24} {}",
                    if *found { "found" } else { "absent" }
                );
            }
            println!("------------------------------------------------");
        }
        return Ok(0);
    }
    let json = args.iter().any(|a| a == "--json");
    let env = vole_audio::evidence::environment::Environment::capture();
    let hw = vole_audio::evidence::hardware::Hardware::capture();
    if json {
        let doc = serde_json::json!({
            "schema": "vole.audio.evidence.v1",
            "kind": "probe",
            "environment": env,
            "hardware": hw,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&doc).map_err(|e| Error::new(Kind::Io, e.to_string()))?
        );
        return Ok(0);
    }
    println!("vole-audio probe (evidence constitution)");
    println!("------------------------------------------------");
    if let Some(id) = env.git_identity() {
        println!("git:            {id}");
    } else {
        println!("git:            (not a git work tree)");
    }
    if let Some(rc) = &env.rustc_version {
        println!("rustc:          {rc}");
    }
    println!("crate:          vole-audio {}", env.crate_version);
    if let Some(os) = &env.os.os_pretty_name {
        println!("os:             {os}");
    }
    if let Some(rel) = &env.os.release {
        println!("kernel:         {rel}");
    }
    if let Some(m) = &env.cpu.model {
        println!("cpu:            {m}");
    }
    let simd = if env.cpu.has_avx512 {
        "avx2+avx512"
    } else if env.cpu.has_avx2 {
        "avx2"
    } else if env.cpu.has_neon {
        "neon"
    } else {
        "scalar"
    };
    println!("simd:           {simd}");
    if let Some(t) = env.memory.total_bytes {
        println!("memory:         {} bytes", t);
    }
    if let Some(g) = &env.cpu_governor {
        println!("governor:       {g}");
    }
    let gpus: Vec<_> = hw
        .pci_devices
        .iter()
        .filter(|d| d.is_display() && (d.is_nvidia() || d.is_amd()))
        .collect();
    for g in &gpus {
        println!(
            "gpu pci:        {} vendor={} device={} driver={}",
            g.bdf,
            g.vendor.clone().unwrap_or_default(),
            g.device.clone().unwrap_or_default(),
            g.driver.clone().unwrap_or_default(),
        );
    }
    if gpus.is_empty() {
        println!("gpu pci:        (none found in sysfs)");
    }
    let audio: Vec<_> = hw.pci_devices.iter().filter(|d| d.is_audio()).collect();
    for a in &audio {
        println!(
            "audio pci:      {} vendor={} device={} driver={}",
            a.bdf,
            a.vendor.clone().unwrap_or_default(),
            a.device.clone().unwrap_or_default(),
            a.driver.clone().unwrap_or_default(),
        );
    }
    if audio.is_empty() {
        println!("audio pci:      (none found in sysfs)");
    }
    println!("------------------------------------------------");
    Ok(0)
}

/// `vole-audio seal verify`: executable phase-seal gate over the newest
/// receipts (see `seal::verify_seal`). Default expectations: the always-expected
/// A–H courts, `inverse`/`flattening` (Phase K), and `h2` SUPPORTED, and `rocm`
/// restricted to the explicit allowed set `UNSUPPORTED_BY_HARDWARE |
/// UNSUPPORTED_BY_API | INCONCLUSIVE` (never the corruption/execution-failure
/// classes). The default invariant is `verifier seal subject == receipt seal
/// subject` (all bound); `--historical` relaxes the verifier requirement and
/// accepts pre-amendment receipts without a subject. `vole-audio seal subject`
/// prints the current subject hash for inspection.
fn cmd_seal(args: &[String]) -> Result<u8> {
    let sub = args.first().map(String::as_str).unwrap_or("verify");
    match sub {
        "verify" => cmd_seal_verify(&args[1..]),
        "subject" => {
            let e = vole_audio::evidence::environment::Environment::capture();
            match &e.seal_subject_hash {
                Some(s) => {
                    println!("{s}");
                    Ok(0)
                }
                None => {
                    eprintln!("no seal subject: not inside a git work tree");
                    Ok(1)
                }
            }
        }
        other => Err(Error::malformed(format!(
            "unknown seal subcommand '{other}'"
        ))),
    }
}

fn cmd_seal_verify(args: &[String]) -> Result<u8> {
    let mut receipts = std::path::PathBuf::from("receipts");
    let mut historical = false;
    let mut expect = "semantic=SUPPORTED,authored=SUPPORTED,simd=SUPPORTED,\
                       facts=SUPPORTED,inverse=SUPPORTED,flattening=SUPPORTED,\
                       inverse-search=SUPPORTED,conventional=SUPPORTED,corpus=SUPPORTED,\
                       fullobj=SUPPORTED,\
                       cuda=SUPPORTED,d1=SUPPORTED,h2=SUPPORTED,\
                       rocm=UNSUPPORTED_BY_HARDWARE|UNSUPPORTED_BY_API|INCONCLUSIVE,\
                       rocm-d0=UNSUPPORTED_BY_HARDWARE|UNSUPPORTED_BY_API|INCONCLUSIVE,\
                       rocm-d1=UNSUPPORTED_BY_HARDWARE|UNSUPPORTED_BY_API|INCONCLUSIVE"
        .to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--receipts" => {
                i += 1;
                receipts = std::path::PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| Error::malformed("--receipts requires a directory"))?,
                );
            }
            "--expect" => {
                i += 1;
                expect = args
                    .get(i)
                    .ok_or_else(|| Error::malformed("--expect requires a matrix string"))?
                    .clone();
            }
            "--historical" => historical = true,
            other => return Err(Error::malformed(format!("unknown seal flag '{other}'"))),
        }
        i += 1;
    }
    let expectations = vole_audio::seal::parse_expectations(&expect)?;
    let verifier = vole_audio::evidence::environment::Environment::capture();
    let (ok, rows) =
        vole_audio::seal::verify_seal(&receipts, &expectations, &verifier, historical)?;
    println!(
        "vole-audio seal verify (receipts root: {})",
        receipts.display()
    );
    println!(
        "  verifier: {} (binding: {}, seal subject: {})",
        verifier
            .git_identity()
            .unwrap_or_else(|| "no git identity".into()),
        verifier.source_binding().label(),
        verifier
            .seal_subject_hash
            .as_deref()
            .unwrap_or("unavailable")
    );
    if historical {
        println!("  mode: --historical (verifier requirement relaxed)");
    }
    println!("------------------------------------------------");
    for row in &rows {
        println!(
            "  {:<12} expected={:<34} got={:<24} {}{}",
            row.court,
            row.expected,
            row.got,
            if row.pass { "PASS" } else { "FAIL" },
            row.note
                .as_ref()
                .map(|n| format!("  [{n}]"))
                .unwrap_or_default()
        );
    }
    if ok {
        println!(
            "seal verified: {} receipts, expected matrix satisfied",
            rows.len()
        );
        Ok(0)
    } else {
        println!("seal NOT verified: the matrix above must all PASS");
        Ok(1)
    }
}

fn cmd_court(args: &[String]) -> Result<u8> {
    let name = args.first().map(String::as_str);
    let Some(name) = name else {
        eprintln!("available courts:");
        for (n, desc) in vole_audio::courts::COURT_NAMES {
            eprintln!("  {n:<14} {desc}");
        }
        return Ok(0);
    };
    let mut receipts = std::path::PathBuf::from("receipts");
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--receipts" => {
                i += 1;
                receipts = std::path::PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| Error::malformed("--receipts requires a directory"))?,
                );
            }
            "--emit-audio" => {
                // Court-specific opt-in: audible content is never the
                // default (courts are silence-safe probes). `court d1` and
                // `court entropy-d1` understand it today; the flag is
                // rejected elsewhere.
                if name == "d1" {
                    // SAFETY: single-threaded CLI setup before any court work.
                    unsafe { std::env::set_var("VOLE_D1_EMIT_AUDIO", "1") };
                } else if name == "entropy-d1" {
                    // SAFETY: single-threaded CLI setup before any court work.
                    unsafe { std::env::set_var("VOLE_ENTROPY_D1_EMIT_AUDIO", "1") };
                } else {
                    return Err(Error::malformed(format!(
                        "--emit-audio is only valid for court d1 / entropy-d1 (got court '{name}')"
                    )));
                }
            }
            other => {
                return Err(Error::malformed(format!("unknown court flag '{other}'")));
            }
        }
        i += 1;
    }
    // Courts record evidence; the process exits 0 when a receipt was written
    // (whatever its verdict), nonzero only on operational failure.
    let verdict = vole_audio::courts::run(name, &receipts)?;
    println!("court {name}: {verdict}");
    Ok(0)
}

fn cmd_corpus(args: &[String]) -> Result<u8> {
    match args.first().map(String::as_str) {
        None | Some("list") => {
            let m = vole_audio::corpus::manifest()?;
            println!(
                "flagship corpus: {} objects ({} B1-comparable, {} excluded by format domain)",
                m.populations.whole_corpus_objects,
                m.populations.b1_comparable_objects,
                m.populations.b1_excluded_objects
            );
            println!("  schema: {}  state: {}", m.schema, m.state);
            println!("  corpus sha256: {}", m.corpus_sha256);
            for o in &m.objects {
                println!(
                    "  {:<44} {:>6} Hz {:>2}ch {:>8} f  {}/{}/{}/{}/{}  {}",
                    o.id,
                    o.sample_rate_hz,
                    o.channels,
                    o.frames,
                    o.source_structure_class,
                    o.amplitude_class,
                    o.channel_structure,
                    o.temporal_class,
                    o.entropy_class,
                    if o.b1_comparable { "B1" } else { "B1-N/A" },
                );
            }
            Ok(0)
        }
        Some("verify") => {
            let mut receipts = std::path::PathBuf::from("receipts");
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--receipts" => {
                        i += 1;
                        receipts =
                            std::path::PathBuf::from(args.get(i).ok_or_else(|| {
                                Error::malformed("--receipts requires a directory")
                            })?);
                    }
                    other => {
                        return Err(Error::malformed(format!("unknown corpus flag '{other}'")));
                    }
                }
                i += 1;
            }
            // Unlike a court, this is a verification command: a failed corpus
            // gate exits nonzero so a script cannot ignore it. The receipt is
            // written either way.
            let verdict = vole_audio::courts::corpus::run(&receipts)?;
            match verdict {
                vole_audio::status::Verdict::Supported => Ok(0),
                other => {
                    eprintln!("corpus verify: {other}");
                    Ok(1)
                }
            }
        }
        Some("freeze") => {
            let mut out = std::path::PathBuf::from("corpus/manifest.json");
            let mut amend: Option<String> = None;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--out" => {
                        i += 1;
                        out = std::path::PathBuf::from(
                            args.get(i)
                                .ok_or_else(|| Error::malformed("--out requires a path"))?,
                        );
                    }
                    "--amend-frozen" => {
                        i += 1;
                        let reason = args
                            .get(i)
                            .ok_or_else(|| Error::malformed("--amend-frozen requires a reason"))?;
                        if reason.trim().is_empty() {
                            return Err(Error::malformed(
                                "--amend-frozen reason must not be empty",
                            ));
                        }
                        amend = Some(reason.clone());
                    }
                    other => {
                        return Err(Error::malformed(format!("unknown corpus flag '{other}'")));
                    }
                }
                i += 1;
            }

            // Freezing is a one-time act. Silently overwriting a frozen manifest
            // would let a population be retuned after results exist, so it is
            // refused unless the operator deliberately defines a new corpus
            // identity with an explicit reason.
            if out.exists() {
                let existing = std::fs::read_to_string(&out).map_err(|e| {
                    Error::internal(format!(
                        "cannot read existing manifest {}: {e}",
                        out.display()
                    ))
                })?;
                let state = serde_json::from_str::<vole_audio::corpus::Manifest>(&existing)
                    .ok()
                    .map(|m| m.state);
                match state.as_deref() {
                    Some(vole_audio::corpus::STATE_FROZEN) if amend.is_none() => {
                        return Err(Error::malformed(format!(
                            "refusing to overwrite the frozen manifest at {} (state FROZEN): \
                             amend it deliberately with --amend-frozen <reason>, which defines \
                             a new corpus identity",
                            out.display()
                        )));
                    }
                    None if amend.is_none() => {
                        return Err(Error::malformed(format!(
                            "existing manifest {} does not parse; remove it deliberately or \
                             pass --amend-frozen <reason>",
                            out.display()
                        )));
                    }
                    _ => {}
                }
            }

            let specs = vole_audio::corpus::specs::specs();
            let m = vole_audio::corpus::frozen_manifest(&specs)?;
            let json = serde_json::to_string_pretty(&m)?;
            std::fs::write(&out, format!("{json}\n"))?;
            println!(
                "corpus freeze: wrote {} objects to {}",
                m.objects.len(),
                out.display()
            );
            if let Some(reason) = &amend {
                println!("  amendment: {reason}");
            }
            println!(
                "  populations: whole {} / b1-comparable {} / excluded {}",
                m.populations.whole_corpus_objects,
                m.populations.b1_comparable_objects,
                m.populations.b1_excluded_objects
            );
            println!("  corpus sha256: {}", m.corpus_sha256);
            println!(
                "  manifest sha256: {}",
                vole_audio::hash::sha256::hex(&vole_audio::corpus::manifest_sha256(&m))
            );
            Ok(0)
        }
        Some(other) => Err(Error::malformed(format!(
            "unknown corpus subcommand '{other}' (list | verify | freeze)"
        ))),
    }
}

fn cmd_receipt(args: &[String]) -> Result<u8> {
    let sub = args.first().map(String::as_str).unwrap_or("show");
    match sub {
        "show" => {
            let path = args
                .get(1)
                .ok_or_else(|| Error::malformed("receipt show <file>"))?;
            let bytes = std::fs::read(path)?;
            let env: ReceiptEnvelope = ReceiptEnvelope::from_json_bytes(&bytes)?;
            let r = &env.receipt;
            println!("schema:         {}", r.schema);
            println!("run_id:         {}", r.run_id);
            println!("court:          {}", r.court);
            println!("result:         {}", r.result);
            if let Some(d) = &r.result_detail {
                println!("detail:         {d}");
            }
            if let Some(b) = &r.params.backend {
                println!("backend:        {b}");
            }
            if let Some(u) = &r.params.universe {
                println!("universe:       {u}");
            }
            println!("self-hash:      {}", env.receipt_sha256);
            if let Some(gh) = &r.provenance.reference_hash {
                println!("reference sha:  {gh}");
            }
            if let Some(gh) = &r.provenance.backend_hash {
                println!("backend sha:    {gh}");
            }
            if let Some(c) = &r.provenance.gpu_artifact_hash {
                println!("gpu artifact:   {c}");
            }
            println!(
                "git commit:     {}",
                r.environment.git_identity().unwrap_or_else(|| "?".into())
            );
            Ok(0)
        }
        "perf" => {
            // Render the `throughput_cells` table of a court receipt as
            // Markdown, so PERFORMANCE.md tables are generated from the
            // receipt instead of being transcribed by hand.
            let path = args
                .get(1)
                .ok_or_else(|| Error::malformed("receipt perf <file>"))?;
            let bytes = std::fs::read(path)?;
            let env: ReceiptEnvelope = ReceiptEnvelope::from_json_bytes(&bytes)?;
            let cells = env
                .receipt
                .extras
                .get("throughput_cells")
                .ok_or_else(|| Error::malformed("receipt has no throughput_cells"))?;
            let map = cells
                .as_object()
                .ok_or_else(|| Error::malformed("throughput_cells is not an object"))?;
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let fmt = |v: &serde_json::Value| -> String {
                v.as_f64()
                    .map(|f| format!("{f:.3}"))
                    .unwrap_or_else(|| "—".to_string())
            };
            println!("| cell | scalar | avx2 | avx512 | cuda-d0 wall | cuda kernel |");
            println!("|---|---:|---:|---:|---:|---:|");
            for k in keys {
                let row = map[k].as_object().expect("cell object");
                let get = |s: &str| -> String {
                    row.get(s)
                        .and_then(|v| v.get("mean_ms").or_else(|| v.get("wall_mean_ms")))
                        .map(fmt)
                        .unwrap_or_else(|| "—".to_string())
                };
                let kernel = row
                    .get("cuda-d0")
                    .and_then(|v| v.get("kernel_mean_ms"))
                    .map(fmt)
                    .unwrap_or_else(|| "—".to_string());
                println!(
                    "| {k} | {} | {} | {} | {} | {kernel} |",
                    get("scalar"),
                    get("avx2"),
                    get("avx512"),
                    get("cuda-d0")
                );
            }
            println!();
            println!(
                "_generated by `vole-audio receipt perf` from {} (run {}; {})_",
                path,
                env.receipt.run_id,
                env.receipt
                    .environment
                    .git_identity()
                    .unwrap_or_else(|| "?".into())
            );
            Ok(0)
        }
        other => Err(Error::malformed(format!(
            "unknown receipt subcommand '{other}' (expected: show)"
        ))),
    }
}

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
                          cuda, d1, entropy-rans, entropy-literal, entropy-residual,
                          entropy-pages, entropy-partial, entropy-cuda, entropy-d1,
                          entropyfs, dsfb-entropy)
                          [--receipts DIR]; court d1 accepts --emit-audio (court
                          entropy-d1 honors VOLE_ENTROPY_D1_EMIT_AUDIO=1)
    receipt show <file>   Verify and print an evidence receipt
    receipt perf <file>   Render a receipt's throughput_cells as Markdown
    version               Print version and build identity
    help                  Show this help

Planned commands arrive with their phases (inspect/verify/encode/observe/play,
bench/corpus, court inverse|flattening|rocm|d1|d2|depth|conventional|
random-access|negative|interference, and probe cuda|rocm|alsa|d1|d2). Until
implemented they exit with NOT_IMPLEMENTED (3); the CLI never implies support
that is absent.

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
            Ok(0)
        }
        "probe" => cmd_probe(&args[2..]),
        "receipt" => cmd_receipt(&args[2..]),
        "court" => cmd_court(&args[2..]),
        "inspect" | "verify" | "encode" | "observe" | "play" | "bench" | "corpus" => {
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

/// Run an executable court: `vole-audio court <name> [--receipts DIR]`.
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

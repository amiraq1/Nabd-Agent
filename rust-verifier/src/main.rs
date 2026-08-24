use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::exit;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(name = "verifier", about = "Nabd Agent V2 file verification tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Calculate SHA256 hash of a file
    Hash { path: String },

    /// Verify file exists and matches expected hash
    Verify { path: String, expected_hash: String },

    /// Check if path is inside workspace
    Jail { workspace: String, path: String },

    /// Create timestamped backup of a file
    Backup {
        path: String,
        #[arg(short, long)]
        dir: Option<String>,
    },
}

fn compute_sha256(path: &Path) -> Result<String, String> {
    let content = fs::read(path).map_err(|e| format!("read failed: {}", e))?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    Ok(format!("{:x}", hasher.finalize()))
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_secs()
}

fn cmd_hash(path: &str) {
    match compute_sha256(Path::new(path)) {
        Ok(h) => {
            println!("{}", h);
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            exit(1);
        }
    }
}

fn cmd_verify(path: &str, expected: &str) {
    let p = Path::new(path);
    if !p.exists() {
        eprintln!("ERROR: file does not exist: {}", path);
        exit(1);
    }
    match compute_sha256(p) {
        Ok(actual) => {
            if actual == expected {
                println!("OBSERVED");
            } else {
                println!("MISMATCH");
                eprintln!("expected: {}", expected);
                eprintln!("actual:   {}", actual);
                exit(1);
            }
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            exit(1);
        }
    }
}

fn cmd_jail(workspace: &str, path: &str) {
    let ws = match fs::canonicalize(workspace) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: workspace: {}", e);
            exit(1);
        }
    };

    // For non-existent paths, canonicalize the deepest existing ancestor
    let target = Path::new(path);
    let resolved = if target.exists() {
        match fs::canonicalize(target) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("ERROR: path: {}", e);
                exit(1);
            }
        }
    } else {
        // Walk up to find existing parent
        let mut current = target.to_path_buf();
        loop {
            if current.exists() {
                match fs::canonicalize(&current) {
                    Ok(p) => break p,
                    Err(e) => {
                        eprintln!("ERROR: parent: {}", e);
                        exit(1);
                    }
                }
            }
            if !current.pop() {
                eprintln!("ERROR: no existing ancestor found");
                exit(1);
            }
        }
    };

    if resolved.starts_with(&ws) {
        println!("SAFE");
    } else {
        println!("UNSAFE");
        exit(1);
    }
}

fn cmd_backup(path: &str, backup_dir: Option<&str>) {
    let p = Path::new(path);
    if !p.exists() {
        eprintln!("ERROR: file does not exist: {}", path);
        exit(1);
    }

    let ts = get_timestamp();
    let fname = p.file_name().unwrap().to_string_lossy();
    let backup_name = format!("{}.backup.{}", fname, ts);

    let backup_path = match backup_dir {
        Some(d) => {
            let dir = Path::new(d);
            if !dir.exists() {
                fs::create_dir_all(dir).unwrap_or_else(|e| {
                    eprintln!("ERROR: create backup dir: {}", e);
                    exit(1);
                });
            }
            dir.join(backup_name)
        }
        None => p.parent().unwrap().join(backup_name),
    };

    match fs::copy(p, &backup_path) {
        Ok(_) => println!("{}", backup_path.display()),
        Err(e) => {
            eprintln!("ERROR: backup failed: {}", e);
            exit(1);
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Hash { path } => cmd_hash(&path),
        Commands::Verify { path, expected_hash } => cmd_verify(&path, &expected_hash),
        Commands::Jail { workspace, path } => cmd_jail(&workspace, &path),
        Commands::Backup { path, dir } => cmd_backup(&path, dir.as_deref()),
    }
}


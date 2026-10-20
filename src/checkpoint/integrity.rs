use super::schema::{Envelope, FORMAT_VERSION, MAX_CHECKPOINT_BYTES, Payload};
use crate::SimError;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) fn read_payload(path: &Path) -> Result<Payload, SimError> {
    let file = File::open(path).map_err(|error| io_error("buka checkpoint", error))?;
    let metadata = file
        .metadata()
        .map_err(|error| io_error("baca metadata checkpoint", error))?;
    if metadata.len() > MAX_CHECKPOINT_BYTES {
        return Err(invalid_checkpoint(
            "checkpoint melebihi batas ukuran 256 MiB",
        ));
    }
    let mut limited = file.take(MAX_CHECKPOINT_BYTES + 1);
    let mut bytes = Vec::new();
    limited
        .read_to_end(&mut bytes)
        .map_err(|error| io_error("baca checkpoint", error))?;
    if bytes.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(invalid_checkpoint(
            "checkpoint melebihi batas ukuran 256 MiB",
        ));
    }
    let envelope: Envelope =
        serde_json::from_slice(&bytes).map_err(|error| invalid_checkpoint(error.to_string()))?;
    if envelope.format_version != FORMAT_VERSION {
        return Err(SimError::UnsupportedCheckpointVersion {
            version: envelope.format_version,
        });
    }
    let expected = format!("{:016x}", fnv1a64(envelope.payload_json.as_bytes()));
    if expected != envelope.checksum_fnv64 {
        return Err(invalid_checkpoint("checksum payload tidak cocok"));
    }
    serde_json::from_str(&envelope.payload_json)
        .map_err(|error| invalid_checkpoint(error.to_string()))
}

pub(super) fn write_payload(path: &Path, payload_json: String) -> Result<(), SimError> {
    let envelope = Envelope {
        format_version: FORMAT_VERSION,
        checksum_fnv64: format!("{:016x}", fnv1a64(payload_json.as_bytes())),
        payload_json,
    };
    let bytes = serde_json::to_vec(&envelope).map_err(|error| SimError::SerializationFailed {
        subsystem: "checkpoint",
        reason: error.to_string(),
    })?;
    if bytes.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(invalid_checkpoint(
            "checkpoint melebihi batas ukuran 256 MiB",
        ));
    }
    atomic_write(path, &bytes, "checkpoint", "tulis checkpoint atomik")
}

pub(crate) fn atomic_write(
    path: &Path,
    bytes: &[u8],
    subsystem: &'static str,
    operation: &'static str,
) -> Result<(), SimError> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path.file_name().ok_or_else(|| SimError::CorruptData {
        subsystem,
        reason: "jalur tujuan tidak memiliki nama file".to_owned(),
    })?;
    let mut temporary = PathBuf::from(parent);
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut temporary_name = file_name.to_os_string();
    temporary_name.push(format!(".tmp.{}.{}", std::process::id(), sequence));
    temporary.push(temporary_name);

    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options
            .open(&temporary)
            .map_err(|error| io_error(operation, error))?;
        file.write_all(bytes)
            .map_err(|error| io_error(operation, error))?;
        file.sync_all()
            .map_err(|error| io_error(operation, error))?;
        drop(file);
        fs::rename(&temporary, path).map_err(|error| io_error(operation, error))?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub(super) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub(super) fn invalid_checkpoint(reason: impl Into<String>) -> SimError {
    SimError::InvalidCheckpoint {
        reason: reason.into(),
    }
}

pub(super) fn io_error(operation: &'static str, error: std::io::Error) -> SimError {
    SimError::Io {
        operation,
        kind: error.kind(),
    }
}

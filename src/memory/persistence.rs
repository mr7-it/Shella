use super::record::StoreData;
use crate::SimError;
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

pub(super) const FORMAT_VERSION: u32 = 1;
const MAX_STORE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FIELD_BYTES: usize = 16 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Serialize, Deserialize)]
struct Envelope {
    format_version: u32,
    checksum_fnv64: String,
    payload_json: String,
}

pub(super) fn validate_field(name: &'static str, value: &str) -> Result<(), SimError> {
    if value.trim().is_empty()
        || value.len() > MAX_FIELD_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(SimError::InvalidInput {
            parameter: name,
            reason: "tidak boleh kosong/kontrol dan harus <= 16 KiB",
        });
    }
    Ok(())
}

pub(super) fn normalize_tags(tags: &[String]) -> Result<Vec<String>, SimError> {
    let mut result = tags.to_vec();
    for tag in &result {
        validate_field("tag", tag)?;
    }
    result.sort();
    result.dedup();
    Ok(result)
}

pub(super) fn validate_store(data: &StoreData) -> Result<(), SimError> {
    if data.format_version != FORMAT_VERSION || data.next_id == 0 {
        return Err(invalid("versi atau penghitung ID database tidak sah"));
    }
    let mut ids = std::collections::BTreeSet::new();
    for (key, record) in &data.records {
        validate_field("key", key).map_err(|error| invalid(error.to_string()))?;
        if key != &record.key || record.id.is_empty() || !ids.insert(record.id.as_str()) {
            return Err(invalid(
                "kunci dan ID catatan tidak konsisten atau duplikat",
            ));
        }
        for tag in &record.tags {
            validate_field("tag", tag).map_err(|error| invalid(error.to_string()))?;
        }
        if record.tags.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid("tag record tidak unik dan terurut"));
        }
    }
    Ok(())
}

pub(super) fn read_store(path: &Path) -> Result<StoreData, SimError> {
    let file = File::open(path).map_err(|error| io_error("buka memori", error))?;
    if file
        .metadata()
        .map_err(|error| io_error("metadata memori", error))?
        .len()
        > MAX_STORE_BYTES
    {
        return Err(invalid("file memori melebihi batas 64 MiB"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_STORE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error("baca memori", error))?;
    if bytes.len() as u64 > MAX_STORE_BYTES {
        return Err(invalid("file memori melebihi batas 64 MiB"));
    }
    let envelope: Envelope =
        serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
    if envelope.format_version != FORMAT_VERSION {
        return Err(SimError::UnsupportedDataVersion {
            subsystem: "memori",
            version: envelope.format_version,
        });
    }
    if envelope.checksum_fnv64 != format!("{:016x}", checksum(envelope.payload_json.as_bytes())) {
        return Err(invalid("checksum database memori tidak cocok"));
    }
    let data: StoreData =
        serde_json::from_str(&envelope.payload_json).map_err(|error| invalid(error.to_string()))?;
    validate_store(&data)?;
    Ok(data)
}

pub(super) fn persist_store(path: &Path, data: &StoreData) -> Result<(), SimError> {
    validate_store(data)?;
    let payload_json =
        serde_json::to_string(data).map_err(|error| SimError::SerializationFailed {
            subsystem: "memori",
            reason: error.to_string(),
        })?;
    let envelope = Envelope {
        format_version: FORMAT_VERSION,
        checksum_fnv64: format!("{:016x}", checksum(payload_json.as_bytes())),
        payload_json,
    };
    let bytes = serde_json::to_vec(&envelope).map_err(|error| SimError::SerializationFailed {
        subsystem: "memori",
        reason: error.to_string(),
    })?;
    if bytes.len() as u64 > MAX_STORE_BYTES {
        return Err(invalid("database memori melebihi batas 64 MiB"));
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| invalid("path database tidak memiliki nama file"))?;
    let id = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut temp_name = file_name.to_os_string();
    temp_name.push(format!(".tmp.{}.{}", std::process::id(), id));
    let temp_path = parent.join(temp_name);
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options
            .open(&temp_path)
            .map_err(|error| io_error("buat file sementara memori", error))?;
        file.write_all(&bytes)
            .map_err(|error| io_error("tulis memori", error))?;
        file.sync_all()
            .map_err(|error| io_error("sinkronkan memori", error))?;
        drop(file);
        fs::rename(&temp_path, path).map_err(|error| io_error("pasang memori atomik", error))?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |mut hash, byte| {
            hash ^= u64::from(*byte);
            hash.wrapping_mul(0x0000_0100_0000_01b3)
        })
}
fn invalid(reason: impl Into<String>) -> SimError {
    SimError::CorruptData {
        subsystem: "memori",
        reason: reason.into(),
    }
}
fn io_error(operation: &'static str, error: std::io::Error) -> SimError {
    SimError::Io {
        operation,
        kind: error.kind(),
    }
}

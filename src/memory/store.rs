use super::{
    persistence::{self, FORMAT_VERSION, validate_field},
    record::{MemoryRecord, RecordId, StoreData},
};
use crate::SimError;
use fs2::FileExt;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

/// Database file-backed dengan satu penulis lintas proses.
///
/// Setiap mutasi menulis snapshot baru ke file sementara, `sync_all`, lalu `rename`;
/// state dalam memori baru di-commit setelah persist berhasil. Lock advisory pada
/// sidecar dipertahankan selama store terbuka; proses lain yang membuka path sama
/// menerima `WouldBlock`. Penghapusan hanya terjadi saat `delete`; retensi otomatis
/// tidak diterapkan.
#[derive(Debug)]
pub struct MemoryStore {
    pub(super) path: PathBuf,
    pub(super) data: StoreData,
    _lock_file: File,
}

impl MemoryStore {
    pub(crate) fn checkpoint_fingerprint(&self) -> Result<String, SimError> {
        let bytes =
            serde_json::to_vec(&self.data).map_err(|error| SimError::SerializationFailed {
                subsystem: "memory store fingerprint",
                reason: error.to_string(),
            })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Membuka database eksklusif atau membuat database kosong pada path yang diberikan.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SimError> {
        let path = normalize_path(path.as_ref())?;
        let mut lock_path = path.as_os_str().to_os_string();
        lock_path.push(".lock");
        let lock_file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(PathBuf::from(lock_path))
            .map_err(|error| SimError::Io {
                operation: "buka lock memori",
                kind: error.kind(),
            })?;
        lock_file
            .try_lock_exclusive()
            .map_err(|error| SimError::Io {
                operation: "kunci memori",
                kind: error.kind(),
            })?;

        if path.exists() {
            let data = persistence::read_store(&path)?;
            let mut store = Self {
                path,
                data,
                _lock_file: lock_file,
            };
            store.purge_expired()?;
            Ok(store)
        } else {
            let data = StoreData {
                format_version: FORMAT_VERSION,
                next_id: 1,
                records: BTreeMap::new(),
            };
            persistence::persist_store(&path, &data)?;
            Ok(Self {
                path,
                data,
                _lock_file: lock_file,
            })
        }
    }

    /// Menyimpan atau memperbarui catatan. Upsert mempertahankan ID catatan.
    pub fn put(
        &mut self,
        key: impl Into<String>,
        value: Value,
        tags: &[String],
    ) -> Result<RecordId, SimError> {
        self.put_with_embedding(key, value, tags, None)
    }

    /// Menyimpan record dengan embedding numerik yang akan dipakai retrieval semantik.
    pub fn put_with_embedding(
        &mut self,
        key: impl Into<String>,
        value: Value,
        tags: &[String],
        embedding: Option<&[f32]>,
    ) -> Result<RecordId, SimError> {
        let key = key.into();
        validate_field("key", &key)?;
        let tags = persistence::normalize_tags(tags)?;
        if let Some(embedding) = embedding {
            if embedding.is_empty() || embedding.iter().any(|value| !value.is_finite()) {
                return Err(SimError::InvalidInput {
                    parameter: "embedding",
                    reason: "harus non-kosong dan semua nilai finite",
                });
            }
        }
        let mut next = self.data.clone();
        let id = if let Some(existing) = next.records.get(&key) {
            existing.id.clone()
        } else {
            let current = next.next_id;
            next.next_id = current.checked_add(1).ok_or(SimError::CounterOverflow {
                counter: "ID memori",
            })?;
            format!("rec-{current:016x}")
        };
        next.records.insert(
            key.clone(),
            MemoryRecord {
                id: id.clone(),
                key,
                value,
                tags,
                embedding: embedding.map(ToOwned::to_owned),
                expires_at_unix: None,
            },
        );
        self.commit(next)?;
        Ok(id)
    }

    /// Menyimpan record dengan TTL absolut dalam detik Unix.
    pub fn put_with_ttl(
        &mut self,
        key: impl Into<String>,
        value: Value,
        tags: &[String],
        ttl: Duration,
    ) -> Result<RecordId, SimError> {
        let key = key.into();
        let id = self.put(key.clone(), value, tags)?;
        let expires_at_unix =
            now_unix()
                .checked_add(ttl.as_secs())
                .ok_or(SimError::CounterOverflow {
                    counter: "TTL memori",
                })?;
        let mut next = self.data.clone();
        if let Some(record) = next.records.get_mut(&key) {
            record.expires_at_unix = Some(expires_at_unix.max(now_unix() + 1));
        }
        self.commit(next)?;
        Ok(id)
    }

    /// Menghapus record yang telah melewati TTL dan mengembalikan jumlah penghapusan.
    pub fn purge_expired(&mut self) -> Result<usize, SimError> {
        let now = now_unix();
        let expired = self
            .data
            .records
            .iter()
            .filter(|(_, record)| record.is_expired(now))
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        if expired.is_empty() {
            return Ok(0);
        }
        let mut next = self.data.clone();
        for key in &expired {
            next.records.remove(key);
        }
        self.commit(next)?;
        Ok(expired.len())
    }

    /// Menghapus record; nilai kembalian menandakan apakah key ditemukan.
    pub fn delete(&mut self, key: &str) -> Result<bool, SimError> {
        if !self.data.records.contains_key(key) {
            return Ok(false);
        }
        let mut next = self.data.clone();
        next.records.remove(key);
        self.commit(next)?;
        Ok(true)
    }

    fn commit(&mut self, next: StoreData) -> Result<(), SimError> {
        persistence::validate_store(&next)?;
        persistence::persist_store(&self.path, &next)?;
        self.data = next;
        Ok(())
    }
}

impl Drop for MemoryStore {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self._lock_file);
    }
}

fn normalize_path(path: &Path) -> Result<PathBuf, SimError> {
    let resolved = if path.exists() {
        path.canonicalize()
    } else {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let name = path.file_name().ok_or(SimError::InvalidInput {
            parameter: "memory_path",
            reason: "harus menunjuk ke nama file",
        })?;
        parent.canonicalize().map(|parent| parent.join(name))
    };
    resolved.map_err(|error| SimError::Io {
        operation: "normalisasi path memori",
        kind: error.kind(),
    })
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

use super::{
    persistence::{self, FORMAT_VERSION, validate_field},
    record::{MemoryRecord, RecordId, StoreData},
};
use crate::SimError;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// Database file-backed satu proses/satu penulis.
///
/// Setiap mutasi menulis snapshot baru ke file sementara, `sync_all`, lalu `rename`;
/// state dalam memori baru di-commit setelah persist berhasil. Beberapa writer proses
/// paralel pada path yang sama tidak didukung. Penghapusan hanya terjadi saat `delete`;
/// retensi otomatis tidak diterapkan.
#[derive(Debug)]
pub struct MemoryStore {
    pub(super) path: PathBuf,
    pub(super) data: StoreData,
}

impl MemoryStore {
    /// Membuka database lama atau membuat database kosong pada path yang diberikan.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SimError> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            let data = persistence::read_store(&path)?;
            Ok(Self { path, data })
        } else {
            let data = StoreData {
                format_version: FORMAT_VERSION,
                next_id: 1,
                records: BTreeMap::new(),
            };
            persistence::persist_store(&path, &data)?;
            Ok(Self { path, data })
        }
    }

    /// Menyimpan atau memperbarui catatan. Upsert mempertahankan ID catatan.
    pub fn put(
        &mut self,
        key: impl Into<String>,
        value: Value,
        tags: &[String],
    ) -> Result<RecordId, SimError> {
        let key = key.into();
        validate_field("key", &key)?;
        let tags = persistence::normalize_tags(tags)?;
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
            },
        );
        self.commit(next)?;
        Ok(id)
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

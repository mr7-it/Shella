use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// ID catatan yang stabil dan tidak berubah saat nilai pada kunci yang sama diperbarui.
pub type RecordId = String;

/// Catatan persistensi yang dapat ditelusuri dan dicari menurut key atau tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub(super) id: RecordId,
    pub(super) key: String,
    pub(super) value: Value,
    pub(super) tags: Vec<String>,
    #[serde(default)]
    pub(super) embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub(super) expires_at_unix: Option<u64>,
}

impl MemoryRecord {
    /// ID stabil.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Kunci unik catatan.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }
    /// Nilai JSON tersimpan.
    #[must_use]
    pub const fn value(&self) -> &Value {
        &self.value
    }
    /// Tag asosiatif catatan.
    #[must_use]
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Embedding terhitung saat record ditulis, bila tersedia.
    #[must_use]
    pub fn embedding(&self) -> Option<&[f32]> {
        self.embedding.as_deref()
    }

    #[must_use]
    pub fn expires_at_unix(&self) -> Option<u64> {
        self.expires_at_unix
    }

    #[must_use]
    pub fn is_expired(&self, now_unix: u64) -> bool {
        self.expires_at_unix
            .is_some_and(|expires| expires <= now_unix)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct StoreData {
    pub(super) format_version: u32,
    pub(super) next_id: u64,
    pub(super) records: BTreeMap<String, MemoryRecord>,
}

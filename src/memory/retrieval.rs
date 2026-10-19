use super::record::MemoryRecord;
use super::store::MemoryStore;
use std::time::{SystemTime, UNIX_EPOCH};

impl MemoryStore {
    /// Mengambil record dengan embedding paling mirip secara deterministik.
    #[must_use]
    pub fn search_embedding(&self, query: &[f32], limit: usize) -> Vec<(&MemoryRecord, f64)> {
        if query.is_empty() || query.iter().any(|value| !value.is_finite()) || limit == 0 {
            return Vec::new();
        }
        let query_norm = query
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if query_norm == 0.0 {
            return Vec::new();
        }
        let mut matches = self
            .data
            .records
            .values()
            .filter(|record| !record.is_expired(now_unix()))
            .filter_map(|record| {
                let embedding = record.embedding.as_ref()?;
                if embedding.len() != query.len() {
                    return None;
                }
                let dot = embedding
                    .iter()
                    .zip(query)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>();
                let norm = embedding
                    .iter()
                    .map(|value| f64::from(*value) * f64::from(*value))
                    .sum::<f64>()
                    .sqrt();
                (norm > 0.0).then_some((record, (dot / (norm * query_norm)).clamp(0.0, 1.0)))
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.id.cmp(&right.0.id))
        });
        matches.truncate(limit);
        matches
    }

    /// Membaca record berdasarkan kunci unik.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MemoryRecord> {
        self.data
            .records
            .get(key)
            .filter(|record| !record.is_expired(now_unix()))
    }

    /// Membaca record berdasarkan ID stabil.
    #[must_use]
    pub fn get_by_id(&self, id: &str) -> Option<&MemoryRecord> {
        self.data
            .records
            .values()
            .find(|record| record.id == id && !record.is_expired(now_unix()))
    }

    /// Mengambil record dengan prefiks key, berurutan leksikografis.
    #[must_use]
    pub fn search_prefix(&self, prefix: &str) -> Vec<&MemoryRecord> {
        self.data
            .records
            .range(prefix.to_owned()..)
            .take_while(|(key, _)| key.starts_with(prefix))
            .filter(|(_, record)| !record.is_expired(now_unix()))
            .map(|(_, record)| record)
            .collect()
    }

    /// Mengambil record yang memiliki tag persis sama, dalam urutan key.
    #[must_use]
    pub fn search_tag(&self, tag: &str) -> Vec<&MemoryRecord> {
        self.data
            .records
            .values()
            .filter(|record| {
                !record.is_expired(now_unix()) && record.tags.iter().any(|item| item == tag)
            })
            .collect()
    }

    /// Banyak record.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data
            .records
            .values()
            .filter(|record| !record.is_expired(now_unix()))
            .count()
    }
    /// Apakah database kosong.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Snapshot baca-saja semua record terurut key.
    pub fn records(&self) -> impl Iterator<Item = &MemoryRecord> {
        self.data
            .records
            .values()
            .filter(|record| !record.is_expired(now_unix()))
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

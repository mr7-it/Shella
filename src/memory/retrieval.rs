use super::record::MemoryRecord;
use super::store::MemoryStore;

impl MemoryStore {
    /// Membaca record berdasarkan kunci unik.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MemoryRecord> {
        self.data.records.get(key)
    }

    /// Membaca record berdasarkan ID stabil.
    #[must_use]
    pub fn get_by_id(&self, id: &str) -> Option<&MemoryRecord> {
        self.data.records.values().find(|record| record.id == id)
    }

    /// Mengambil record dengan prefiks key, berurutan leksikografis.
    #[must_use]
    pub fn search_prefix(&self, prefix: &str) -> Vec<&MemoryRecord> {
        self.data
            .records
            .range(prefix.to_owned()..)
            .take_while(|(key, _)| key.starts_with(prefix))
            .map(|(_, record)| record)
            .collect()
    }

    /// Mengambil record yang memiliki tag persis sama, dalam urutan key.
    #[must_use]
    pub fn search_tag(&self, tag: &str) -> Vec<&MemoryRecord> {
        self.data
            .records
            .values()
            .filter(|record| record.tags.iter().any(|item| item == tag))
            .collect()
    }

    /// Banyak record.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.records.len()
    }
    /// Apakah database kosong.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.records.is_empty()
    }
    /// Snapshot baca-saja semua record terurut key.
    pub fn records(&self) -> impl Iterator<Item = &MemoryRecord> {
        self.data.records.values()
    }
}

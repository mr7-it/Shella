//! Penyimpanan catatan JSON persisten dengan write-atomic dan pencarian deterministik.
mod embedding;
mod persistence;
mod record;
mod retrieval;
mod secure;
mod store;

pub use embedding::TextEmbedding;
pub use record::{MemoryRecord, RecordId};
pub use secure::SecureMemoryStore;
pub use store::MemoryStore;

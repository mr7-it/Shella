//! Penyimpanan catatan JSON persisten dengan write-atomic dan pencarian deterministik.
mod persistence;
mod record;
mod retrieval;
mod store;

pub use record::{MemoryRecord, RecordId};
pub use store::MemoryStore;

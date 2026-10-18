//! Checkpoint JSON versioned untuk state neuron, jaringan, dan evaluator.
mod integrity;
mod restore;
mod save;
mod schema;

pub use schema::RestoredState;

/// Operasi simpan/pulihkan checkpoint versi 1.
pub struct Checkpoint;

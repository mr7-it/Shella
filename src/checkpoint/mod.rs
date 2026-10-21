//! Checkpoint JSON versioned untuk state neuron, jaringan, dan evaluator.
mod integrity;
mod restore;
mod save;
mod schema;

pub use crate::ReasoningCheckpoint;
pub(crate) use integrity::atomic_write;
pub use schema::{
    CheckpointFingerprints, RestoredIntegratedState, RestoredKnowledgeState, RestoredState,
};

/// Operasi simpan/pulihkan checkpoint versi 1.
pub struct Checkpoint;

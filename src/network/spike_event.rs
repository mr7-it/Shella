use serde::{Deserialize, Serialize};

/// Input sinaptik yang dijadwalkan untuk langkah jaringan tertentu.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(super) struct DelayedInput {
    pub(super) due_step: u64,
    pub(super) target: usize,
    pub(super) current_na: f64,
}

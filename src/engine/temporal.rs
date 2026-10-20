use super::MemoryAttention;
use crate::{MemoryRecall, SimError};

pub(super) fn calculate_attention(
    recalled: &[MemoryRecall],
    temporal_decay: f64,
) -> Result<Vec<MemoryAttention>, SimError> {
    if recalled.is_empty() {
        return Ok(Vec::new());
    }
    let now = crate::learning::unix_timestamp_nanos()?;
    let mut raw_weights = Vec::new();
    raw_weights
        .try_reserve_exact(recalled.len())
        .map_err(|_| SimError::AllocationFailed)?;
    for recall in recalled {
        let age = now.saturating_sub(recall.created_at_unix_nanos()) as f64
            / crate::learning::nanos_per_second();
        let recency = (-temporal_decay * age).exp();
        let raw = recall.retrieval_score() * recency;
        if !raw.is_finite() {
            return Err(SimError::NumericalOverflow {
                operation: "perhatian temporal memori",
            });
        }
        raw_weights.push(raw);
    }
    let total = raw_weights.iter().sum::<f64>();
    if !total.is_finite() {
        return Err(SimError::NumericalOverflow {
            operation: "normalisasi perhatian temporal",
        });
    }
    let mut attention = Vec::new();
    attention
        .try_reserve_exact(recalled.len())
        .map_err(|_| SimError::AllocationFailed)?;
    for (recall, raw) in recalled.iter().zip(raw_weights) {
        attention.push(MemoryAttention {
            record_id: recall.record_id().to_owned(),
            similarity: recall.similarity(),
            weight: if total > 0.0 { raw / total } else { 0.0 },
        });
    }
    Ok(attention)
}

use super::ReasoningConfig;
use crate::reasoning::EvidenceContradiction;
use crate::{MemoryRecall, SimError};

pub(super) fn detect_contradictions(
    recalled: &[MemoryRecall],
) -> Result<Vec<EvidenceContradiction>, SimError> {
    let pair_count = recalled
        .len()
        .checked_mul(recalled.len().saturating_sub(1))
        .and_then(|count| count.checked_div(2))
        .ok_or(SimError::AllocationFailed)?;
    let mut contradictions = Vec::new();
    contradictions
        .try_reserve(pair_count)
        .map_err(|_| SimError::AllocationFailed)?;
    for (index, left) in recalled.iter().enumerate() {
        let Some(left_action) = left.expected_action() else {
            continue;
        };
        for right in recalled.iter().skip(index + 1) {
            let Some(right_action) = right.expected_action() else {
                continue;
            };
            if left_action == right_action {
                continue;
            }
            let similarity = cosine_features(left.features(), right.features());
            if similarity >= 0.45 {
                contradictions.push(EvidenceContradiction {
                    left_record_id: left.record_id().to_owned(),
                    right_record_id: right.record_id().to_owned(),
                    left_action,
                    right_action,
                    similarity,
                });
            }
        }
    }
    Ok(contradictions)
}

fn cosine_features(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() {
        return 0.0;
    }
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    (dot / (left_norm * right_norm)).clamp(0.0, 1.0)
}

pub(super) fn select_compute_budget(
    config: ReasoningConfig,
    input: &[f64],
    recalled: &[MemoryRecall],
    contradictions: &[EvidenceContradiction],
) -> usize {
    if !config.adaptive_compute || config.iterations <= 1 {
        return config.iterations;
    }
    let input_complexity =
        input.iter().filter(|value| **value != 0.0).count() as f64 / input.len().max(1) as f64;
    let mut action_counts = std::collections::BTreeMap::<usize, usize>::new();
    for action in recalled.iter().filter_map(MemoryRecall::expected_action) {
        *action_counts.entry(action).or_default() += 1;
    }
    let has_recall = !recalled.is_empty();
    let recall_complexity = if action_counts.len() <= 1 || !has_recall {
        None
    } else {
        let total = recalled.len() as f64;
        let entropy = action_counts
            .values()
            .map(|count| {
                let probability = *count as f64 / total;
                -probability * probability.ln()
            })
            .sum::<f64>();
        Some((entropy / (action_counts.len() as f64).ln()).clamp(0.0, 1.0))
    };
    let contradiction_complexity = (recalled.len() >= 2).then(|| {
        (contradictions.len() as f64 / (recalled.len() * (recalled.len() - 1) / 2) as f64)
            .clamp(0.0, 1.0)
    });
    let mut complexity_sum = input_complexity;
    let mut complexity_sources = 1.0;
    if let Some(recall_complexity) = recall_complexity {
        complexity_sum += recall_complexity;
        complexity_sources += 1.0;
    }
    if let Some(contradiction_complexity) = contradiction_complexity {
        complexity_sum += contradiction_complexity;
        complexity_sources += 1.0;
    }
    let complexity = complexity_sum / complexity_sources;
    (1 + (complexity * (config.iterations - 1) as f64).ceil() as usize).min(config.iterations)
}

pub(super) fn normalize(values: &mut [f64]) {
    let total = values.iter().sum::<f64>();
    if total > 0.0 && total.is_finite() {
        for value in values {
            *value /= total;
        }
    }
}

pub(super) fn support_metrics(support: &[f64]) -> (f64, f64) {
    let total = support.iter().sum::<f64>();
    if total <= 0.0 || !total.is_finite() {
        return (1.0, 0.0);
    }
    let consistency = support.iter().copied().fold(0.0_f64, f64::max) / total;
    if support.len() <= 1 {
        return (0.0, consistency);
    }
    let entropy = support
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| {
            let probability = value / total;
            -probability * probability.ln()
        })
        .sum::<f64>();
    (entropy / (support.len() as f64).ln(), consistency)
}

#[cfg(test)]
mod tests {
    use super::support_metrics;

    #[test]
    fn confidence_signals_follow_measured_action_distribution() {
        let (balanced_uncertainty, balanced_consistency) = support_metrics(&[0.5, 0.5]);
        assert!((balanced_uncertainty - 1.0).abs() < 1e-12);
        assert!((balanced_consistency - 0.5).abs() < 1e-12);

        let (certain_uncertainty, certain_consistency) = support_metrics(&[1.0, 0.0]);
        assert_eq!(certain_uncertainty, 0.0);
        assert_eq!(certain_consistency, 1.0);

        let (empty_uncertainty, empty_consistency) = support_metrics(&[0.0, 0.0]);
        assert_eq!(empty_uncertainty, 1.0);
        assert_eq!(empty_consistency, 0.0);
    }
}

use super::{AbstentionReason, Decision, policy::validate_id};
use crate::SimError;

/// Akumulator confusion metrics untuk keputusan dengan kelas atau target diam.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecisionMetrics {
    samples: u64,
    correct: u64,
    true_positives: u64,
    false_positives: u64,
    false_negatives: u64,
    abstentions: u64,
    correct_abstentions: u64,
    false_abstentions: u64,
    positive_samples: u64,
    negative_samples: u64,
    false_positive_on_negative: u64,
    no_candidates: u64,
    below_minimum_score: u64,
    below_minimum_margin: u64,
    no_action_signal: u64,
}

impl DecisionMetrics {
    #[must_use]
    pub const fn samples(self) -> u64 {
        self.samples
    }
    #[must_use]
    pub const fn correct(self) -> u64 {
        self.correct
    }
    #[must_use]
    pub const fn true_positives(self) -> u64 {
        self.true_positives
    }
    #[must_use]
    pub const fn false_positives(self) -> u64 {
        self.false_positives
    }
    #[must_use]
    pub const fn false_negatives(self) -> u64 {
        self.false_negatives
    }
    #[must_use]
    pub const fn abstentions(self) -> u64 {
        self.abstentions
    }
    #[must_use]
    pub const fn correct_abstentions(self) -> u64 {
        self.correct_abstentions
    }
    #[must_use]
    pub const fn false_abstentions(self) -> u64 {
        self.false_abstentions
    }
    #[must_use]
    pub const fn positive_samples(self) -> u64 {
        self.positive_samples
    }
    #[must_use]
    pub const fn negative_samples(self) -> u64 {
        self.negative_samples
    }
    #[must_use]
    pub const fn false_positive_on_negative(self) -> u64 {
        self.false_positive_on_negative
    }
    #[must_use]
    pub const fn no_candidates(self) -> u64 {
        self.no_candidates
    }
    #[must_use]
    pub const fn below_minimum_score(self) -> u64 {
        self.below_minimum_score
    }
    #[must_use]
    pub const fn below_minimum_margin(self) -> u64 {
        self.below_minimum_margin
    }
    #[must_use]
    pub const fn no_action_signal(self) -> u64 {
        self.no_action_signal
    }
    #[must_use]
    pub fn accuracy(self) -> Option<f64> {
        rate(self.correct, self.samples)
    }
    #[must_use]
    pub fn precision(self) -> Option<f64> {
        rate(
            self.true_positives,
            self.true_positives + self.false_positives,
        )
    }
    #[must_use]
    pub fn recall(self) -> Option<f64> {
        rate(
            self.true_positives,
            self.true_positives + self.false_negatives,
        )
    }
    #[must_use]
    pub fn false_positive_rate(self) -> Option<f64> {
        rate(self.false_positive_on_negative, self.negative_samples)
    }
    #[must_use]
    pub fn false_negative_rate(self) -> Option<f64> {
        rate(self.false_negatives, self.positive_samples)
    }
    #[must_use]
    pub fn abstention_rate(self) -> Option<f64> {
        rate(self.abstentions, self.samples)
    }
}

/// Mengevaluasi prediksi terhadap label aktual; `None` berarti tidak ada aksi yang benar.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecisionEvaluator {
    metrics: DecisionMetrics,
}

impl DecisionEvaluator {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            metrics: DecisionMetrics {
                samples: 0,
                correct: 0,
                true_positives: 0,
                false_positives: 0,
                false_negatives: 0,
                abstentions: 0,
                correct_abstentions: 0,
                false_abstentions: 0,
                positive_samples: 0,
                negative_samples: 0,
                false_positive_on_negative: 0,
                no_candidates: 0,
                below_minimum_score: 0,
                below_minimum_margin: 0,
                no_action_signal: 0,
            },
        }
    }

    #[must_use]
    pub const fn metrics(&self) -> DecisionMetrics {
        self.metrics
    }

    /// Mencatat keputusan. Salah kelas menambah FP dan FN; abstain pada target aksi
    /// dihitung sebagai FN, sedangkan abstain pada target diam adalah benar.
    pub fn record(
        &mut self,
        expected: Option<&str>,
        predicted: Option<&str>,
        abstention_reason: Option<AbstentionReason>,
    ) -> Result<(), SimError> {
        if let Some(expected) = expected {
            validate_id(expected, "expected_decision")?;
        }
        if let Some(predicted) = predicted {
            validate_id(predicted, "predicted_decision")?;
        }
        if predicted.is_some() && abstention_reason.is_some() {
            return Err(SimError::InvalidInput {
                parameter: "abstention_reason",
                reason: "hanya valid ketika keputusan abstain",
            });
        }

        let matches = expected == predicted;
        let positive_match = expected.is_some() && matches;
        let false_positive = predicted.is_some() && !matches;
        let false_negative = expected.is_some() && !matches;
        let abstained = predicted.is_none();
        let correct_abstention = expected.is_none() && abstained;
        let false_abstention = expected.is_some() && abstained;
        let false_positive_on_negative = expected.is_none() && predicted.is_some();
        let mut next = self.metrics;
        increment(&mut next.samples)?;
        if matches {
            increment(&mut next.correct)?;
        }
        if positive_match {
            increment(&mut next.true_positives)?;
        }
        if false_positive {
            increment(&mut next.false_positives)?;
        }
        if false_negative {
            increment(&mut next.false_negatives)?;
        }
        if abstained {
            increment(&mut next.abstentions)?;
        }
        if correct_abstention {
            increment(&mut next.correct_abstentions)?;
        }
        if false_abstention {
            increment(&mut next.false_abstentions)?;
        }
        if expected.is_some() {
            increment(&mut next.positive_samples)?;
        }
        if expected.is_none() {
            increment(&mut next.negative_samples)?;
        }
        if false_positive_on_negative {
            increment(&mut next.false_positive_on_negative)?;
        }
        if abstained {
            match abstention_reason {
                Some(AbstentionReason::NoCandidates) => increment(&mut next.no_candidates)?,
                Some(AbstentionReason::BelowMinimumScore) => {
                    increment(&mut next.below_minimum_score)?
                }
                Some(AbstentionReason::BelowMinimumMargin) => {
                    increment(&mut next.below_minimum_margin)?
                }
                Some(AbstentionReason::NoActionSignal) => increment(&mut next.no_action_signal)?,
                None => {}
            }
        }
        self.metrics = next;
        Ok(())
    }

    pub fn evaluate_decision(
        &mut self,
        expected: Option<&str>,
        decision: &Decision,
    ) -> Result<(), SimError> {
        self.record(
            expected,
            decision.selected_id(),
            decision.abstention_reason(),
        )
    }
}

fn increment(value: &mut u64) -> Result<(), SimError> {
    *value = value.checked_add(1).ok_or(SimError::CounterOverflow {
        counter: "metrik keputusan",
    })?;
    Ok(())
}

fn rate(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::DecisionMetrics;

    #[test]
    fn precision_recall_tidak_terdefinisi_tanpa_positif() {
        let metrics = DecisionMetrics::default();
        assert_eq!(metrics.precision(), None);
        assert_eq!(metrics.recall(), None);
    }
}

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

/// Akumulasi kalibrasi untuk confidence pada rentang probabilitas `[0, 1]`.
/// Skor linear dan margin keputusan tidak diterima karena keduanya bukan probabilitas.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CalibrationMetrics {
    samples: u64,
    brier_sum: f64,
    absolute_error_sum: f64,
}

impl CalibrationMetrics {
    #[must_use]
    pub const fn samples(self) -> u64 {
        self.samples
    }

    /// Brier score rata-rata; makin kecil makin baik.
    #[must_use]
    pub fn brier_score(self) -> Option<f64> {
        rate_f64(self.brier_sum, self.samples)
    }

    /// Galat absolut rata-rata antara confidence dan outcome biner.
    #[must_use]
    pub fn mean_absolute_calibration_error(self) -> Option<f64> {
        rate_f64(self.absolute_error_sum, self.samples)
    }
}

/// Evaluator terpisah untuk confidence probabilistik yang sudah dikalibrasi.
/// Library tidak mengubah score linear menjadi probabilitas secara diam-diam.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CalibrationEvaluator {
    metrics: CalibrationMetrics,
}

impl CalibrationEvaluator {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            metrics: CalibrationMetrics {
                samples: 0,
                brier_sum: 0.0,
                absolute_error_sum: 0.0,
            },
        }
    }

    /// Mencatat confidence dan hasil biner aktual.
    pub fn record(&mut self, confidence: f64, correct: bool) -> Result<(), SimError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(SimError::InvalidInput {
                parameter: "calibrated_confidence",
                reason: "harus finite dan berada pada rentang 0..1",
            });
        }
        let error = confidence - f64::from(correct);
        let mut next = self.metrics;
        next.samples = next
            .samples
            .checked_add(1)
            .ok_or(SimError::CounterOverflow {
                counter: "sampel kalibrasi",
            })?;
        next.brier_sum += error * error;
        next.absolute_error_sum += error.abs();
        if !next.brier_sum.is_finite() || !next.absolute_error_sum.is_finite() {
            return Err(SimError::NumericalOverflow {
                operation: "akumulasi kalibrasi",
            });
        }
        self.metrics = next;
        Ok(())
    }

    #[must_use]
    pub const fn metrics(&self) -> CalibrationMetrics {
        self.metrics
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

fn rate_f64(numerator: f64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| numerator / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::{CalibrationEvaluator, DecisionMetrics};

    #[test]
    fn precision_recall_tidak_terdefinisi_tanpa_positif() {
        let metrics = DecisionMetrics::default();
        assert_eq!(metrics.precision(), None);
        assert_eq!(metrics.recall(), None);
    }

    #[test]
    fn kalibrasi_hanya_menerima_confidence_probabilistik() {
        let mut evaluator = CalibrationEvaluator::new();
        assert!(evaluator.record(1.2, true).is_err());
        evaluator.record(0.9, true).expect("confidence valid");
        evaluator.record(0.2, false).expect("confidence valid");
        let metrics = evaluator.metrics();
        assert_eq!(metrics.samples(), 2);
        assert!((metrics.brier_score().expect("brier") - 0.025).abs() < 1e-12);
        assert!((metrics.mean_absolute_calibration_error().expect("mae") - 0.15).abs() < 1e-12);
    }
}

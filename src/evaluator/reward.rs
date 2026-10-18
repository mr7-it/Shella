use super::{
    config::RewardEvaluatorConfig,
    metrics::{RewardEvaluation, RewardMetrics},
};
use crate::{LifNetwork, SimError};
use serde::{Deserialize, Serialize};

/// Evaluator deterministik untuk aksi yang dikodekan sebagai spike output.
///
/// `observed_actions[i] == true` berarti kanal aksi `i` spike. Jika beberapa
/// kanal aktif, kanal aktif berindeks terendah menjadi prediksi secara eksplisit.
/// Target `None` merepresentasikan respons diam yang memang diharapkan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardEvaluator {
    config: RewardEvaluatorConfig,
    metrics: RewardMetrics,
}

impl RewardEvaluator {
    /// Membuat evaluator dengan reward dan penalti positif yang tervalidasi.
    pub fn new(config: RewardEvaluatorConfig) -> Result<Self, SimError> {
        config.validate()?;
        Ok(Self {
            config,
            metrics: RewardMetrics::default(),
        })
    }

    /// Menghasilkan reward dari target dan kanal spike yang benar-benar teramati.
    ///
    /// Metrik hanya berubah jika target, input, dan akumulasi hasil semuanya valid.
    pub fn evaluate(
        &mut self,
        expected_action: Option<usize>,
        observed_actions: &[bool],
    ) -> Result<RewardEvaluation, SimError> {
        let (evaluation, next_metrics) =
            self.prepare_evaluation(expected_action, observed_actions)?;
        self.metrics = next_metrics;
        Ok(evaluation)
    }

    /// Mengevaluasi respons, menerapkan reward ke eligibility jaringan, lalu mencatat metrik.
    ///
    /// Jika validasi atau `LifNetwork::apply_reward` gagal, metrik evaluator tidak berubah.
    pub fn evaluate_and_apply(
        &mut self,
        expected_action: Option<usize>,
        observed_actions: &[bool],
        network: &mut LifNetwork,
    ) -> Result<RewardEvaluation, SimError> {
        let (evaluation, next_metrics) =
            self.prepare_evaluation(expected_action, observed_actions)?;
        network.apply_reward(evaluation.reward)?;
        self.metrics = next_metrics;
        Ok(evaluation)
    }

    /// Snapshot metrik yang telah terkumpul.
    #[must_use]
    pub const fn metrics(&self) -> RewardMetrics {
        self.metrics
    }

    /// Konfigurasi reward evaluator.
    #[must_use]
    pub const fn config(&self) -> RewardEvaluatorConfig {
        self.config
    }

    pub(crate) fn validate_restored_state(&self) -> Result<(), SimError> {
        self.config.validate()?;
        if self.metrics.correct > self.metrics.samples
            || self.metrics.no_response > self.metrics.samples
            || !self.metrics.total_reward.is_finite()
        {
            return Err(SimError::InvalidCheckpoint {
                reason: "metrik evaluator tidak konsisten".to_owned(),
            });
        }
        Ok(())
    }

    fn prepare_evaluation(
        &self,
        expected_action: Option<usize>,
        observed_actions: &[bool],
    ) -> Result<(RewardEvaluation, RewardMetrics), SimError> {
        if observed_actions.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "observed_actions",
                reason: "harus memiliki sekurang-kurangnya satu kanal aksi",
            });
        }
        if expected_action.is_some_and(|action| action >= observed_actions.len()) {
            return Err(SimError::InvalidInput {
                parameter: "expected_action",
                reason: "indeks target berada di luar jumlah kanal aksi",
            });
        }

        let predicted_action = observed_actions.iter().position(|active| *active);
        let correct = predicted_action == expected_action;
        let reward = if correct {
            self.config.correct_reward
        } else {
            -self.config.incorrect_penalty
        };
        let evaluation = RewardEvaluation {
            expected_action,
            predicted_action,
            correct,
            reward,
        };
        let samples = self
            .metrics
            .samples
            .checked_add(1)
            .ok_or(SimError::CounterOverflow {
                counter: "sampel reward evaluator",
            })?;
        let correct_count = self.metrics.correct.checked_add(u64::from(correct)).ok_or(
            SimError::CounterOverflow {
                counter: "respons benar evaluator",
            },
        )?;
        let no_response = self
            .metrics
            .no_response
            .checked_add(u64::from(predicted_action.is_none()))
            .ok_or(SimError::CounterOverflow {
                counter: "respons diam evaluator",
            })?;
        let total_reward = self.metrics.total_reward + reward;
        if !total_reward.is_finite() {
            return Err(SimError::NumericalOverflow {
                operation: "akumulasi reward evaluator",
            });
        }
        Ok((
            evaluation,
            RewardMetrics {
                samples,
                correct: correct_count,
                no_response,
                total_reward,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{RewardEvaluator, RewardEvaluatorConfig};
    use crate::SimError;

    #[test]
    fn penghitung_sampel_overflow_tidak_mengubah_metrik() {
        let mut evaluator =
            RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 1.0)).expect("konfigurasi sah");
        evaluator.metrics.samples = u64::MAX;
        let before = evaluator.metrics;

        assert!(matches!(
            evaluator.evaluate(Some(0), &[true]),
            Err(SimError::CounterOverflow { .. })
        ));
        assert_eq!(evaluator.metrics, before);
    }

    #[test]
    fn overflow_total_reward_tidak_mencatat_sampel_kedua() {
        let mut evaluator = RewardEvaluator::new(RewardEvaluatorConfig::new(f64::MAX, 1.0))
            .expect("reward hingga sah");
        evaluator
            .evaluate(Some(0), &[true])
            .expect("reward maksimum finite");
        let before = evaluator.metrics;

        assert!(matches!(
            evaluator.evaluate(Some(0), &[true]),
            Err(SimError::NumericalOverflow { .. })
        ));
        assert_eq!(evaluator.metrics, before);
    }
}

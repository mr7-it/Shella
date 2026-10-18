use crate::SimError;
use serde::{Deserialize, Serialize};

/// Besaran reward untuk respons benar dan penalti untuk respons salah.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RewardEvaluatorConfig {
    pub(super) correct_reward: f64,
    pub(super) incorrect_penalty: f64,
}

impl RewardEvaluatorConfig {
    /// Membuat konfigurasi; nilai diperiksa saat evaluator dibuat.
    #[must_use]
    pub const fn new(correct_reward: f64, incorrect_penalty: f64) -> Self {
        Self {
            correct_reward,
            incorrect_penalty,
        }
    }

    /// Besaran reward non-negatif untuk respons yang cocok dengan target.
    #[must_use]
    pub const fn correct_reward(self) -> f64 {
        self.correct_reward
    }

    /// Besaran penalti positif untuk respons yang tidak cocok dengan target.
    #[must_use]
    pub const fn incorrect_penalty(self) -> f64 {
        self.incorrect_penalty
    }

    pub(super) fn validate(self) -> Result<(), SimError> {
        if !self.correct_reward.is_finite() || self.correct_reward <= 0.0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "correct_reward",
                reason: "harus finite dan lebih besar dari nol",
            });
        }
        if !self.incorrect_penalty.is_finite() || self.incorrect_penalty <= 0.0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "incorrect_penalty",
                reason: "harus finite dan lebih besar dari nol",
            });
        }
        Ok(())
    }
}

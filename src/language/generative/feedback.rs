use serde::{Deserialize, Serialize};

use super::{GenerativeError, Result};

/// Keputusan yang dipelajari head LIF dari teks dan evidence yang diberikan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridAction {
    Answer,
    Revise,
    Abstain,
}

impl HybridAction {
    pub(crate) const fn index(self) -> u32 {
        match self {
            Self::Answer => 0,
            Self::Revise => 1,
            Self::Abstain => 2,
        }
    }

    pub(crate) fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Answer),
            1 => Some(Self::Revise),
            2 => Some(Self::Abstain),
            _ => None,
        }
    }
}

/// Authority attached to feedback; model-generated unverified answers are not an authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackAuthority {
    HumanConfirmed,
    TrustedEvaluator,
}

/// One explicitly reviewed interaction used for a single hybrid training objective.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedActionExample {
    pub source_id: String,
    pub prompt: String,
    pub candidate_response: Option<String>,
    pub target_response: Option<String>,
    pub expected_action: HybridAction,
    pub authority: FeedbackAuthority,
}

impl VerifiedActionExample {
    pub fn validate(&self) -> Result<()> {
        if self.source_id.trim().is_empty()
            || self.source_id.len() > 256
            || self.source_id.chars().any(char::is_control)
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "feedback_source_id",
                reason: "harus berupa ID sumber nonkosong, tanpa karakter kontrol, maksimal 256 byte",
            });
        }
        if self.prompt.trim().is_empty() || self.prompt.len() > 1_000_000 {
            return Err(GenerativeError::InvalidInput {
                parameter: "feedback_prompt",
                reason: "harus nonkosong dan tidak melebihi 1 MiB",
            });
        }
        for response in [&self.candidate_response, &self.target_response]
            .into_iter()
            .flatten()
        {
            if response.trim().is_empty() || response.len() > 1_000_000 {
                return Err(GenerativeError::InvalidInput {
                    parameter: "feedback_response",
                    reason: "respons opsional harus nonkosong dan tidak melebihi 1 MiB",
                });
            }
        }
        match self.expected_action {
            HybridAction::Answer | HybridAction::Revise if self.target_response.is_none() => {
                return Err(GenerativeError::InvalidInput {
                    parameter: "target_response",
                    reason: "aksi answer/revise harus menyertakan jawaban acuan yang ditinjau",
                });
            }
            HybridAction::Abstain if self.target_response.is_some() => {
                return Err(GenerativeError::InvalidInput {
                    parameter: "target_response",
                    reason: "aksi abstain tidak boleh melatih target jawaban",
                });
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridTrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub language_loss_weight: f64,
    pub action_loss_weight: f64,
    pub gradient_clip_norm: f64,
    pub seed: u64,
}

impl Default for HybridTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 4,
            learning_rate: 0.0001,
            language_loss_weight: 0.1,
            action_loss_weight: 1.0,
            gradient_clip_norm: 1.0,
            seed: 42,
        }
    }
}

impl HybridTrainingConfig {
    pub(crate) fn validate(&self, example_count: usize) -> Result<()> {
        if example_count == 0 {
            return Err(GenerativeError::InvalidInput {
                parameter: "verified_feedback",
                reason: "sedikitnya satu contoh feedback terverifikasi diperlukan",
            });
        }
        if self.epochs == 0 || self.epochs > 100 {
            return Err(GenerativeError::InvalidInput {
                parameter: "feedback_epochs",
                reason: "harus berada pada rentang 1..=100",
            });
        }
        if self.batch_size == 0 || self.batch_size > 64 {
            return Err(GenerativeError::InvalidInput {
                parameter: "feedback_batch_size",
                reason: "harus berada pada rentang 1..=64",
            });
        }
        if !self.learning_rate.is_finite() || !(1e-7..=0.01).contains(&self.learning_rate) {
            return Err(GenerativeError::InvalidInput {
                parameter: "feedback_learning_rate",
                reason: "harus finite dan berada pada rentang 1e-7..=0.01",
            });
        }
        if !self.language_loss_weight.is_finite()
            || !(0.0..=10.0).contains(&self.language_loss_weight)
            || !self.action_loss_weight.is_finite()
            || !(0.0..=10.0).contains(&self.action_loss_weight)
            || self.language_loss_weight + self.action_loss_weight == 0.0
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "hybrid_loss_weights",
                reason: "bobot loss harus finite, nonnegatif, dan tidak semuanya nol",
            });
        }
        if !self.gradient_clip_norm.is_finite()
            || !(1e-6..=100.0).contains(&self.gradient_clip_norm)
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "feedback_gradient_clip_norm",
                reason: "harus finite dan berada pada rentang 1e-6..=100",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridTrainingReport {
    pub epochs: usize,
    pub examples: usize,
    pub optimizer_steps: usize,
    pub mean_language_loss: f64,
    pub mean_action_loss: f64,
    pub training_action_accuracy: f64,
    pub feedback_sources: Vec<String>,
    pub authority_counts: FeedbackAuthorityCounts,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridLearningOutcome {
    pub training: HybridTrainingReport,
    pub learning_memory_record_ids: Vec<String>,
    pub checkpoint_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedbackAuthorityCounts {
    pub human_confirmed: usize,
    pub trusted_evaluator: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridActionPrediction {
    pub action: HybridAction,
    /// Raw logits; they are not calibrated probabilities.
    pub logits: [f32; 3],
    pub probabilities: [f32; 3],
}

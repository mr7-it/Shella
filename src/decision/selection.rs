use super::policy::{Candidate, DecisionPolicy, validate_id};
use crate::SimError;
use std::collections::BTreeSet;

impl DecisionPolicy {
    /// Skor setiap kandidat, memilih skor tertinggi; seri diputus oleh ID leksikografis.
    /// Kandidat abstain ketika skor terbaik atau margin tidak melewati batas kebijakan.
    pub fn select(&self, candidates: &[Candidate]) -> Result<Decision, SimError> {
        if candidates.is_empty() {
            return Ok(Decision {
                selected_id: None,
                best_score: None,
                runner_up_score: None,
                margin: None,
                contributions: Vec::new(),
                abstention_reason: Some(AbstentionReason::NoCandidates),
                reason: "tidak ada kandidat".to_owned(),
            });
        }
        let mut seen = BTreeSet::new();
        let mut scored = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            validate_id(&candidate.id, "candidate_id")?;
            if !seen.insert(candidate.id.as_str()) {
                return Err(SimError::InvalidInput {
                    parameter: "candidates",
                    reason: "ID kandidat harus unik",
                });
            }
            let mut score = 0.0;
            let mut contributions = Vec::with_capacity(candidate.features.len());
            for (feature, value) in &candidate.features {
                if !value.is_finite() {
                    return Err(SimError::InvalidInput {
                        parameter: "candidate_feature",
                        reason: "semua sinyal harus finite",
                    });
                }
                let weight = self.weights.get(feature).ok_or(SimError::InvalidInput {
                    parameter: "candidate_feature",
                    reason: "fitur tidak dikenal oleh kebijakan",
                })?;
                let contribution = weight * value;
                score += contribution;
                if !contribution.is_finite() || !score.is_finite() {
                    return Err(SimError::NumericalOverflow {
                        operation: "skor keputusan linear",
                    });
                }
                contributions.push(FeatureContribution {
                    feature: feature.clone(),
                    value: *value,
                    weight: *weight,
                    contribution,
                });
            }
            scored.push((candidate.id.clone(), score, contributions));
        }
        scored.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let (best_id, best_score, contributions) = scored[0].clone();
        let runner_up_score = scored.get(1).map(|entry| entry.1);
        let margin = if let Some(score) = runner_up_score {
            let margin = best_score - score;
            if !margin.is_finite() {
                return Err(SimError::NumericalOverflow {
                    operation: "margin keputusan",
                });
            }
            margin
        } else {
            f64::INFINITY
        };
        let (selected_id, reason, abstention_reason) = if best_score < self.minimum_score {
            (
                None,
                "skor kandidat terbaik di bawah ambang".to_owned(),
                Some(AbstentionReason::BelowMinimumScore),
            )
        } else if margin < self.minimum_margin {
            (
                None,
                "margin kandidat di bawah ambang abstain".to_owned(),
                Some(AbstentionReason::BelowMinimumMargin),
            )
        } else {
            (
                Some(best_id),
                "kandidat dengan skor linear tertinggi; seri diputus berdasarkan ID".to_owned(),
                None,
            )
        };
        Ok(Decision {
            selected_id,
            best_score: Some(best_score),
            runner_up_score,
            margin: Some(margin),
            contributions,
            abstention_reason,
            reason,
        })
    }
}

/// Alasan terstruktur mengapa kebijakan tidak memilih kandidat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstentionReason {
    NoCandidates,
    BelowMinimumScore,
    BelowMinimumMargin,
    NoActionSignal,
}

/// Kontribusi satu fitur ke skor kandidat terpilih.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureContribution {
    pub feature: String,
    pub value: f64,
    pub weight: f64,
    pub contribution: f64,
}

/// Hasil keputusan yang dapat dijelaskan; score dan margin bukan probabilitas.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    selected_id: Option<String>,
    best_score: Option<f64>,
    runner_up_score: Option<f64>,
    margin: Option<f64>,
    contributions: Vec<FeatureContribution>,
    abstention_reason: Option<AbstentionReason>,
    reason: String,
}
impl Decision {
    /// Kandidat terpilih atau `None` jika abstain.
    #[must_use]
    pub fn selected_id(&self) -> Option<&str> {
        self.selected_id.as_deref()
    }
    /// Skor linear kandidat terbaik, bukan probabilitas.
    #[must_use]
    pub const fn best_score(&self) -> Option<f64> {
        self.best_score
    }
    /// Skor runner-up bila ada.
    #[must_use]
    pub const fn runner_up_score(&self) -> Option<f64> {
        self.runner_up_score
    }
    /// Selisih skor terbaik-runner up; infinity jika hanya satu kandidat.
    #[must_use]
    pub const fn score_margin(&self) -> Option<f64> {
        self.margin
    }
    /// Alasan dan rincian fitur kandidat terbaik.
    #[must_use]
    pub fn contributions(&self) -> &[FeatureContribution] {
        &self.contributions
    }
    /// Kategori abstain, atau `None` bila kandidat dipilih.
    #[must_use]
    pub const fn abstention_reason(&self) -> Option<AbstentionReason> {
        self.abstention_reason
    }
    /// Penjelasan berdasarkan aturan aktual yang dieksekusi.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

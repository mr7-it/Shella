use crate::SimError;
use std::collections::BTreeMap;

/// Kandidat tindakan dan sinyal terukur yang menjadi masukan kebijakan.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub(super) id: String,
    pub(super) features: BTreeMap<String, f64>,
}

impl Candidate {
    /// Membuat kandidat; validasi dilakukan oleh `DecisionPolicy::select`.
    pub fn new(id: impl Into<String>, features: BTreeMap<String, f64>) -> Self {
        Self {
            id: id.into(),
            features,
        }
    }
    /// ID kandidat stabil untuk pemilihan dan audit.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Fitur masukan kandidat.
    #[must_use]
    pub const fn features(&self) -> &BTreeMap<String, f64> {
        &self.features
    }
}

/// Kebijakan skor linear dan aturan abstain.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionPolicy {
    pub(super) weights: BTreeMap<String, f64>,
    pub(super) minimum_score: f64,
    pub(super) minimum_margin: f64,
}

impl DecisionPolicy {
    /// Bobot tiap sinyal, batas skor minimum, dan margin minimum di atas runner-up.
    pub fn new(
        weights: BTreeMap<String, f64>,
        minimum_score: f64,
        minimum_margin: f64,
    ) -> Result<Self, SimError> {
        if weights.is_empty() || weights.values().any(|value| !value.is_finite()) {
            return Err(SimError::InvalidConfiguration {
                parameter: "decision_weights",
                reason: "harus memuat bobot finite",
            });
        }
        if !minimum_score.is_finite() || !minimum_margin.is_finite() || minimum_margin < 0.0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "decision_threshold",
                reason: "score dan margin harus finite; margin tidak negatif",
            });
        }
        for key in weights.keys() {
            validate_id(key, "decision_feature")?;
        }
        Ok(Self {
            weights,
            minimum_score,
            minimum_margin,
        })
    }

    /// Bobot sinyal yang digunakan kebijakan.
    #[must_use]
    pub const fn weights(&self) -> &BTreeMap<String, f64> {
        &self.weights
    }
}

pub(super) fn validate_id(value: &str, parameter: &'static str) -> Result<(), SimError> {
    if value.trim().is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(SimError::InvalidInput {
            parameter,
            reason: "harus non-kosong, tanpa karakter kontrol, dan <= 256 byte",
        });
    }
    Ok(())
}

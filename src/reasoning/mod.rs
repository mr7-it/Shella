//! State serializable dan tipe data khusus orkestrasi reasoning.
mod checkpoint;
mod context;

pub use checkpoint::{CalibrationState, ReasoningCheckpoint};
pub use context::{
    CausalAssociation, ContextFrame, EvidenceContradiction, UncertaintyDecomposition,
};

/// Sampel confidence ber-ID untuk mencegah overlap antara calibration fit/evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSample {
    sample_id: String,
    score: f64,
    correct: bool,
}

impl CalibrationSample {
    #[must_use]
    pub fn new(sample_id: impl Into<String>, score: f64, correct: bool) -> Self {
        Self {
            sample_id: sample_id.into(),
            score,
            correct,
        }
    }

    #[must_use]
    pub fn sample_id(&self) -> &str {
        &self.sample_id
    }

    #[must_use]
    pub const fn score(&self) -> f64 {
        self.score
    }

    #[must_use]
    pub const fn correct(&self) -> bool {
        self.correct
    }
}

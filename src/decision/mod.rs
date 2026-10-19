//! Pemilihan kandidat berbasis fitur terstruktur dengan skor yang dapat diaudit.
mod calibration;
mod evaluation;
mod policy;
mod selection;

pub use calibration::{IsotonicCalibrator, PlattCalibrator};
pub use evaluation::{
    CalibrationEvaluator, CalibrationMetrics, DecisionEvaluator, DecisionMetrics,
};
pub use policy::{Candidate, DecisionPolicy};
pub use selection::{AbstentionReason, Decision, FeatureContribution};

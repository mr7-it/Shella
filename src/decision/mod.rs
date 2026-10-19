//! Pemilihan kandidat berbasis fitur terstruktur dengan skor yang dapat diaudit.
mod evaluation;
mod policy;
mod selection;

pub use evaluation::{DecisionEvaluator, DecisionMetrics};
pub use policy::{Candidate, DecisionPolicy};
pub use selection::{AbstentionReason, Decision, FeatureContribution};

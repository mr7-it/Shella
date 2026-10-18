//! Pemilihan kandidat berbasis fitur terstruktur dengan skor yang dapat diaudit.
mod policy;
mod selection;

pub use policy::{Candidate, DecisionPolicy};
pub use selection::{Decision, FeatureContribution};

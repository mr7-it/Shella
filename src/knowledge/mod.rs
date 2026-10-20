//! Versioned Iris class knowledge built from training data.
mod class_knowledge;
mod feature_schema;
mod prototype;
mod statistics;

pub use class_knowledge::{
    CalibrationParameters, ClassKnowledge, ClassKnowledgeInference, ClassKnowledgeModel,
    ClassPrediction, Exemplar, ExemplarPolicy, ExemplarReason, FeatureMetadata,
    LegacyPrototypeSnapshot, MODEL_SCHEMA_VERSION, ModelApprovalStatus, PrototypeCandidateScore,
    SCORING_VERSION,
};
pub use feature_schema::{
    FEATURE_COUNT, FEATURE_SCHEMA_VERSION, FeatureDefinition, FeatureScaler, FeatureType,
    IrisFeatures, feature_definitions,
};
pub use prototype::Prototype;
pub use statistics::FeatureStatistics;

//! Shella: jaringan LIF dan SDK Rust modular untuk aplikasi neuromorfik.
//!
//! Crate menyediakan runtime/neuron LIF, jaringan sinkron, STDP reward-modulated,
//! evaluator dan benchmark, checkpoint tervalidasi, memori file-backed, decision
//! scoring deterministik, preprocessing teks dasar, serta SDK input/output.
//! Pipeline teks dan decision engine transparan tetapi tidak mengklaim pemahaman
//! bahasa atau confidence probabilistik.

pub mod benchmark;
pub mod checkpoint;
pub mod config;
pub mod decision;
pub mod error;
pub mod evaluator;
pub mod knowledge;
pub mod language;
pub mod learning;
pub mod memory;
pub mod network;
pub mod neuron;
pub mod runtime;
pub mod sdk;

pub use benchmark::{
    ConditioningBenchmarkConfig, ConditioningBenchmarkResult, ContinualBenchmarkConfig,
    ContinualBenchmarkResult, ContinualReplayBenchmarkConfig, ContinualReplayBenchmarkResult,
    ContinualReplayNoiseResult, ContinualReplayRunResult, ContinualReplayTaskMetrics,
    ContinualReplayValidation, ContinualSeedResult, GeneralizationBenchmarkConfig,
    GeneralizationBenchmarkResult, GeneralizationNoiseResult, GeneralizationRunResult,
    InterferenceBenchmarkConfig, InterferenceBenchmarkResult, InterferenceSeedResult, IrisSample,
    LongTermLearningBenchmarkConfig, LongTermLearningBenchmarkResult, LongTermLearningNoiseResult,
    LongTermLearningRunResult, LongTermLearningTaskMetrics, SelfEvaluation, StratifiedFold,
    StratifiedSplit, TrainScaler, evaluate_predictions, noise, stratified_folds, stratified_split,
};
pub use checkpoint::{Checkpoint, RestoredKnowledgeState, RestoredState};
pub use config::LifConfig;
pub use decision::{
    AbstentionReason, CalibrationEvaluator, CalibrationMetrics, Candidate, Decision,
    DecisionEvaluator, DecisionMetrics, DecisionPolicy, FeatureContribution, IsotonicCalibrator,
    PlattCalibrator,
};
pub use error::SimError;
pub use evaluator::{RewardEvaluation, RewardEvaluator, RewardEvaluatorConfig, RewardMetrics};
pub use knowledge::{
    CalibrationParameters, ClassKnowledge, ClassKnowledgeInference, ClassKnowledgeModel,
    ClassPrediction, Exemplar, ExemplarPolicy, ExemplarReason, FeatureDefinition, FeatureMetadata,
    FeatureScaler, FeatureStatistics, FeatureType, IrisFeatures, LegacyPrototypeSnapshot,
    ModelApprovalStatus, Prototype, PrototypeCandidateScore, SCORING_VERSION, feature_definitions,
};
pub use language::{EncodedText, TextEncoder, Vocabulary, normalize_text, tokenize};
pub use learning::{
    ForgettingReport, LearningMemory, MemoryConflict, MemoryFeedbackWrite, MemoryInference,
    MemoryRecall, ReplayReport, RewardModulatedStdpConfig, StdpConfig,
};
pub use memory::{MemoryRecord, MemoryStore, RecordId, SecureMemoryStore, TextEmbedding};
pub use network::{LifNetwork, NetworkStep, Synapse};
pub use neuron::lif::{LifNeuron, StepOutcome};
pub use runtime::{LifRuntime, SimulationClock};
pub use sdk::{
    Engine, Input as SdkInput, InputKind as SdkInputKind, Output as SdkOutput,
    Session as SdkSession,
};

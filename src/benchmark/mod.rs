//! Benchmark perilaku jaringan dan statistik evaluasi.
mod continual;
mod continual_replay;
mod experiments;
mod generalization;
mod interference;
mod long_term;
mod self_learning;
mod statistics;

pub use continual::{ContinualBenchmarkConfig, ContinualBenchmarkResult, ContinualSeedResult};
pub use continual_replay::{
    ContinualReplayBenchmarkConfig, ContinualReplayBenchmarkResult, ContinualReplayNoiseResult,
    ContinualReplayRunResult, ContinualReplayTaskMetrics, ContinualReplayValidation,
};
pub use experiments::{ConditioningBenchmarkConfig, ConditioningBenchmarkResult};
pub use generalization::{
    GeneralizationBenchmarkConfig, GeneralizationBenchmarkResult, GeneralizationNoiseResult,
    GeneralizationRunResult,
};
pub use interference::{
    InterferenceBenchmarkConfig, InterferenceBenchmarkResult, InterferenceSeedResult,
};
pub use long_term::{
    LongTermLearningBenchmarkConfig, LongTermLearningBenchmarkResult, LongTermLearningNoiseResult,
    LongTermLearningRunResult, LongTermLearningTaskMetrics,
};
pub use self_learning::{
    IrisSample, SelfEvaluation, StratifiedFold, StratifiedSplit, TrainScaler, evaluate_predictions,
    noise, stratified_folds, stratified_split,
};

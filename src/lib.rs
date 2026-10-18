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
pub mod language;
pub mod learning;
pub mod memory;
pub mod network;
pub mod neuron;
pub mod runtime;
pub mod sdk;

pub use benchmark::{
    ConditioningBenchmarkConfig, ConditioningBenchmarkResult, ContinualBenchmarkConfig,
    ContinualBenchmarkResult, ContinualSeedResult,
};
pub use checkpoint::{Checkpoint, RestoredState};
pub use config::LifConfig;
pub use decision::{Candidate, Decision, DecisionPolicy, FeatureContribution};
pub use error::SimError;
pub use evaluator::{RewardEvaluation, RewardEvaluator, RewardEvaluatorConfig, RewardMetrics};
pub use language::{EncodedText, TextEncoder, Vocabulary, normalize_text, tokenize};
pub use learning::{RewardModulatedStdpConfig, StdpConfig};
pub use memory::{MemoryRecord, MemoryStore, RecordId};
pub use network::{LifNetwork, NetworkStep, Synapse};
pub use neuron::lif::{LifNeuron, StepOutcome};
pub use runtime::{LifRuntime, SimulationClock};
pub use sdk::{
    Engine, Input as SdkInput, InputKind as SdkInputKind, Output as SdkOutput,
    Session as SdkSession,
};

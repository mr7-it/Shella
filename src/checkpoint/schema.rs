use crate::{
    ClassKnowledgeModel, IndonesianVocabulary, LanguageNetwork, LifNetwork, LifRuntime,
    ReasoningCheckpoint, RewardEvaluator,
};
use serde::{Deserialize, Serialize};

pub(super) const FORMAT_VERSION: u32 = 1;
pub(super) const MAX_CHECKPOINT_BYTES: u64 = 256 * 1024 * 1024;

/// State jaringan yang dipulihkan dari checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct RestoredState {
    /// Jaringan termasuk neuron, sinaps, trace STDP/eligibility, clock, dan event tertunda.
    pub network: LifNetwork,
    /// Evaluator dan metrik kumulatif bila disertakan.
    pub evaluator: Option<RewardEvaluator>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestoredKnowledgeState {
    pub network: LifNetwork,
    pub class_knowledge: ClassKnowledgeModel,
    pub action_neurons: Vec<usize>,
    pub evaluator: Option<RewardEvaluator>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "state")]
pub(super) enum Payload {
    Network {
        network: LifNetwork,
        evaluator: Option<RewardEvaluator>,
    },
    SingleNeuronRuntime {
        runtime: LifRuntime,
    },
    KnowledgeNetwork {
        network: LifNetwork,
        class_knowledge: Box<ClassKnowledgeModel>,
        action_neurons: Vec<usize>,
        evaluator: Option<RewardEvaluator>,
    },
    Reasoning {
        state: ReasoningCheckpoint,
    },
    LanguageNetwork {
        state: Box<LanguageNetwork>,
    },
    IndonesianVocabulary {
        state: IndonesianVocabulary,
    },
}

#[derive(Serialize, Deserialize)]
pub(super) struct Envelope {
    pub(super) format_version: u32,
    pub(super) checksum_fnv64: String,
    pub(super) payload_json: String,
}

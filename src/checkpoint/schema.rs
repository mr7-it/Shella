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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointFingerprints {
    pub memory_store: Option<String>,
    pub learning_memory: Option<String>,
    pub semantic_memory: Option<String>,
}

impl CheckpointFingerprints {
    pub(crate) fn validate(&self) -> Result<(), crate::SimError> {
        let valid = self
            .memory_store
            .as_ref()
            .is_none_or(|value| valid_hex(value, 64))
            && self
                .learning_memory
                .as_ref()
                .is_none_or(|value| valid_hex(value, 16))
            && self
                .semantic_memory
                .as_ref()
                .is_none_or(|value| valid_hex(value, 64));
        if !valid {
            return Err(crate::SimError::InvalidCheckpoint {
                reason: "fingerprint checkpoint engine invalid".to_owned(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestoredIntegratedState {
    pub network: LifNetwork,
    pub evaluator: Option<RewardEvaluator>,
    pub fingerprints: CheckpointFingerprints,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "state")]
pub(super) enum Payload {
    Network {
        network: LifNetwork,
        evaluator: Option<RewardEvaluator>,
    },
    IntegratedEngine {
        network: LifNetwork,
        evaluator: Option<RewardEvaluator>,
        fingerprints: CheckpointFingerprints,
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

fn valid_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

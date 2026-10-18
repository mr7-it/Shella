use crate::{LifNetwork, LifRuntime, RewardEvaluator};
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
}

#[derive(Serialize, Deserialize)]
pub(super) struct Envelope {
    pub(super) format_version: u32,
    pub(super) checksum_fnv64: String,
    pub(super) payload_json: String,
}

use super::{Checkpoint, integrity, schema::Payload};
use crate::{ClassKnowledgeModel, LifNetwork, LifRuntime, RewardEvaluator, SimError};
use std::path::Path;

impl Checkpoint {
    pub fn save_knowledge(
        path: impl AsRef<Path>,
        network: &LifNetwork,
        class_knowledge: &ClassKnowledgeModel,
        action_neurons: &[usize],
        evaluator: Option<&RewardEvaluator>,
    ) -> Result<(), SimError> {
        network.validate_restored_state()?;
        class_knowledge.validate()?;
        validate_action_mapping(network, action_neurons)?;
        if let Some(evaluator) = evaluator {
            evaluator.validate_restored_state()?;
        }
        let payload_json = serde_json::to_string(&Payload::KnowledgeNetwork {
            network: network.clone(),
            class_knowledge: class_knowledge.clone(),
            action_neurons: action_neurons.to_vec(),
            evaluator: evaluator.cloned(),
        })
        .map_err(|error| SimError::SerializationFailed {
            subsystem: "knowledge checkpoint",
            reason: error.to_string(),
        })?;
        integrity::write_payload(path.as_ref(), payload_json)
    }

    /// Menulis snapshot konsisten jaringan dan evaluator opsional secara atomik.
    pub fn save(
        path: impl AsRef<Path>,
        network: &LifNetwork,
        evaluator: Option<&RewardEvaluator>,
    ) -> Result<(), SimError> {
        network.validate_restored_state()?;
        if let Some(evaluator) = evaluator {
            evaluator.validate_restored_state()?;
        }
        let payload_json = serde_json::to_string(&Payload::Network {
            network: network.clone(),
            evaluator: evaluator.cloned(),
        })
        .map_err(|error| SimError::SerializationFailed {
            subsystem: "checkpoint",
            reason: error.to_string(),
        })?;
        integrity::write_payload(path.as_ref(), payload_json)
    }

    /// Menyimpan seluruh state dari runtime satu-neuron.
    pub fn save_runtime(path: impl AsRef<Path>, runtime: &LifRuntime) -> Result<(), SimError> {
        runtime.validate_restored_state()?;
        let payload_json = serde_json::to_string(&Payload::SingleNeuronRuntime {
            runtime: runtime.clone(),
        })
        .map_err(|error| SimError::SerializationFailed {
            subsystem: "checkpoint",
            reason: error.to_string(),
        })?;
        integrity::write_payload(path.as_ref(), payload_json)
    }
}

fn validate_action_mapping(network: &LifNetwork, action_neurons: &[usize]) -> Result<(), SimError> {
    let mut unique = std::collections::BTreeSet::new();
    if action_neurons.len() != 3
        || action_neurons
            .iter()
            .any(|neuron| *neuron >= network.neuron_count() || !unique.insert(*neuron))
    {
        return Err(SimError::InvalidInput {
            parameter: "action_neurons",
            reason: "checkpoint ClassKnowledge memerlukan tiga indeks neuron valid dan unik",
        });
    }
    Ok(())
}

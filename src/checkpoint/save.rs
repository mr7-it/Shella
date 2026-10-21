use super::{Checkpoint, integrity, schema::Payload};
use crate::{ClassKnowledgeModel, LifNetwork, LifRuntime, RewardEvaluator, SimError};
use std::path::Path;

impl Checkpoint {
    /// Menyimpan neural state dengan identitas store eksternal yang harus cocok saat restore.
    pub fn save_integrated_engine(
        path: impl AsRef<Path>,
        network: &LifNetwork,
        evaluator: Option<&RewardEvaluator>,
        fingerprints: &super::schema::CheckpointFingerprints,
    ) -> Result<(), SimError> {
        fingerprints.validate()?;
        let mut network = network.clone();
        network.migrate_restored_identity_state()?;
        if let Some(evaluator) = evaluator {
            evaluator.validate_restored_state()?;
        }
        let payload_json = serde_json::to_string(&Payload::IntegratedEngine {
            network,
            evaluator: evaluator.cloned(),
            fingerprints: fingerprints.clone(),
        })
        .map_err(|error| SimError::SerializationFailed {
            subsystem: "integrated engine checkpoint",
            reason: error.to_string(),
        })?;
        integrity::write_payload(path.as_ref(), payload_json)
    }

    /// Menyimpan lexicon Indonesia tervalidasi melalui envelope checksummed atomik.
    pub fn save_indonesian_vocabulary(
        path: impl AsRef<Path>,
        state: &crate::IndonesianVocabulary,
    ) -> Result<(), SimError> {
        state.validate()?;
        let payload_json = serde_json::to_string(&Payload::IndonesianVocabulary {
            state: state.clone(),
        })
        .map_err(|error| SimError::SerializationFailed {
            subsystem: "Indonesian lexicon",
            reason: error.to_string(),
        })?;
        integrity::write_payload(path.as_ref(), payload_json)
    }

    /// Menyimpan vocabulary, ID, relasi neuron, bobot sinaps, dan state LIF atomik.
    pub fn save_language_network(
        path: impl AsRef<Path>,
        state: &crate::LanguageNetwork,
    ) -> Result<(), SimError> {
        state.validate()?;
        let payload_json = serde_json::to_string(&Payload::LanguageNetwork {
            state: Box::new(state.clone()),
        })
        .map_err(|error| SimError::SerializationFailed {
            subsystem: "language network",
            reason: error.to_string(),
        })?;
        integrity::write_payload(path.as_ref(), payload_json)
    }

    /// Menyimpan snapshot reasoning berversi dengan checksum dan atomic replace.
    pub fn save_reasoning(
        path: impl AsRef<Path>,
        state: &crate::ReasoningCheckpoint,
    ) -> Result<(), SimError> {
        state.validate()?;
        let payload_json = serde_json::to_string(&Payload::Reasoning {
            state: state.clone(),
        })
        .map_err(|error| SimError::SerializationFailed {
            subsystem: "reasoning checkpoint",
            reason: error.to_string(),
        })?;
        integrity::write_payload(path.as_ref(), payload_json)
    }

    pub fn save_knowledge(
        path: impl AsRef<Path>,
        network: &LifNetwork,
        class_knowledge: &ClassKnowledgeModel,
        action_neurons: &[usize],
        evaluator: Option<&RewardEvaluator>,
    ) -> Result<(), SimError> {
        let mut network = network.clone();
        network.migrate_restored_identity_state()?;
        class_knowledge.validate()?;
        validate_action_mapping(&network, action_neurons)?;
        if let Some(evaluator) = evaluator {
            evaluator.validate_restored_state()?;
        }
        let payload_json = serde_json::to_string(&Payload::KnowledgeNetwork {
            network,
            class_knowledge: Box::new(class_knowledge.clone()),
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
        let mut network = network.clone();
        network.migrate_restored_identity_state()?;
        if let Some(evaluator) = evaluator {
            evaluator.validate_restored_state()?;
        }
        let payload_json = serde_json::to_string(&Payload::Network {
            network,
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

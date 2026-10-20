use super::{Checkpoint, RestoredKnowledgeState, RestoredState, integrity, schema::Payload};
use crate::{LifRuntime, SimError};
use std::path::Path;

impl Checkpoint {
    pub fn restore_knowledge(path: impl AsRef<Path>) -> Result<RestoredKnowledgeState, SimError> {
        let payload = integrity::read_payload(path.as_ref())?;
        let Payload::KnowledgeNetwork {
            network,
            class_knowledge,
            action_neurons,
            evaluator,
        } = payload
        else {
            return Err(integrity::invalid_checkpoint(
                "checkpoint tidak berisi state ClassKnowledge",
            ));
        };
        network.validate_restored_state()?;
        class_knowledge.validate()?;
        validate_action_mapping(&network, &action_neurons)?;
        if let Some(evaluator) = &evaluator {
            evaluator.validate_restored_state()?;
        }
        Ok(RestoredKnowledgeState {
            network,
            class_knowledge,
            action_neurons,
            evaluator,
        })
    }

    /// Membaca, memeriksa versi/checksum, dan memvalidasi seluruh state jaringan.
    pub fn restore(path: impl AsRef<Path>) -> Result<RestoredState, SimError> {
        let payload = integrity::read_payload(path.as_ref())?;
        let Payload::Network { network, evaluator } = payload else {
            return Err(integrity::invalid_checkpoint(
                "checkpoint tidak berisi state jaringan",
            ));
        };
        network.validate_restored_state()?;
        if let Some(evaluator) = &evaluator {
            evaluator.validate_restored_state()?;
        }
        Ok(RestoredState { network, evaluator })
    }

    /// Memulihkan runtime satu-neuron dari checkpoint skema yang sama.
    pub fn restore_runtime(path: impl AsRef<Path>) -> Result<LifRuntime, SimError> {
        let payload = integrity::read_payload(path.as_ref())?;
        let Payload::SingleNeuronRuntime { runtime } = payload else {
            return Err(integrity::invalid_checkpoint(
                "checkpoint tidak berisi runtime satu-neuron",
            ));
        };
        runtime.validate_restored_state()?;
        Ok(runtime)
    }
}

fn validate_action_mapping(
    network: &crate::LifNetwork,
    action_neurons: &[usize],
) -> Result<(), SimError> {
    let mut unique = std::collections::BTreeSet::new();
    if action_neurons.len() != 3
        || action_neurons
            .iter()
            .any(|neuron| *neuron >= network.neuron_count() || !unique.insert(*neuron))
    {
        return Err(integrity::invalid_checkpoint(
            "mapping action checkpoint ClassKnowledge tidak valid",
        ));
    }
    Ok(())
}

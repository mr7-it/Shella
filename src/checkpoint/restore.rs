use super::{
    Checkpoint, RestoredIntegratedState, RestoredKnowledgeState, RestoredState, integrity,
    schema::Payload,
};
use crate::{LifRuntime, SimError};
use std::path::Path;

impl Checkpoint {
    /// Memulihkan neural state dan fingerprint store yang perlu dipadankan pemanggil.
    pub fn restore_integrated_engine(
        path: impl AsRef<Path>,
    ) -> Result<RestoredIntegratedState, SimError> {
        let payload = integrity::read_payload(path.as_ref())?;
        let Payload::IntegratedEngine {
            mut network,
            evaluator,
            fingerprints,
        } = payload
        else {
            return Err(integrity::invalid_checkpoint(
                "checkpoint tidak berisi integrated Engine state",
            ));
        };
        fingerprints.validate()?;
        network.migrate_restored_identity_state()?;
        if let Some(evaluator) = &evaluator {
            evaluator.validate_restored_state()?;
        }
        Ok(RestoredIntegratedState {
            network,
            evaluator,
            fingerprints,
        })
    }

    /// Memulihkan lexicon Indonesia setelah validasi format dan invariants.
    pub fn restore_indonesian_vocabulary(
        path: impl AsRef<Path>,
    ) -> Result<crate::IndonesianVocabulary, SimError> {
        let payload = integrity::read_payload(path.as_ref())?;
        let Payload::IndonesianVocabulary { state } = payload else {
            return Err(integrity::invalid_checkpoint(
                "checkpoint tidak berisi Indonesian vocabulary",
            ));
        };
        state.validate()?;
        Ok(state)
    }

    /// Memulihkan seluruh state language network setelah pemeriksaan checksum/invariants.
    pub fn restore_language_network(
        path: impl AsRef<Path>,
    ) -> Result<crate::LanguageNetwork, SimError> {
        let payload = integrity::read_payload(path.as_ref())?;
        let Payload::LanguageNetwork { state } = payload else {
            return Err(integrity::invalid_checkpoint(
                "checkpoint tidak berisi language network",
            ));
        };
        state.validate()?;
        Ok(*state)
    }

    /// Memulihkan reasoning state setelah pemeriksaan checksum dan invariants.
    pub fn restore_reasoning(
        path: impl AsRef<Path>,
    ) -> Result<crate::ReasoningCheckpoint, SimError> {
        let payload = integrity::read_payload(path.as_ref())?;
        let Payload::Reasoning { mut state } = payload else {
            return Err(integrity::invalid_checkpoint(
                "checkpoint tidak berisi reasoning state",
            ));
        };
        state.migrate_restored_identity_state()?;
        state.validate()?;
        Ok(state)
    }

    pub fn restore_knowledge(path: impl AsRef<Path>) -> Result<RestoredKnowledgeState, SimError> {
        let payload = integrity::read_payload(path.as_ref())?;
        let Payload::KnowledgeNetwork {
            mut network,
            class_knowledge,
            action_neurons,
            evaluator,
        } = payload
        else {
            return Err(integrity::invalid_checkpoint(
                "checkpoint tidak berisi state ClassKnowledge",
            ));
        };
        network.migrate_restored_identity_state()?;
        class_knowledge.validate()?;
        validate_action_mapping(&network, &action_neurons)?;
        if let Some(evaluator) = &evaluator {
            evaluator.validate_restored_state()?;
        }
        Ok(RestoredKnowledgeState {
            network,
            class_knowledge: *class_knowledge,
            action_neurons,
            evaluator,
        })
    }

    /// Membaca, memeriksa versi/checksum, dan memvalidasi seluruh state jaringan.
    pub fn restore(path: impl AsRef<Path>) -> Result<RestoredState, SimError> {
        let payload = integrity::read_payload(path.as_ref())?;
        let Payload::Network {
            mut network,
            evaluator,
        } = payload
        else {
            return Err(integrity::invalid_checkpoint(
                "checkpoint tidak berisi state jaringan",
            ));
        };
        network.migrate_restored_identity_state()?;
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

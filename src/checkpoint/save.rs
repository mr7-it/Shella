use super::{Checkpoint, integrity, schema::Payload};
use crate::{LifNetwork, LifRuntime, RewardEvaluator, SimError};
use std::path::Path;

impl Checkpoint {
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

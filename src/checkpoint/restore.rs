use super::{Checkpoint, RestoredState, integrity, schema::Payload};
use crate::{LifRuntime, SimError};
use std::path::Path;

impl Checkpoint {
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

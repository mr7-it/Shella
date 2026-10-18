use crate::{SimError, neuron::lif::LifNeuron};

/// Memastikan neuron yang direstorasi membentuk satu populasi dengan clock sinkron.
pub(super) fn validate_population(
    neurons: &[LifNeuron],
    completed_steps: u64,
) -> Result<(), SimError> {
    if neurons.is_empty() {
        return Err(SimError::InvalidCheckpoint {
            reason: "jaringan tidak memiliki neuron".to_owned(),
        });
    }
    let shared_dt = neurons[0].config().dt_ms();
    for neuron in neurons {
        neuron.validate_restored_state()?;
        if neuron.completed_steps() != completed_steps || neuron.config().dt_ms() != shared_dt {
            return Err(SimError::InvalidCheckpoint {
                reason: "jam atau interval waktu neuron tidak konsisten".to_owned(),
            });
        }
    }
    Ok(())
}

use crate::reasoning::CalibrationSample;
use crate::{LifNetwork, SimError};
use std::collections::BTreeSet;

pub(super) fn validate_calibration_samples(
    samples: &[(f64, bool)],
    require_both_outcomes: bool,
) -> Result<(), SimError> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|(score, _)| !score.is_finite() || !(0.0..=1.0).contains(score))
        || (require_both_outcomes
            && (!samples.iter().any(|(_, correct)| *correct)
                || !samples.iter().any(|(_, correct)| !*correct)))
    {
        return Err(SimError::InvalidInput {
            parameter: "confidence_validation_samples",
            reason: "nilai harus berada pada 0..=1; fit perlu outcome benar dan salah",
        });
    }
    Ok(())
}

pub(super) fn validate_sample_ids(
    validation: &[CalibrationSample],
    evaluation: &[CalibrationSample],
) -> Result<(), SimError> {
    let mut validation_ids = BTreeSet::new();
    if validation.iter().any(|sample| {
        sample.sample_id().trim().is_empty() || !validation_ids.insert(sample.sample_id())
    }) {
        return Err(SimError::InvalidInput {
            parameter: "validation_sample_ids",
            reason: "ID harus non-kosong dan unik dalam validation split",
        });
    }
    let mut evaluation_ids = BTreeSet::new();
    if evaluation.iter().any(|sample| {
        sample.sample_id().trim().is_empty()
            || !evaluation_ids.insert(sample.sample_id())
            || validation_ids.contains(sample.sample_id())
    }) {
        return Err(SimError::InvalidInput {
            parameter: "evaluation_sample_ids",
            reason: "ID harus non-kosong, unik, dan tidak overlap validation split",
        });
    }
    Ok(())
}

pub(super) fn validate_reasoning_input(
    network: &LifNetwork,
    currents: &[f64],
    action_neurons: &[usize],
) -> Result<(), SimError> {
    if currents.len() != network.neuron_count() || currents.iter().any(|value| !value.is_finite()) {
        return Err(SimError::InvalidInput {
            parameter: "input_currents_na",
            reason: "panjang harus sama dengan neuron dan seluruh nilai harus finite",
        });
    }
    if action_neurons.is_empty()
        || action_neurons
            .iter()
            .any(|index| *index >= network.neuron_count())
        || action_neurons
            .iter()
            .enumerate()
            .any(|(index, value)| action_neurons[..index].contains(value))
    {
        return Err(SimError::InvalidInput {
            parameter: "action_neurons",
            reason: "harus memuat indeks valid dan unik",
        });
    }
    Ok(())
}

pub(super) fn validate_action_label(
    action: Option<usize>,
    action_count: usize,
    parameter: &'static str,
) -> Result<(), SimError> {
    if action.is_some_and(|index| index >= action_count) {
        return Err(SimError::InvalidInput {
            parameter,
            reason: "aksi berada di luar mapping action neuron",
        });
    }
    Ok(())
}

pub(super) fn validate_readout_update(
    network: &LifNetwork,
    features: &[f64],
    target_action: usize,
    action_neurons: &[usize],
    learning_rate: f64,
) -> Result<(), SimError> {
    let target_neuron = action_neurons[target_action];
    for synapse in network.synapses() {
        if synapse.source() >= features.len() || !action_neurons.contains(&synapse.target()) {
            continue;
        }
        let direction = if synapse.target() == target_neuron {
            1.0
        } else {
            -1.0
        };
        let delta = direction * learning_rate * features[synapse.source()];
        let candidate = synapse.weight_na() + delta;
        if !delta.is_finite() || !candidate.is_finite() {
            return Err(SimError::NumericalOverflow {
                operation: "validasi pembaruan readout pengalaman",
            });
        }
    }
    Ok(())
}

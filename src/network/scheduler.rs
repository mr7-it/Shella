use super::{DelayedInput, Synapse};
use crate::{SimError, neuron::lif::LifStepPlan};
use std::convert::TryFrom;

/// Menyiapkan antrean event berikutnya tanpa mengubah antrean aktif.
pub(super) fn schedule_next(
    pending: &[DelayedInput],
    synapses: &[Synapse],
    plans: &[LifStepPlan],
    next_step: u64,
) -> Result<Vec<DelayedInput>, SimError> {
    let future_count = pending
        .iter()
        .filter(|event| event.due_step > next_step)
        .count();
    let capacity = future_count
        .checked_add(synapses.len())
        .ok_or(SimError::AllocationFailed)?;
    let mut scheduled = Vec::new();
    scheduled
        .try_reserve_exact(capacity)
        .map_err(|_| SimError::AllocationFailed)?;
    for event in pending {
        if event.due_step > next_step {
            scheduled.push(*event);
        }
    }
    for synapse in synapses {
        if plans
            .get(synapse.source)
            .ok_or(SimError::InvalidTopology {
                reason: "sumber sinaps di luar jumlah neuron",
            })?
            .outcome
            .spiked
        {
            let delay = u64::try_from(synapse.delay_steps).map_err(|_| SimError::InvalidInput {
                parameter: "delay_steps",
                reason: "tidak dapat direpresentasikan sebagai penghitung langkah",
            })?;
            let due_step = next_step
                .checked_add(delay)
                .ok_or(SimError::StepCounterOverflow)?;
            scheduled.push(DelayedInput {
                due_step,
                target: synapse.target,
                current_na: synapse.weight_na,
            });
        }
    }
    Ok(scheduled)
}

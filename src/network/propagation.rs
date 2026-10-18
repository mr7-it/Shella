use super::DelayedInput;
use crate::SimError;

/// Menjumlahkan arus input dan seluruh event yang tiba pada langkah berikutnya.
pub(super) fn aggregate_currents(
    external: &[f64],
    pending: &[DelayedInput],
    next_step: u64,
) -> Result<Vec<f64>, SimError> {
    let mut total = Vec::new();
    total
        .try_reserve_exact(external.len())
        .map_err(|_| SimError::AllocationFailed)?;
    total.extend_from_slice(external);
    for event in pending {
        if event.due_step < next_step {
            return Err(SimError::InvalidTopology {
                reason: "antrean memuat peristiwa yang sudah terlewat",
            });
        }
        if event.due_step == next_step {
            let target = total
                .get_mut(event.target)
                .ok_or(SimError::InvalidTopology {
                    reason: "target event di luar jumlah neuron",
                })?;
            *target += event.current_na;
            if !target.is_finite() {
                return Err(SimError::NumericalOverflow {
                    operation: "penjumlahan arus sinaptik",
                });
            }
        }
    }
    Ok(total)
}

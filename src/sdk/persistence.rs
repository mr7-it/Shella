use crate::{Checkpoint, LifNetwork, RewardEvaluator, SimError};
use std::path::Path;

pub(super) fn save(
    path: impl AsRef<Path>,
    network: &LifNetwork,
    evaluator: Option<&RewardEvaluator>,
) -> Result<(), SimError> {
    Checkpoint::save(path, network, evaluator)
}

pub(super) fn restore(
    path: impl AsRef<Path>,
) -> Result<(LifNetwork, Option<RewardEvaluator>), SimError> {
    let state = Checkpoint::restore(path)?;
    Ok((state.network, state.evaluator))
}

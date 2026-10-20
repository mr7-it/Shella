use crate::{CalibrationMetrics, LifNetwork, PlattCalibrator, SimError};
use serde::{Deserialize, Serialize};

use super::ContextFrame;

const REASONING_CHECKPOINT_VERSION: u32 = 1;
const CALIBRATION_STATE_VERSION: u32 = 1;

/// Kalibrator Platt beserta provenance data, hiperparameter, dan metrik evaluasi.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationState {
    version: u32,
    iterations: usize,
    learning_rate: f64,
    validation_sample_ids: Vec<String>,
    evaluation_sample_ids: Vec<String>,
    calibrator: PlattCalibrator,
    metrics: CalibrationMetrics,
}

impl CalibrationState {
    pub(crate) fn new(
        iterations: usize,
        learning_rate: f64,
        validation_sample_ids: Vec<String>,
        evaluation_sample_ids: Vec<String>,
        calibrator: PlattCalibrator,
        metrics: CalibrationMetrics,
    ) -> Self {
        Self {
            version: CALIBRATION_STATE_VERSION,
            iterations,
            learning_rate,
            validation_sample_ids,
            evaluation_sample_ids,
            calibrator,
            metrics,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), SimError> {
        if self.version != CALIBRATION_STATE_VERSION
            || self.iterations == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.calibrator.slope().is_finite()
            || !self.calibrator.intercept().is_finite()
            || self.metrics.samples() == 0
            || self
                .metrics
                .brier_score()
                .is_none_or(|score| !score.is_finite() || !(0.0..=1.0).contains(&score))
            || self
                .metrics
                .mean_absolute_calibration_error()
                .is_none_or(|error| !error.is_finite() || !(0.0..=1.0).contains(&error))
            || self.validation_sample_ids.is_empty() != self.evaluation_sample_ids.is_empty()
        {
            return Err(SimError::InvalidCheckpoint {
                reason: "state kalibrasi reasoning tidak konsisten".to_owned(),
            });
        }
        let mut validation_ids = std::collections::BTreeSet::new();
        let mut evaluation_ids = std::collections::BTreeSet::new();
        if self
            .validation_sample_ids
            .iter()
            .any(|id| id.trim().is_empty() || !validation_ids.insert(id))
            || self.evaluation_sample_ids.iter().any(|id| {
                id.trim().is_empty() || !evaluation_ids.insert(id) || validation_ids.contains(id)
            })
        {
            return Err(SimError::InvalidCheckpoint {
                reason: "provenance kalibrasi reasoning memiliki ID invalid/overlap".to_owned(),
            });
        }
        Ok(())
    }

    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    #[must_use]
    pub const fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[must_use]
    pub fn validation_sample_ids(&self) -> &[String] {
        &self.validation_sample_ids
    }

    #[must_use]
    pub fn evaluation_sample_ids(&self) -> &[String] {
        &self.evaluation_sample_ids
    }

    #[must_use]
    pub const fn calibrator(&self) -> PlattCalibrator {
        self.calibrator
    }

    #[must_use]
    pub const fn metrics(&self) -> CalibrationMetrics {
        self.metrics
    }
}

/// Snapshot reasoning berversi. LearningMemory disimpan atomik pada store terpisah.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningCheckpoint {
    pub(crate) version: u32,
    pub(crate) network: LifNetwork,
    pub(crate) iterations: usize,
    pub(crate) recall_limit: usize,
    pub(crate) memory_current_na: f64,
    pub(crate) temporal_decay: f64,
    pub(crate) minimum_confidence: f64,
    pub(crate) context_window_capacity: usize,
    pub(crate) adaptive_compute: bool,
    pub(crate) active_context_id: Option<String>,
    pub(crate) context_frames: Vec<ContextFrame>,
    pub(crate) calibration: Option<CalibrationState>,
    pub(crate) memory_fingerprint: String,
}

impl ReasoningCheckpoint {
    pub(crate) fn migrate_restored_identity_state(&mut self) -> Result<(), SimError> {
        self.network.migrate_restored_identity_state()
    }

    pub(crate) fn validate(&self) -> Result<(), SimError> {
        if self.version != REASONING_CHECKPOINT_VERSION
            || !(1..=1024).contains(&self.iterations)
            || !self.memory_current_na.is_finite()
            || self.memory_current_na < 0.0
            || !self.temporal_decay.is_finite()
            || self.temporal_decay < 0.0
            || !self.minimum_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_confidence)
            || self.context_frames.len() > self.context_window_capacity
            || self.context_window_capacity > 4096
            || self.memory_fingerprint.len() != 16
            || !self
                .memory_fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(SimError::InvalidCheckpoint {
                reason: "konfigurasi reasoning checkpoint invalid".to_owned(),
            });
        }
        self.network.validate_restored_state()?;
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
        }
        for frame in &self.context_frames {
            if frame.context_id.trim().is_empty()
                || self.active_context_id.as_deref() != Some(frame.context_id.as_str())
                || frame.currents_na.len() != self.network.neuron_count()
                || frame.currents_na.iter().any(|value| !value.is_finite())
                || frame
                    .spiking_neurons
                    .iter()
                    .any(|index| *index >= self.network.neuron_count())
                || frame.completed_step > self.network.completed_steps()
            {
                return Err(SimError::InvalidCheckpoint {
                    reason: "frame context reasoning checkpoint invalid".to_owned(),
                });
            }
        }
        Ok(())
    }

    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }
}

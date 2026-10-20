use super::ReasoningEngine;
use super::validation::{validate_calibration_samples, validate_sample_ids};
use crate::reasoning::{CalibrationSample, CalibrationState};
use crate::{CalibrationEvaluator, CalibrationMetrics, PlattCalibrator, SimError};

impl ReasoningEngine {
    /// Fit Platt pada validation split dan mengukur Brier/MAE pada split evaluasi
    /// terpisah. Pemanggil bertanggung jawab agar kedua split tidak overlap dan
    /// keduanya terpisah dari data pelatihan model utama.
    pub fn fit_confidence_calibration(
        &mut self,
        validation_samples: &[(f64, bool)],
        evaluation_samples: &[(f64, bool)],
        iterations: usize,
        learning_rate: f64,
    ) -> Result<CalibrationMetrics, SimError> {
        validate_calibration_samples(validation_samples, true)?;
        validate_calibration_samples(evaluation_samples, false)?;
        let calibrator = PlattCalibrator::fit(validation_samples, iterations, learning_rate)?;
        let mut evaluator = CalibrationEvaluator::new();
        for (score, correct) in evaluation_samples {
            evaluator.record(calibrator.predict(*score), *correct)?;
        }
        let metrics = evaluator.metrics();
        self.calibration = Some(CalibrationState::new(
            iterations,
            learning_rate,
            Vec::new(),
            Vec::new(),
            calibrator,
            metrics,
        ));
        Ok(metrics)
    }

    #[must_use]
    pub const fn calibration_metrics(&self) -> Option<CalibrationMetrics> {
        match &self.calibration {
            Some(state) => Some(state.metrics()),
            None => None,
        }
    }

    /// Fit dan evaluasi kalibrator memakai ID unik untuk memastikan split terpisah.
    pub fn fit_confidence_calibration_with_ids(
        &mut self,
        validation_samples: &[CalibrationSample],
        evaluation_samples: &[CalibrationSample],
        iterations: usize,
        learning_rate: f64,
    ) -> Result<CalibrationMetrics, SimError> {
        validate_sample_ids(validation_samples, evaluation_samples)?;
        let validation = validation_samples
            .iter()
            .map(|sample| (sample.score(), sample.correct()))
            .collect::<Vec<_>>();
        let evaluation = evaluation_samples
            .iter()
            .map(|sample| (sample.score(), sample.correct()))
            .collect::<Vec<_>>();
        validate_calibration_samples(&validation, true)?;
        validate_calibration_samples(&evaluation, false)?;
        let calibrator = PlattCalibrator::fit(&validation, iterations, learning_rate)?;
        let mut evaluator = CalibrationEvaluator::new();
        for (score, correct) in &evaluation {
            evaluator.record(calibrator.predict(*score), *correct)?;
        }
        let metrics = evaluator.metrics();
        self.calibration = Some(CalibrationState::new(
            iterations,
            learning_rate,
            validation_samples
                .iter()
                .map(|sample| sample.sample_id().to_owned())
                .collect(),
            evaluation_samples
                .iter()
                .map(|sample| sample.sample_id().to_owned())
                .collect(),
            calibrator,
            metrics,
        ));
        Ok(metrics)
    }

    #[must_use]
    pub const fn calibration_state(&self) -> Option<&CalibrationState> {
        self.calibration.as_ref()
    }
}

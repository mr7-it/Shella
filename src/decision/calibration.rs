use crate::SimError;
use serde::{Deserialize, Serialize};

/// Kalibrator sigmoid Platt yang dilatih pada split validasi berlabel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlattCalibrator {
    slope: f64,
    intercept: f64,
}

impl PlattCalibrator {
    pub fn fit(
        samples: &[(f64, bool)],
        iterations: usize,
        learning_rate: f64,
    ) -> Result<Self, SimError> {
        validate_samples(samples, iterations, learning_rate)?;
        let mut slope = 0.0;
        let mut intercept = 0.0;
        for _ in 0..iterations {
            let mut grad_slope = 0.0;
            let mut grad_intercept = 0.0;
            for (score, correct) in samples {
                let prediction = sigmoid(slope * *score + intercept);
                let error = prediction - f64::from(*correct);
                grad_slope += error * score;
                grad_intercept += error;
            }
            let scale = 1.0 / samples.len() as f64;
            slope -= learning_rate * grad_slope * scale;
            intercept -= learning_rate * grad_intercept * scale;
            if !slope.is_finite() || !intercept.is_finite() {
                return Err(SimError::NumericalOverflow {
                    operation: "fit Platt scaling",
                });
            }
        }
        Ok(Self { slope, intercept })
    }

    #[must_use]
    pub const fn slope(self) -> f64 {
        self.slope
    }
    #[must_use]
    pub const fn intercept(self) -> f64 {
        self.intercept
    }
    #[must_use]
    pub fn predict(self, score: f64) -> f64 {
        sigmoid(self.slope * score + self.intercept)
    }
}

/// Kalibrator isotonic dengan pool-adjacent-violators (PAVA).
#[derive(Debug, Clone, PartialEq)]
pub struct IsotonicCalibrator {
    blocks: Vec<(f64, f64)>,
}

impl IsotonicCalibrator {
    pub fn fit(samples: &[(f64, bool)]) -> Result<Self, SimError> {
        if samples.len() < 2 || samples.iter().any(|(score, _)| !score.is_finite()) {
            return Err(SimError::InvalidInput {
                parameter: "calibration_samples",
                reason: "minimal dua score finite diperlukan",
            });
        }
        let mut sorted = samples
            .iter()
            .map(|(score, correct)| (*score, f64::from(*correct)))
            .collect::<Vec<_>>();
        sorted.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut blocks: Vec<(f64, f64, usize, f64)> = Vec::new();
        for (score, target) in sorted {
            blocks.push((score, target, 1, target));
            while blocks.len() >= 2 {
                let right = blocks[blocks.len() - 1];
                let left = blocks[blocks.len() - 2];
                if left.3 <= right.3 {
                    break;
                }
                let count = left.2 + right.2;
                let average = (left.3 * left.2 as f64 + right.3 * right.2 as f64) / count as f64;
                blocks.pop();
                blocks.pop();
                blocks.push((left.0, right.0, count, average));
            }
        }
        Ok(Self {
            blocks: blocks
                .into_iter()
                .map(|(low, high, _, value)| ((low + high) / 2.0, value.clamp(0.0, 1.0)))
                .collect(),
        })
    }

    #[must_use]
    pub fn predict(&self, score: f64) -> f64 {
        if !score.is_finite() || self.blocks.is_empty() {
            return 0.5;
        }
        self.blocks
            .iter()
            .min_by(|left, right| (left.0 - score).abs().total_cmp(&(right.0 - score).abs()))
            .map_or(0.5, |(_, value)| *value)
    }

    #[must_use]
    pub fn blocks(&self) -> &[(f64, f64)] {
        &self.blocks
    }
}

fn validate_samples(
    samples: &[(f64, bool)],
    iterations: usize,
    learning_rate: f64,
) -> Result<(), SimError> {
    if samples.len() < 2
        || iterations == 0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || samples.iter().any(|(score, _)| !score.is_finite())
    {
        return Err(SimError::InvalidInput {
            parameter: "calibration_samples",
            reason: "minimal dua score finite, iterasi positif, dan learning rate positif diperlukan",
        });
    }
    Ok(())
}
fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

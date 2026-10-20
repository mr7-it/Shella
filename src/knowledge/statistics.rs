use super::FEATURE_COUNT;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureStatistics {
    pub sample_count: usize,
    pub mean: [f64; FEATURE_COUNT],
    pub variance: [f64; FEATURE_COUNT],
    pub standard_deviation: [f64; FEATURE_COUNT],
    pub minimum: [f64; FEATURE_COUNT],
    pub maximum: [f64; FEATURE_COUNT],
    pub quantiles: [[f64; 3]; FEATURE_COUNT],
    pub covariance: [[f64; FEATURE_COUNT]; FEATURE_COUNT],
    pub outlier_count: usize,
    pub borderline_count: usize,
}

impl FeatureStatistics {
    pub(super) fn compute(rows: &[[f64; FEATURE_COUNT]]) -> Self {
        let count = rows.len();
        let mut mean = [0.0; FEATURE_COUNT];
        let mut minimum = [f64::INFINITY; FEATURE_COUNT];
        let mut maximum = [f64::NEG_INFINITY; FEATURE_COUNT];
        for row in rows {
            for feature in 0..FEATURE_COUNT {
                mean[feature] += row[feature];
                minimum[feature] = minimum[feature].min(row[feature]);
                maximum[feature] = maximum[feature].max(row[feature]);
            }
        }
        if count > 0 {
            for value in &mut mean {
                *value /= count as f64;
            }
        }
        let mut variance = [0.0; FEATURE_COUNT];
        let mut covariance = [[0.0; FEATURE_COUNT]; FEATURE_COUNT];
        if count > 0 {
            for row in rows {
                for left in 0..FEATURE_COUNT {
                    let left_delta = row[left] - mean[left];
                    variance[left] += left_delta * left_delta;
                    for right in 0..FEATURE_COUNT {
                        covariance[left][right] += left_delta * (row[right] - mean[right]);
                    }
                }
            }
            let denominator = count as f64;
            for (feature, covariance_row) in covariance.iter_mut().enumerate() {
                variance[feature] /= denominator;
                for value in covariance_row {
                    *value /= denominator;
                }
            }
        }
        let standard_deviation = variance.map(f64::sqrt);
        let mut quantiles = [[0.0; 3]; FEATURE_COUNT];
        let mut bounds = [[0.0; 3]; FEATURE_COUNT];
        for feature in 0..FEATURE_COUNT {
            let mut values = rows.iter().map(|row| row[feature]).collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            for (slot, probability) in [0.25, 0.5, 0.75].into_iter().enumerate() {
                quantiles[feature][slot] = quantile(&values, probability);
            }
            let spread = quantiles[feature][2] - quantiles[feature][0];
            bounds[feature] = [
                quantiles[feature][0] - 1.5 * spread,
                quantiles[feature][2] + 1.5 * spread,
                spread * 0.05,
            ];
        }
        let outlier_count = rows
            .iter()
            .filter(|row| {
                (0..FEATURE_COUNT).any(|feature| {
                    row[feature] < bounds[feature][0] || row[feature] > bounds[feature][1]
                })
            })
            .count();
        let borderline_count = rows
            .iter()
            .filter(|row| {
                (0..FEATURE_COUNT).any(|feature| {
                    let lower_distance = (row[feature] - bounds[feature][0]).abs();
                    let upper_distance = (row[feature] - bounds[feature][1]).abs();
                    lower_distance <= bounds[feature][2] || upper_distance <= bounds[feature][2]
                })
            })
            .count();
        Self {
            sample_count: count,
            mean,
            variance,
            standard_deviation,
            minimum,
            maximum,
            quantiles,
            covariance,
            outlier_count,
            borderline_count,
        }
    }
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    match sorted.len() {
        0 => 0.0,
        1 => sorted[0],
        length => {
            let position = probability * (length - 1) as f64;
            let lower = position.floor() as usize;
            let upper = position.ceil() as usize;
            sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
        }
    }
}

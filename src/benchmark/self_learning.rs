use crate::{DecisionEvaluator, DecisionMetrics, SimError};

/// Satu baris Iris yang diparsing dari dataset sumber tanpa mengubah nilainya.
#[derive(Debug, Clone, PartialEq)]
pub struct IrisSample {
    features: [f64; 4],
    label: usize,
}

impl IrisSample {
    pub fn new(features: [f64; 4], label: usize) -> Result<Self, SimError> {
        if features.iter().any(|value| !value.is_finite()) || label > 2 {
            return Err(SimError::InvalidInput {
                parameter: "iris_sample",
                reason: "fitur harus finite dan label berada pada 0..2",
            });
        }
        Ok(Self { features, label })
    }
    #[must_use]
    pub const fn features(&self) -> &[f64; 4] {
        &self.features
    }
    #[must_use]
    pub const fn label(&self) -> usize {
        self.label
    }
}

/// Split train/validation/test per kelas secara deterministik.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StratifiedSplit {
    train: Vec<usize>,
    validation: Vec<usize>,
    test: Vec<usize>,
}

pub type StratifiedFold = (Vec<usize>, Vec<usize>);

impl StratifiedSplit {
    #[must_use]
    pub fn train(&self) -> &[usize] {
        &self.train
    }
    #[must_use]
    pub fn validation(&self) -> &[usize] {
        &self.validation
    }
    #[must_use]
    pub fn test(&self) -> &[usize] {
        &self.test
    }
}

/// Membagi indeks non-test ke k fold stratified; fold hanya dipakai untuk validasi
/// hyperparameter, bukan untuk mengukur hasil akhir test.
pub fn stratified_folds(
    samples: &[IrisSample],
    indices: &[usize],
    folds: usize,
) -> Result<Vec<StratifiedFold>, SimError> {
    if folds < 2 || indices.len() < folds {
        return Err(SimError::InvalidInput {
            parameter: "cross_validation_folds",
            reason: "folds minimal dua dan tidak melebihi jumlah indeks",
        });
    }
    let mut per_class = [Vec::new(), Vec::new(), Vec::new()];
    for &index in indices {
        let sample = samples.get(index).ok_or(SimError::InvalidInput {
            parameter: "cross_validation_indices",
            reason: "indeks di luar dataset",
        })?;
        per_class[sample.label()].push(index);
    }
    let mut result = Vec::with_capacity(folds);
    for fold in 0..folds {
        let mut validation = Vec::new();
        for class_indices in &per_class {
            for (position, &index) in class_indices.iter().enumerate() {
                if position % folds == fold {
                    validation.push(index);
                }
            }
        }
        validation.sort_unstable();
        let mut training = indices
            .iter()
            .copied()
            .filter(|index| validation.binary_search(index).is_err())
            .collect::<Vec<_>>();
        training.sort_unstable();
        result.push((training, validation));
    }
    Ok(result)
}

/// Membentuk split 60/20/20 dengan rotasi seed per kelas; indeks test tidak dipakai
/// selama fitting scaler, pembentukan memori, atau tuning parameter.
pub fn stratified_split(samples: &[IrisSample], seed: u64) -> Result<StratifiedSplit, SimError> {
    if samples.len() < 15 {
        return Err(SimError::InvalidInput {
            parameter: "iris_samples",
            reason: "minimal 15 sampel",
        });
    }
    let mut per_class = [Vec::new(), Vec::new(), Vec::new()];
    for (index, sample) in samples.iter().enumerate() {
        per_class[sample.label()].push(index);
    }
    if per_class.iter().any(|indices| indices.len() < 3) {
        return Err(SimError::InvalidInput {
            parameter: "iris_labels",
            reason: "setiap kelas harus memiliki minimal tiga sampel",
        });
    }
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();
    for indices in per_class {
        let offset = (seed as usize) % indices.len();
        for position in 0..indices.len() {
            let index = indices[(position + offset) % indices.len()];
            match position % 5 {
                0 => test.push(index),
                1 => validation.push(index),
                _ => train.push(index),
            }
        }
    }
    train.sort_unstable();
    validation.sort_unstable();
    test.sort_unstable();
    Ok(StratifiedSplit {
        train,
        validation,
        test,
    })
}

/// Scaler fit hanya pada train split.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainScaler {
    min: [f64; 4],
    max: [f64; 4],
}
impl TrainScaler {
    pub fn fit(samples: &[IrisSample], indices: &[usize]) -> Result<Self, SimError> {
        if indices.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "train_indices",
                reason: "tidak boleh kosong",
            });
        }
        let mut min = [f64::INFINITY; 4];
        let mut max = [f64::NEG_INFINITY; 4];
        for &index in indices {
            let sample = samples.get(index).ok_or(SimError::InvalidInput {
                parameter: "train_indices",
                reason: "indeks di luar dataset",
            })?;
            for feature in 0..4 {
                min[feature] = min[feature].min(sample.features()[feature]);
                max[feature] = max[feature].max(sample.features()[feature]);
            }
        }
        if (0..4).any(|feature| min[feature] == max[feature]) {
            return Err(SimError::InvalidInput {
                parameter: "train_scaler",
                reason: "rentang fitur train harus positif",
            });
        }
        Ok(Self { min, max })
    }
    #[must_use]
    pub fn transform(&self, features: &[f64; 4]) -> [f64; 4] {
        std::array::from_fn(|i| (features[i] - self.min[i]) / (self.max[i] - self.min[i]))
    }
}

/// Self-evaluation berisi performa dan sinyal kelemahan yang terukur.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfEvaluation {
    metrics: DecisionMetrics,
    weak_samples: usize,
    uncertainty_samples: usize,
}
impl SelfEvaluation {
    #[must_use]
    pub const fn metrics(&self) -> DecisionMetrics {
        self.metrics
    }
    #[must_use]
    pub const fn weak_samples(&self) -> usize {
        self.weak_samples
    }
    #[must_use]
    pub const fn uncertainty_samples(&self) -> usize {
        self.uncertainty_samples
    }
    #[must_use]
    pub fn error_rate(&self) -> Option<f64> {
        self.metrics.accuracy().map(|accuracy| 1.0 - accuracy)
    }
}

/// Mengubah prediksi menjadi metrik dan menghitung sampel yang layak untuk perbaikan.
pub fn evaluate_predictions(
    records: &[(usize, Option<usize>, f64)],
) -> Result<SelfEvaluation, SimError> {
    let mut evaluator = DecisionEvaluator::new();
    let mut weak_samples = 0;
    let mut uncertainty_samples = 0;
    for (expected, predicted, margin) in records {
        let expected_name = format!("kelas_{expected}");
        let predicted_name = predicted.map(|value| format!("kelas_{value}"));
        let low_margin = *margin < 0.15;
        weak_samples += usize::from(predicted != &Some(*expected));
        uncertainty_samples += usize::from(low_margin);
        evaluator.record(
            Some(&expected_name),
            predicted_name.as_deref(),
            (predicted.is_none() && low_margin).then_some(crate::AbstentionReason::NoActionSignal),
        )?;
    }
    Ok(SelfEvaluation {
        metrics: evaluator.metrics(),
        weak_samples,
        uncertainty_samples,
    })
}

/// Generator noise deterministik, dipakai hanya pada observasi evaluasi.
#[must_use]
pub fn noise(seed: u64, sample: usize, feature: usize, amplitude: f64) -> f64 {
    let mut state = seed
        ^ (sample as u64).wrapping_mul(0x9e37_79b9)
        ^ (feature as u64).wrapping_mul(0xd134_2543);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    ((state >> 11) as f64 / ((1_u64 << 53) as f64) * 2.0 - 1.0) * amplitude
}

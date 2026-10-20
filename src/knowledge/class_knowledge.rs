use super::{
    FEATURE_COUNT, FEATURE_SCHEMA_VERSION, FeatureScaler, FeatureStatistics, IrisFeatures,
    Prototype, feature_definitions,
    prototype::{fit_kmeans, squared_distance},
};
use crate::decision::PlattCalibrator;
use crate::{SimError, benchmark::IrisSample};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MODEL_SCHEMA_VERSION: u32 = 1;
pub const SCORING_VERSION: u32 = 1;
const CANDIDATE_PROTOTYPES: [usize; 6] = [2, 3, 4, 5, 6, 8];
const CLASS_NAMES: [&str; 3] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Exemplar {
    pub exemplar_id: String,
    pub class_id: usize,
    pub raw_features: [f64; 4],
    pub derived_features: [f64; FEATURE_COUNT],
    pub embedding: [f64; FEATURE_COUNT],
    pub expected_label: usize,
    pub predicted_label_at_capture: usize,
    pub calibrated_confidence: Option<f64>,
    pub uncertainty: f64,
    pub boundary_score: f64,
    pub importance: f64,
    pub usage_count: u64,
    pub last_retrieved_at_unix: Option<u64>,
    pub feedback: f64,
    pub source_dataset_version: String,
    pub source_training_run: String,
    pub reason: ExemplarReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExemplarReason {
    Representative,
    ClassBoundary,
    ClusterRepresentative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExemplarPolicy {
    pub max_per_class: usize,
}

impl ExemplarPolicy {
    pub fn new(max_per_class: usize) -> Result<Self, SimError> {
        if max_per_class == 0 || max_per_class > 64 {
            return Err(SimError::InvalidConfiguration {
                parameter: "max_exemplars_per_class",
                reason: "budget exemplar harus berada pada 1..=64",
            });
        }
        Ok(Self { max_per_class })
    }
}

impl Default for ExemplarPolicy {
    fn default() -> Self {
        Self { max_per_class: 2 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassKnowledge {
    pub class_id: usize,
    pub class_name: String,
    pub sample_count: usize,
    pub statistics: FeatureStatistics,
    pub prototypes: Vec<Prototype>,
    pub exemplars: Vec<Exemplar>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassKnowledgeModel {
    pub model_id: String,
    pub schema_version: u32,
    pub feature_schema_version: u32,
    pub scoring_version: u32,
    pub model_version: u32,
    pub parent_model_version: Option<u32>,
    pub approval_status: ModelApprovalStatus,
    pub dataset_version: String,
    pub training_run_id: String,
    pub seed: u64,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
    pub feature_schema: Vec<super::FeatureDefinition>,
    pub feature_metadata: Vec<FeatureMetadata>,
    pub scaler: FeatureScaler,
    pub training_statistics: FeatureStatistics,
    pub selected_prototypes_per_class: usize,
    pub prototype_candidate_scores: Vec<PrototypeCandidateScore>,
    pub exemplar_policy: ExemplarPolicy,
    pub validation_accuracy: f64,
    pub calibration: [CalibrationParameters; 3],
    #[serde(default)]
    pub class_score_bias: [f64; 3],
    pub classes: Vec<ClassKnowledge>,
    #[serde(default)]
    pub legacy_snapshot: Option<LegacyPrototypeSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureMetadata {
    pub definition: super::FeatureDefinition,
    pub training_minimum: f64,
    pub training_maximum: f64,
    pub training_mean: f64,
    pub training_variance: f64,
    pub training_standard_deviation: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelApprovalStatus {
    Candidate,
    Approved,
    Rejected,
    RolledBack,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrototypeCandidateScore {
    pub prototypes_per_class: usize,
    pub validation_accuracy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationParameters {
    pub slope: f64,
    pub intercept: f64,
}

impl CalibrationParameters {
    fn predict(self, score: f64) -> f64 {
        let value = self.slope * score + self.intercept;
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exponential = value.exp();
            exponential / (1.0 + exponential)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyPrototypeSnapshot {
    pub source_model_version: u32,
    pub centers: [[f64; 4]; 3],
}

impl LegacyPrototypeSnapshot {
    pub fn new(centers: [[f64; 4]; 3]) -> Result<Self, SimError> {
        if centers.iter().flatten().any(|value| !value.is_finite()) {
            return Err(SimError::InvalidInput {
                parameter: "legacy_prototypes",
                reason: "semua nilai prototype lama harus finite",
            });
        }
        Ok(Self {
            source_model_version: 1,
            centers,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassPrediction {
    pub class_id: usize,
    pub prototype_id: usize,
    pub score: f64,
    pub margin: f64,
    pub calibrated_confidence: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassKnowledgeInference {
    pub prediction_without_memory: Option<usize>,
    pub prediction_from_knowledge: usize,
    pub prediction_from_network: Option<usize>,
    pub prototype_id: String,
    pub injected_currents_na: Vec<f64>,
    pub spiking_neurons: Vec<usize>,
    pub completed_step: u64,
}

impl ClassKnowledgeModel {
    /// Fit k-means candidates on train only and select k by validation accuracy.
    pub fn fit(
        training: &[IrisSample],
        validation: &[IrisSample],
        dataset_version: impl Into<String>,
        training_run_id: impl Into<String>,
        seed: u64,
    ) -> Result<Self, SimError> {
        Self::fit_with_exemplar_policy(
            training,
            validation,
            dataset_version,
            training_run_id,
            seed,
            ExemplarPolicy::default(),
        )
    }

    pub fn fit_with_exemplar_policy(
        training: &[IrisSample],
        validation: &[IrisSample],
        dataset_version: impl Into<String>,
        training_run_id: impl Into<String>,
        seed: u64,
        exemplar_policy: ExemplarPolicy,
    ) -> Result<Self, SimError> {
        if training.is_empty() || validation.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "class_knowledge_split",
                reason: "training dan validation harus berisi sampel",
            });
        }
        let dataset_version = dataset_version.into();
        let training_run_id = training_run_id.into();
        let scaler = FeatureScaler::fit(training)?;
        let training_rows = training
            .iter()
            .map(|sample| scaler.transform(*sample.features()))
            .collect::<Result<Vec<_>, _>>()?;
        let validation_rows = validation
            .iter()
            .map(|sample| scaler.transform(*sample.features()))
            .collect::<Result<Vec<_>, _>>()?;
        let class_rows = grouped_rows(training, &training_rows)?;
        if class_rows.iter().any(|rows| rows.len() < 2) {
            return Err(SimError::InvalidInput {
                parameter: "class_knowledge_training",
                reason: "setiap kelas harus memiliki minimal dua sampel training",
            });
        }
        let mut best: Option<(f64, usize, Vec<Vec<Prototype>>)> = None;
        let mut prototype_candidate_scores = Vec::new();
        for prototype_count in CANDIDATE_PROTOTYPES {
            if class_rows.iter().any(|rows| rows.len() < prototype_count) {
                continue;
            }
            let candidates = class_rows
                .iter()
                .enumerate()
                .map(|(class_id, rows)| {
                    fit_kmeans(
                        rows,
                        prototype_count,
                        seed.wrapping_add(class_id as u64),
                        class_id,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let candidate_exemplars = class_rows
                .iter()
                .zip(&candidates)
                .enumerate()
                .map(|(class_id, (rows, _))| {
                    select_exemplar_vectors(
                        rows,
                        &candidates,
                        class_id,
                        exemplar_policy.max_per_class,
                    )
                })
                .collect::<Vec<_>>();
            let correct = validation
                .iter()
                .zip(&validation_rows)
                .filter(|(sample, row)| {
                    predict_from_with_exemplars(&candidates, &candidate_exemplars, row).0
                        == sample.label()
                })
                .count();
            let accuracy = correct as f64 / validation.len() as f64;
            prototype_candidate_scores.push(PrototypeCandidateScore {
                prototypes_per_class: prototype_count,
                validation_accuracy: accuracy,
            });
            if best.as_ref().is_none_or(|(best_accuracy, best_count, _)| {
                accuracy > *best_accuracy
                    || (accuracy == *best_accuracy && prototype_count < *best_count)
            }) {
                best = Some((accuracy, prototype_count, candidates));
            }
        }
        let (validation_accuracy, selected_prototypes_per_class, candidate_prototypes) = best
            .ok_or(SimError::InvalidInput {
                parameter: "prototype_candidates",
                reason: "jumlah kandidat prototype tidak valid untuk ukuran kelas training",
            })?;
        let selected_exemplar_vectors = class_rows
            .iter()
            .enumerate()
            .map(|(class_id, rows)| {
                select_exemplar_vectors(
                    rows,
                    &candidate_prototypes,
                    class_id,
                    exemplar_policy.max_per_class,
                )
            })
            .collect::<Vec<_>>();
        let calibration = (0..CLASS_NAMES.len())
            .map(|class_id| {
                let calibration_samples = validation
                    .iter()
                    .zip(&validation_rows)
                    .map(|(sample, row)| {
                        let (_, distance) = hybrid_class_distance(
                            row,
                            &candidate_prototypes[class_id],
                            &selected_exemplar_vectors[class_id],
                        );
                        (-distance, sample.label() == class_id)
                    })
                    .collect::<Vec<_>>();
                let fitted = PlattCalibrator::fit(&calibration_samples, 1_000, 0.05)?;
                Ok(CalibrationParameters {
                    slope: fitted.slope(),
                    intercept: fitted.intercept(),
                })
            })
            .collect::<Result<Vec<_>, SimError>>()?
            .try_into()
            .map_err(|_| SimError::InvalidInput {
                parameter: "class_calibration",
                reason: "kalibrator harus tersedia untuk seluruh kelas",
            })?;
        let training_statistics = FeatureStatistics::compute(&training_rows);
        let feature_metadata = feature_definitions()
            .into_iter()
            .enumerate()
            .map(|(index, definition)| FeatureMetadata {
                definition,
                training_minimum: scaler.minimum()[index],
                training_maximum: scaler.maximum()[index],
                training_mean: training_statistics.mean[index],
                training_variance: training_statistics.variance[index],
                training_standard_deviation: training_statistics.standard_deviation[index],
            })
            .collect();
        let mut classes = Vec::with_capacity(CLASS_NAMES.len());
        for class_id in 0..CLASS_NAMES.len() {
            let indices = training
                .iter()
                .enumerate()
                .filter_map(|(index, sample)| (sample.label() == class_id).then_some(index))
                .collect::<Vec<_>>();
            let rows = indices
                .iter()
                .map(|index| training_rows[*index])
                .collect::<Vec<_>>();
            let prototypes = candidate_prototypes[class_id].clone();
            let exemplars = select_exemplars(
                ExemplarSelection {
                    samples: training,
                    rows: &training_rows,
                    indices: &indices,
                    prototypes: &candidate_prototypes,
                    exemplar_vectors: &selected_exemplar_vectors,
                    calibration: &calibration,
                    dataset_version: &dataset_version,
                    training_run_id: &training_run_id,
                },
                exemplar_policy.max_per_class,
                class_id,
            );
            classes.push(ClassKnowledge {
                class_id,
                class_name: CLASS_NAMES[class_id].to_owned(),
                sample_count: rows.len(),
                statistics: FeatureStatistics::compute(&rows),
                prototypes,
                exemplars,
            });
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Ok(Self {
            model_id: format!("iris-class-knowledge-{training_run_id}"),
            schema_version: MODEL_SCHEMA_VERSION,
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            scoring_version: SCORING_VERSION,
            model_version: 1,
            parent_model_version: None,
            approval_status: ModelApprovalStatus::Candidate,
            dataset_version,
            training_run_id,
            seed,
            created_at_unix: now,
            updated_at_unix: now,
            feature_schema: feature_definitions().into_iter().collect(),
            feature_metadata,
            scaler,
            training_statistics,
            selected_prototypes_per_class,
            prototype_candidate_scores,
            exemplar_policy,
            validation_accuracy,
            calibration,
            class_score_bias: [0.0; 3],
            classes,
            legacy_snapshot: None,
        })
    }

    /// Refit modern knowledge from train/validation while retaining every legacy center.
    pub fn migrate_legacy(
        legacy_snapshot: LegacyPrototypeSnapshot,
        training: &[IrisSample],
        validation: &[IrisSample],
        dataset_version: impl Into<String>,
        training_run_id: impl Into<String>,
        seed: u64,
    ) -> Result<Self, SimError> {
        let mut model = Self::fit(training, validation, dataset_version, training_run_id, seed)?;
        model.legacy_snapshot = Some(legacy_snapshot);
        Ok(model)
    }

    /// Predict using nearest normalized prototype. Score and margin are not probabilities.
    pub fn predict(&self, raw: [f64; 4]) -> Result<ClassPrediction, SimError> {
        self.validate()?;
        let row = self.scaler.transform(raw)?;
        let (class_id, prototype_id, distance, margin) =
            predict_with_bias(&self.classes, &row, &self.class_score_bias);
        Ok(ClassPrediction {
            class_id,
            prototype_id,
            score: -distance,
            margin,
            calibrated_confidence: Some(self.calibration[class_id].predict(-distance)),
        })
    }

    pub fn apply_validation_feedback(
        &mut self,
        expected_class: usize,
        predicted_class: usize,
        learning_rate: f64,
    ) -> Result<(), SimError> {
        if expected_class >= self.class_score_bias.len()
            || predicted_class >= self.class_score_bias.len()
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
        {
            return Err(SimError::InvalidInput {
                parameter: "class_knowledge_feedback",
                reason: "kelas harus valid dan learning rate harus finite serta positif",
            });
        }
        let parent_version = self.model_version;
        let next_version = self
            .model_version
            .checked_add(1)
            .ok_or(SimError::CounterOverflow {
                counter: "versi class knowledge",
            })?;
        self.class_score_bias[expected_class] =
            (self.class_score_bias[expected_class] + learning_rate).clamp(-2.0, 2.0);
        if predicted_class != expected_class {
            self.class_score_bias[predicted_class] =
                (self.class_score_bias[predicted_class] - learning_rate).clamp(-2.0, 2.0);
        }
        self.model_version = next_version;
        self.parent_model_version = Some(parent_version);
        self.updated_at_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Ok(())
    }

    pub fn refit_calibration(&mut self, validation: &[IrisSample]) -> Result<(), SimError> {
        if validation.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "calibration_validation",
                reason: "validation tidak boleh kosong",
            });
        }
        let rows = validation
            .iter()
            .map(|sample| self.scaler.transform(*sample.features()))
            .collect::<Result<Vec<_>, _>>()?;
        let calibration = (0..self.classes.len())
            .map(|class_id| {
                let samples = validation
                    .iter()
                    .zip(&rows)
                    .map(|(sample, row)| {
                        let class = &self.classes[class_id];
                        let embeddings = class
                            .exemplars
                            .iter()
                            .map(|exemplar| exemplar.embedding)
                            .collect::<Vec<_>>();
                        let distance = hybrid_class_distance(row, &class.prototypes, &embeddings).1
                            - self.class_score_bias[class_id];
                        (-distance, sample.label() == class_id)
                    })
                    .collect::<Vec<_>>();
                let fitted = PlattCalibrator::fit(&samples, 1_000, 0.05)?;
                Ok(CalibrationParameters {
                    slope: fitted.slope(),
                    intercept: fitted.intercept(),
                })
            })
            .collect::<Result<Vec<_>, SimError>>()?
            .try_into()
            .map_err(|_| SimError::InvalidInput {
                parameter: "class_calibration",
                reason: "kalibrator harus tersedia untuk seluruh kelas",
            })?;
        self.calibration = calibration;
        Ok(())
    }

    pub fn predict_normalized_primary(
        &self,
        normalized: [f64; 4],
    ) -> Result<ClassPrediction, SimError> {
        self.predict(self.scaler.restore_primary(normalized)?)
    }

    /// Run a matched baseline and knowledge-injected probe through actual LIF neurons.
    /// The supplied network is updated only after the full knowledge path succeeds.
    pub fn infer_with_lif(
        &self,
        network: &mut crate::LifNetwork,
        raw: [f64; 4],
        action_neurons: &[usize],
    ) -> Result<ClassKnowledgeInference, SimError> {
        self.validate()?;
        if action_neurons.len() != CLASS_NAMES.len() {
            return Err(SimError::InvalidInput {
                parameter: "action_neurons",
                reason: "harus memetakan tepat satu neuron untuk setiap kelas Iris",
            });
        }
        let mut unique = std::collections::BTreeSet::new();
        if action_neurons
            .iter()
            .any(|neuron| *neuron >= network.neuron_count() || !unique.insert(*neuron))
        {
            return Err(SimError::InvalidInput {
                parameter: "action_neurons",
                reason: "indeks neuron harus valid dan unik",
            });
        }
        if network.neuron_count() < raw.len() {
            return Err(SimError::InvalidTopology {
                reason: "network harus menyediakan empat neuron input fitur",
            });
        }
        let knowledge_prediction = self.predict(raw)?;
        let row = self.scaler.transform(raw)?;
        let input_currents = (0..network.neuron_count())
            .map(|index| if index < 4 { row[index] * 20.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let distances = self
            .classes
            .iter()
            .map(|class| {
                let embeddings = class
                    .exemplars
                    .iter()
                    .map(|exemplar| exemplar.embedding)
                    .collect::<Vec<_>>();
                hybrid_class_distance(&row, &class.prototypes, &embeddings).1
                    - self.class_score_bias[class.class_id]
            })
            .collect::<Vec<_>>();
        let best_distance = distances.iter().copied().fold(f64::INFINITY, f64::min);
        let mut affinity = distances
            .iter()
            .map(|distance| (-(distance - best_distance) * 8.0).exp())
            .collect::<Vec<_>>();
        let affinity_total = affinity.iter().sum::<f64>();
        if !affinity_total.is_finite() || affinity_total <= 0.0 {
            return Err(SimError::NumericalOverflow {
                operation: "afinitas prototype untuk LIF",
            });
        }
        for value in &mut affinity {
            *value /= affinity_total;
        }
        let mut injected_currents_na = vec![0.0; network.neuron_count()];
        for (class_id, neuron) in action_neurons.iter().enumerate() {
            injected_currents_na[*neuron] = affinity[class_id] * 30.0;
        }

        let mut baseline = network.clone();
        let mut baseline_spikes = Vec::new();
        for current in [
            input_currents.as_slice(),
            &vec![0.0; network.neuron_count()],
            &vec![0.0; network.neuron_count()],
            &vec![0.0; network.neuron_count()],
        ] {
            let step = baseline.step(current)?;
            baseline_spikes.extend_from_slice(step.spiking_neurons());
        }
        let prediction_without_memory =
            action_from_response(&baseline, action_neurons, &baseline_spikes);

        let mut knowledge_network = network.clone();
        let mut spiking_neurons = Vec::new();
        let mut last_step = 0;
        for current in [
            input_currents.as_slice(),
            injected_currents_na.as_slice(),
            &vec![0.0; network.neuron_count()],
            &vec![0.0; network.neuron_count()],
        ] {
            let step = knowledge_network.step(current)?;
            last_step = step.completed_step();
            spiking_neurons.extend_from_slice(step.spiking_neurons());
        }
        let prediction_from_network =
            action_from_response(&knowledge_network, action_neurons, &spiking_neurons);
        *network = knowledge_network;
        Ok(ClassKnowledgeInference {
            prediction_without_memory,
            prediction_from_knowledge: knowledge_prediction.class_id,
            prediction_from_network,
            prototype_id: format!(
                "class-{}-prototype-{}",
                knowledge_prediction.class_id, knowledge_prediction.prototype_id
            ),
            injected_currents_na,
            spiking_neurons,
            completed_step: last_step,
        })
    }

    pub fn infer_normalized_primary_with_lif(
        &self,
        network: &mut crate::LifNetwork,
        normalized: [f64; 4],
        action_neurons: &[usize],
    ) -> Result<ClassKnowledgeInference, SimError> {
        let raw = self.scaler.restore_primary(normalized)?;
        self.infer_with_lif(network, raw, action_neurons)
    }

    pub fn validate(&self) -> Result<(), SimError> {
        if self.schema_version != MODEL_SCHEMA_VERSION {
            return Err(SimError::UnsupportedDataVersion {
                subsystem: "class knowledge",
                version: self.schema_version,
            });
        }
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION
            || self.scoring_version != SCORING_VERSION
            || self.feature_schema != feature_definitions()
            || self.feature_metadata.len() != FEATURE_COUNT
            || self.classes.len() != CLASS_NAMES.len()
            || self.model_id.trim().is_empty()
            || self.training_run_id.trim().is_empty()
            || self.dataset_version.trim().is_empty()
            || self.model_version == 0
            || !self.validation_accuracy.is_finite()
            || !(0.0..=1.0).contains(&self.validation_accuracy)
            || self.class_score_bias.iter().any(|value| !value.is_finite())
            || self.selected_prototypes_per_class == 0
            || self.exemplar_policy.max_per_class == 0
        {
            return Err(SimError::CorruptData {
                subsystem: "class knowledge",
                reason: "metadata schema atau metrik model tidak konsisten".to_owned(),
            });
        }
        self.scaler.validate()?;
        if self
            .feature_metadata
            .iter()
            .enumerate()
            .any(|(index, metadata)| {
                metadata.definition != feature_definitions()[index]
                    || !metadata.training_minimum.is_finite()
                    || !metadata.training_maximum.is_finite()
                    || !metadata.training_mean.is_finite()
                    || !metadata.training_variance.is_finite()
                    || metadata.training_variance < 0.0
                    || !metadata.training_standard_deviation.is_finite()
                    || metadata.training_standard_deviation < 0.0
                    || metadata.training_minimum != self.scaler.minimum()[index]
                    || metadata.training_maximum != self.scaler.maximum()[index]
            })
        {
            return Err(SimError::CorruptData {
                subsystem: "class knowledge",
                reason: "metadata training per fitur tidak valid".to_owned(),
            });
        }
        if self.prototype_candidate_scores.is_empty()
            || self.prototype_candidate_scores.iter().any(|candidate| {
                candidate.prototypes_per_class == 0
                    || !candidate.validation_accuracy.is_finite()
                    || !(0.0..=1.0).contains(&candidate.validation_accuracy)
            })
            || !valid_statistics(&self.training_statistics)
        {
            return Err(SimError::CorruptData {
                subsystem: "class knowledge",
                reason: "candidate score atau statistik training tidak valid".to_owned(),
            });
        }
        for (class_id, class) in self.classes.iter().enumerate() {
            if class.class_id != class_id
                || class.class_name != CLASS_NAMES[class_id]
                || class.prototypes.len() != self.selected_prototypes_per_class
                || class.sample_count == 0
                || class.sample_count != class.statistics.sample_count
                || class.exemplars.len() > self.exemplar_policy.max_per_class
            {
                return Err(SimError::CorruptData {
                    subsystem: "class knowledge",
                    reason: "identitas class atau jumlah prototype tidak konsisten".to_owned(),
                });
            }
            if !valid_statistics(&class.statistics) {
                return Err(SimError::CorruptData {
                    subsystem: "class knowledge",
                    reason: "statistik class tidak valid".to_owned(),
                });
            }
            for prototype in &class.prototypes {
                if prototype.class_id != class_id
                    || prototype.member_count == 0
                    || prototype.statistics.sample_count != prototype.member_count
                    || prototype
                        .feature_vector
                        .iter()
                        .any(|value| !value.is_finite())
                    || prototype.embedding.iter().any(|value| !value.is_finite())
                    || !prototype.importance.is_finite()
                    || prototype.importance < 0.0
                    || !valid_statistics(&prototype.statistics)
                {
                    return Err(SimError::CorruptData {
                        subsystem: "class knowledge",
                        reason: "prototype mengandung metadata atau fitur invalid".to_owned(),
                    });
                }
            }
            for exemplar in &class.exemplars {
                if exemplar.class_id != class_id
                    || exemplar.expected_label != class_id
                    || exemplar.predicted_label_at_capture >= CLASS_NAMES.len()
                    || exemplar.source_dataset_version.trim().is_empty()
                    || exemplar.source_training_run.trim().is_empty()
                    || exemplar.raw_features.iter().any(|value| !value.is_finite())
                    || exemplar
                        .derived_features
                        .iter()
                        .any(|value| !value.is_finite())
                    || exemplar.embedding.iter().any(|value| !value.is_finite())
                    || !exemplar.uncertainty.is_finite()
                    || !exemplar.boundary_score.is_finite()
                    || !exemplar.importance.is_finite()
                    || !exemplar.feedback.is_finite()
                    || exemplar
                        .calibrated_confidence
                        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
                {
                    return Err(SimError::CorruptData {
                        subsystem: "class knowledge",
                        reason: "exemplar memiliki nilai di luar domain".to_owned(),
                    });
                }
            }
        }
        if self
            .calibration
            .iter()
            .any(|parameters| !parameters.slope.is_finite() || !parameters.intercept.is_finite())
        {
            return Err(SimError::CorruptData {
                subsystem: "class knowledge",
                reason: "parameter Platt tidak finite".to_owned(),
            });
        }
        if self.legacy_snapshot.as_ref().is_some_and(|snapshot| {
            snapshot.source_model_version == 0
                || snapshot
                    .centers
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
        }) {
            return Err(SimError::CorruptData {
                subsystem: "class knowledge",
                reason: "legacy snapshot tidak valid".to_owned(),
            });
        }
        Ok(())
    }

    pub fn save_secure(&self, store: &mut crate::SecureMemoryStore) -> Result<(), SimError> {
        self.validate()?;
        let value = serde_json::to_value(self).map_err(|error| SimError::SerializationFailed {
            subsystem: "secure class knowledge",
            reason: error.to_string(),
        })?;
        store.put(
            format!("class-knowledge/model/{}", self.training_run_id),
            value,
            &[],
        )?;
        Ok(())
    }

    pub fn load_secure(
        store: &crate::SecureMemoryStore,
        training_run_id: &str,
    ) -> Result<Option<Self>, SimError> {
        if training_run_id.trim().is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "training_run_id",
                reason: "harus non-kosong",
            });
        }
        let key = format!("class-knowledge/model/{training_run_id}");
        let Some(record) = store.get(&key) else {
            return Ok(None);
        };
        let model: Self = serde_json::from_value(record.value().clone()).map_err(|error| {
            SimError::CorruptData {
                subsystem: "secure class knowledge",
                reason: error.to_string(),
            }
        })?;
        model.validate()?;
        Ok(Some(model))
    }
}

fn valid_statistics(statistics: &FeatureStatistics) -> bool {
    statistics.sample_count > 0
        && statistics.mean.iter().all(|value| value.is_finite())
        && statistics
            .variance
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && statistics
            .standard_deviation
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && statistics.minimum.iter().all(|value| value.is_finite())
        && statistics.maximum.iter().all(|value| value.is_finite())
        && statistics
            .quantiles
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        && statistics
            .covariance
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        && statistics.outlier_count <= statistics.sample_count
        && statistics.borderline_count <= statistics.sample_count
}

fn predict_with_bias(
    classes: &[ClassKnowledge],
    row: &[f64; FEATURE_COUNT],
    class_score_bias: &[f64; 3],
) -> (usize, usize, f64, f64) {
    let mut ranked = classes
        .iter()
        .map(|class| {
            let embeddings = class
                .exemplars
                .iter()
                .map(|exemplar| exemplar.embedding)
                .collect::<Vec<_>>();
            let (prototype_id, distance) =
                hybrid_class_distance(row, &class.prototypes, &embeddings);
            (
                distance - class_score_bias[class.class_id],
                class.class_id,
                prototype_id,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let (score, class_id, prototype_id) = ranked[0];
    let margin = ranked.get(1).map_or(0.0, |next| next.0 - score);
    (class_id, prototype_id, score, margin)
}

fn hybrid_class_distance(
    row: &[f64; FEATURE_COUNT],
    prototypes: &[Prototype],
    exemplars: &[[f64; FEATURE_COUNT]],
) -> (usize, f64) {
    let (prototype_id, prototype_distance) = prototypes
        .iter()
        .map(|prototype| {
            (
                prototype_index(&prototype.prototype_id),
                squared_distance(row, &prototype.feature_vector),
            )
        })
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        })
        .unwrap_or((0, f64::INFINITY));
    let distance = if exemplars.is_empty() {
        prototype_distance
    } else {
        let exemplar_distance = exemplars
            .iter()
            .map(|exemplar| squared_distance(row, exemplar))
            .fold(f64::INFINITY, f64::min);
        0.7 * prototype_distance + 0.3 * exemplar_distance
    };
    (prototype_id, distance)
}

fn action_from_response(
    network: &crate::LifNetwork,
    action_neurons: &[usize],
    spikes: &[usize],
) -> Option<usize> {
    action_neurons
        .iter()
        .position(|neuron| spikes.contains(neuron))
        .or_else(|| {
            action_neurons
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    network
                        .neuron(**left)
                        .map_or(f64::NEG_INFINITY, |state| state.membrane_potential_mv())
                        .total_cmp(
                            &network
                                .neuron(**right)
                                .map_or(f64::NEG_INFINITY, |state| state.membrane_potential_mv()),
                        )
                })
                .map(|(class_id, _)| class_id)
        })
}

fn grouped_rows(
    samples: &[IrisSample],
    rows: &[[f64; FEATURE_COUNT]],
) -> Result<Vec<Vec<[f64; FEATURE_COUNT]>>, SimError> {
    let mut groups = vec![Vec::new(); CLASS_NAMES.len()];
    for (sample, row) in samples.iter().zip(rows) {
        let group = groups
            .get_mut(sample.label())
            .ok_or(SimError::InvalidInput {
                parameter: "class_label",
                reason: "label kelas di luar rentang",
            })?;
        group.push(*row);
    }
    Ok(groups)
}

fn predict_from(
    classes: &[Vec<Prototype>],
    row: &[f64; FEATURE_COUNT],
) -> (usize, usize, f64, f64) {
    let flattened = classes
        .iter()
        .flat_map(|class| class.iter())
        .map(|prototype| {
            (
                prototype.class_id,
                prototype.prototype_id.clone(),
                prototype,
            )
        })
        .collect::<Vec<_>>();
    let (class_id, prototype_id, distance, margin) = predict_from_flat(&flattened, row);
    (class_id, prototype_id, distance, margin)
}

fn predict_from_with_exemplars(
    classes: &[Vec<Prototype>],
    exemplars: &[Vec<[f64; FEATURE_COUNT]>],
    row: &[f64; FEATURE_COUNT],
) -> (usize, usize) {
    let mut ranked = classes
        .iter()
        .enumerate()
        .map(|(class_id, prototypes)| {
            let (prototype_id, score) =
                hybrid_class_distance(row, prototypes, &exemplars[class_id]);
            (score, class_id, prototype_id)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let (_, class_id, prototype_id) = ranked[0];
    (class_id, prototype_id)
}

fn select_exemplar_vectors(
    rows: &[[f64; FEATURE_COUNT]],
    prototypes: &[Vec<Prototype>],
    class_id: usize,
    maximum: usize,
) -> Vec<[f64; FEATURE_COUNT]> {
    if rows.is_empty() || maximum == 0 {
        return Vec::new();
    }
    let mut scored = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let (predicted, _, _, margin) = predict_from(prototypes, row);
            let own_distance = prototypes[class_id]
                .iter()
                .map(|prototype| squared_distance(row, &prototype.feature_vector))
                .fold(f64::INFINITY, f64::min);
            (index, predicted, own_distance, margin)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        left.2
            .total_cmp(&right.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    let representative = scored[0];
    scored.sort_by(|left, right| {
        left.3
            .abs()
            .total_cmp(&right.3.abs())
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut selected = vec![representative.0];
    if let Some((index, _, _, _)) = scored.iter().find(|sample| sample.0 != representative.0) {
        selected.push(*index);
    }
    for prototype in &prototypes[class_id] {
        if selected.len() >= maximum {
            break;
        }
        if let Some((index, _)) = rows
            .iter()
            .enumerate()
            .map(|(index, row)| (index, squared_distance(row, &prototype.feature_vector)))
            .filter(|(index, _)| !selected.contains(index))
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            })
        {
            selected.push(index);
        }
    }
    selected
        .into_iter()
        .take(maximum)
        .map(|index| rows[index])
        .collect()
}

fn predict_from_flat(
    prototypes: &[(usize, String, &Prototype)],
    row: &[f64; FEATURE_COUNT],
) -> (usize, usize, f64, f64) {
    let mut distances = prototypes
        .iter()
        .map(|(class_id, _, prototype)| {
            (
                squared_distance(row, &prototype.feature_vector),
                *class_id,
                prototype,
            )
        })
        .collect::<Vec<_>>();
    distances.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.prototype_id.cmp(&right.2.prototype_id))
    });
    let Some((distance, class_id, prototype)) = distances.first() else {
        return (0, 0, f64::INFINITY, 0.0);
    };
    let next_class_distance = distances
        .iter()
        .find(|(_, other_class, _)| other_class != class_id)
        .map_or(*distance, |entry| entry.0);
    (
        *class_id,
        prototype_index(&prototype.prototype_id),
        *distance,
        next_class_distance - distance,
    )
}

fn prototype_index(id: &str) -> usize {
    id.rsplit('-')
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

struct ExemplarSelection<'a> {
    samples: &'a [IrisSample],
    rows: &'a [[f64; FEATURE_COUNT]],
    indices: &'a [usize],
    prototypes: &'a [Vec<Prototype>],
    exemplar_vectors: &'a [Vec<[f64; FEATURE_COUNT]>],
    calibration: &'a [CalibrationParameters; 3],
    dataset_version: &'a str,
    training_run_id: &'a str,
}

fn select_exemplars(
    selection: ExemplarSelection<'_>,
    maximum: usize,
    class_id: usize,
) -> Vec<Exemplar> {
    let ExemplarSelection {
        samples,
        rows,
        indices,
        prototypes,
        exemplar_vectors,
        calibration,
        dataset_version,
        training_run_id,
    } = selection;
    if indices.is_empty() {
        return Vec::new();
    }
    let mut scored = indices
        .iter()
        .map(|index| {
            let (predicted, _, _, margin) = predict_from(prototypes, &rows[*index]);
            let own_distance = prototypes[class_id]
                .iter()
                .map(|prototype| squared_distance(&rows[*index], &prototype.feature_vector))
                .fold(f64::INFINITY, f64::min);
            (*index, predicted, own_distance, margin)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        left.2
            .total_cmp(&right.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    let representative = scored[0];
    scored.sort_by(|left, right| {
        left.3
            .abs()
            .total_cmp(&right.3.abs())
            .then_with(|| left.0.cmp(&right.0))
    });
    let boundary = scored
        .iter()
        .find(|(index, _, _, _)| *index != representative.0)
        .copied();
    let mut selected = vec![(representative, ExemplarReason::Representative)];
    if let Some(boundary) = boundary {
        selected.push((boundary, ExemplarReason::ClassBoundary));
    }
    for prototype in &prototypes[class_id] {
        let nearest = indices
            .iter()
            .map(|index| {
                (
                    scored
                        .iter()
                        .find(|sample| sample.0 == *index)
                        .copied()
                        .unwrap_or((*index, class_id, f64::INFINITY, 0.0)),
                    squared_distance(&rows[*index], &prototype.feature_vector),
                )
            })
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.0.cmp(&right.0.0))
            })
            .map(|entry| entry.0);
        if let Some(nearest) = nearest
            && !selected.iter().any(|(sample, _)| sample.0 == nearest.0)
        {
            selected.push((nearest, ExemplarReason::ClusterRepresentative));
        }
    }
    selected
        .into_iter()
        .take(maximum)
        .map(|((index, predicted, _, margin), reason)| {
            let raw = *samples[index].features();
            let derived = IrisFeatures::from_raw(raw)
                .map(|features| *features.values())
                .unwrap_or([0.0; FEATURE_COUNT]);
            Exemplar {
                exemplar_id: format!("{training_run_id}-class-{class_id}-sample-{index}"),
                class_id,
                raw_features: raw,
                derived_features: derived,
                embedding: rows[index],
                expected_label: class_id,
                predicted_label_at_capture: predicted,
                calibrated_confidence: Some(
                    calibration[predicted].predict(
                        -hybrid_class_distance(
                            &rows[index],
                            &prototypes[predicted],
                            &exemplar_vectors[predicted],
                        )
                        .1,
                    ),
                ),
                uncertainty: 1.0 / (1.0 + margin.abs()),
                boundary_score: margin,
                importance: match reason {
                    ExemplarReason::ClassBoundary => 2.0,
                    ExemplarReason::Representative => 1.5,
                    ExemplarReason::ClusterRepresentative => 1.0,
                },
                usage_count: 0,
                last_retrieved_at_unix: None,
                feedback: if predicted == class_id { 1.0 } else { -1.0 },
                source_dataset_version: dataset_version.to_owned(),
                source_training_run: training_run_id.to_owned(),
                reason,
            }
        })
        .collect()
}

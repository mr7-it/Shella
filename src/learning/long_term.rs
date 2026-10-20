use crate::{
    ClassKnowledgeModel, LegacyPrototypeSnapshot, LifNetwork, MemoryStore, NetworkStep,
    RewardEvaluator, RewardEvaluatorConfig, SimError, TextEmbedding,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

const EXPERIENCE_VERSION: u32 = 1;
const EXPERIENCE_PREFIX: &str = "learning/experience/";
const CLASS_KNOWLEDGE_PREFIX: &str = "class-knowledge/model/";
const MAX_IMPORTANCE: f64 = 10.0;
const CONFLICT_SIMILARITY_THRESHOLD: f64 = 0.45;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StoredExperience {
    version: u32,
    task_id: String,
    features: Vec<f64>,
    expected_action: Option<usize>,
    feedback: f64,
    importance: f64,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

/// Konflik label lama yang mirip dengan observasi baru.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryConflict {
    record_id: String,
    task_id: String,
    previous_action: Option<usize>,
    new_action: Option<usize>,
    similarity: f64,
}

impl MemoryConflict {
    #[must_use]
    pub fn record_id(&self) -> &str {
        &self.record_id
    }
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }
    #[must_use]
    pub const fn previous_action(&self) -> Option<usize> {
        self.previous_action
    }
    #[must_use]
    pub const fn new_action(&self) -> Option<usize> {
        self.new_action
    }
    #[must_use]
    pub const fn similarity(&self) -> f64 {
        self.similarity
    }
}

/// Pengalaman terurut dari yang paling relevan untuk query.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRecall {
    record_id: String,
    task_id: String,
    features: Vec<f64>,
    expected_action: Option<usize>,
    feedback: f64,
    importance: f64,
    similarity: f64,
    retrieval_score: f64,
}

impl MemoryRecall {
    #[must_use]
    pub fn record_id(&self) -> &str {
        &self.record_id
    }
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }
    #[must_use]
    pub fn features(&self) -> &[f64] {
        &self.features
    }
    #[must_use]
    pub const fn expected_action(&self) -> Option<usize> {
        self.expected_action
    }
    #[must_use]
    pub const fn feedback(&self) -> f64 {
        self.feedback
    }
    #[must_use]
    pub const fn importance(&self) -> f64 {
        self.importance
    }
    #[must_use]
    pub const fn similarity(&self) -> f64 {
        self.similarity
    }
    /// Skor untuk pengurutan retrieval, bukan probabilitas atau confidence.
    #[must_use]
    pub const fn retrieval_score(&self) -> f64 {
        self.retrieval_score
    }
}

/// Hasil record feedback termasuk konflik yang terdeteksi sebelum insert.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryFeedbackWrite {
    record_id: String,
    prediction_correct: bool,
    conflicts: Vec<MemoryConflict>,
}

impl MemoryFeedbackWrite {
    #[must_use]
    pub fn record_id(&self) -> &str {
        &self.record_id
    }
    #[must_use]
    pub const fn prediction_correct(&self) -> bool {
        self.prediction_correct
    }
    #[must_use]
    pub fn conflicts(&self) -> &[MemoryConflict] {
        &self.conflicts
    }
}

/// Kedua transisi jaringan yang membentuk satu inferensi dengan arus retrieval.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryInference {
    input_step: NetworkStep,
    intermediate_step: NetworkStep,
    propagated_step: NetworkStep,
    recalled: Vec<MemoryRecall>,
    injected_currents_na: Vec<f64>,
}

impl MemoryInference {
    #[must_use]
    pub const fn input_step(&self) -> &NetworkStep {
        &self.input_step
    }
    #[must_use]
    pub const fn intermediate_step(&self) -> &NetworkStep {
        &self.intermediate_step
    }
    #[must_use]
    pub const fn propagated_step(&self) -> &NetworkStep {
        &self.propagated_step
    }
    #[must_use]
    pub fn recalled(&self) -> &[MemoryRecall] {
        &self.recalled
    }
    #[must_use]
    pub fn injected_currents_na(&self) -> &[f64] {
        &self.injected_currents_na
    }
    #[must_use]
    pub fn spiking_neurons(&self) -> Vec<usize> {
        let mut spiking = BTreeSet::new();
        spiking.extend(self.input_step.spiking_neurons().iter().copied());
        spiking.extend(self.intermediate_step.spiking_neurons().iter().copied());
        spiking.extend(self.propagated_step.spiking_neurons().iter().copied());
        spiking.into_iter().collect()
    }
}

/// Hasil replay yang benar-benar dijalankan melalui LIF/STDP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayReport {
    replayed_record_ids: Vec<String>,
    error_records_replayed: usize,
    presentations: usize,
    weight_updates: usize,
}

impl ReplayReport {
    #[must_use]
    pub fn replayed_record_ids(&self) -> &[String] {
        &self.replayed_record_ids
    }
    #[must_use]
    pub const fn error_records_replayed(&self) -> usize {
        self.error_records_replayed
    }
    #[must_use]
    pub const fn presentations(&self) -> usize {
        self.presentations
    }
    #[must_use]
    pub const fn weight_updates(&self) -> usize {
        self.weight_updates
    }
}

/// Hitungan pelupaan terbatas untuk audit retensi memori.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForgettingReport {
    examined: usize,
    protected: usize,
    decayed: usize,
    deleted: usize,
}

impl ForgettingReport {
    #[must_use]
    pub const fn examined(self) -> usize {
        self.examined
    }
    #[must_use]
    pub const fn protected(self) -> usize {
        self.protected
    }
    #[must_use]
    pub const fn decayed(self) -> usize {
        self.decayed
    }
    #[must_use]
    pub const fn deleted(self) -> usize {
        self.deleted
    }
}

/// Memori jangka panjang berbasis `MemoryStore` dengan retrieval dan replay adaptif.
#[derive(Debug)]
pub struct LearningMemory {
    store: MemoryStore,
    next_id: u64,
}

impl LearningMemory {
    /// Membuka store eksklusif dan memvalidasi seluruh record pengalaman.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, SimError> {
        let store = MemoryStore::open(path)?;
        let mut next_id = 1_u64;
        for record in store.records() {
            if let Some(suffix) = record.key().strip_prefix(EXPERIENCE_PREFIX) {
                let experience = decode_experience(record.value())?;
                validate_experience(&experience)?;
                let id =
                    u64::from_str_radix(suffix, 16).map_err(|error| SimError::CorruptData {
                        subsystem: "memori pembelajaran",
                        reason: error.to_string(),
                    })?;
                next_id = next_id.max(id.checked_add(1).ok_or(SimError::CounterOverflow {
                    counter: "ID pengalaman memori",
                })?);
            }
        }
        Ok(Self { store, next_id })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.store
            .records()
            .filter(|record| record.key().starts_with(EXPERIENCE_PREFIX))
            .count()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persist a validated versioned class model in the existing checksummed store.
    pub fn save_class_knowledge(&mut self, model: &ClassKnowledgeModel) -> Result<(), SimError> {
        model.validate()?;
        validate_task_id(&model.training_run_id)?;
        let value = serde_json::to_value(model).map_err(|error| SimError::SerializationFailed {
            subsystem: "class knowledge",
            reason: error.to_string(),
        })?;
        self.store.put(
            format!("{CLASS_KNOWLEDGE_PREFIX}{}", model.training_run_id),
            value,
            &[],
        )?;
        Ok(())
    }

    /// Restore a versioned class model; integrity is checked by MemoryStore on open.
    pub fn load_class_knowledge(
        &self,
        training_run_id: &str,
    ) -> Result<Option<ClassKnowledgeModel>, SimError> {
        validate_task_id(training_run_id)?;
        let key = format!("{CLASS_KNOWLEDGE_PREFIX}{training_run_id}");
        let Some(record) = self.store.get(&key) else {
            return Ok(None);
        };
        let model: ClassKnowledgeModel =
            serde_json::from_value(record.value().clone()).map_err(|error| {
                SimError::CorruptData {
                    subsystem: "class knowledge",
                    reason: error.to_string(),
                }
            })?;
        model.validate()?;
        Ok(Some(model))
    }

    /// Explicitly migrate the old three single-center records without discarding their values.
    pub fn migrate_legacy_prototypes(
        &mut self,
        legacy_task_id: &str,
        training: &[crate::IrisSample],
        validation: &[crate::IrisSample],
        dataset_version: impl Into<String>,
        training_run_id: impl Into<String>,
        seed: u64,
    ) -> Result<ClassKnowledgeModel, SimError> {
        let mut centers = [[0.0; 4]; 3];
        let mut seen = [false; 3];
        for (class, features) in self.load_prototypes(legacy_task_id)? {
            if class >= centers.len() || features.len() != 4 || seen[class] {
                return Err(SimError::CorruptData {
                    subsystem: "migrasi prototype legacy",
                    reason: "kelas prototype lama duplikat atau dimensinya tidak valid".to_owned(),
                });
            }
            centers[class].copy_from_slice(&features);
            seen[class] = true;
        }
        if seen.iter().any(|present| !present) {
            return Err(SimError::CorruptData {
                subsystem: "migrasi prototype legacy",
                reason: "prototype lama harus tersedia untuk ketiga kelas".to_owned(),
            });
        }
        let snapshot = LegacyPrototypeSnapshot::new(centers)?;
        let model = ClassKnowledgeModel::migrate_legacy(
            snapshot,
            training,
            validation,
            dataset_version,
            training_run_id,
            seed,
        )?;
        self.save_class_knowledge(&model)?;
        Ok(model)
    }

    /// Menghapus seluruh pengalaman dari task eksperimen yang tidak lolos validasi.
    /// Dipakai oleh adaptive self-improvement agar update negatif tidak menjadi
    /// pengetahuan permanen.
    pub fn discard_task(&mut self, task_id: &str) -> Result<usize, SimError> {
        validate_task_id(task_id)?;
        let keys = self
            .store
            .records()
            .filter_map(|record| {
                if !record.key().starts_with(EXPERIENCE_PREFIX) {
                    return None;
                }
                decode_experience(record.value())
                    .ok()
                    .filter(|experience| experience.task_id == task_id)
                    .map(|_| record.key().to_owned())
            })
            .collect::<Vec<_>>();
        let mut removed = 0;
        for key in keys {
            removed += usize::from(self.store.delete(&key)?);
        }
        Ok(removed)
    }

    /// Menyimpan prototype neural terlatih sebagai pengalaman konsolidasi.
    pub fn record_prototype(
        &mut self,
        task_id: &str,
        class: usize,
        features: &[f64],
    ) -> Result<MemoryFeedbackWrite, SimError> {
        self.record_with_feedback(task_id, features, Some(class), 1.0, 10.0)
    }

    /// Memulihkan prototype neural terakhir per class dari memory persisten.
    pub fn load_prototypes(&self, task_id: &str) -> Result<Vec<(usize, Vec<f64>)>, SimError> {
        validate_task_id(task_id)?;
        let mut prototypes = std::collections::BTreeMap::new();
        for record in self.store.records() {
            if !record.key().starts_with(EXPERIENCE_PREFIX) {
                continue;
            }
            let experience = decode_experience(record.value())?;
            if experience.task_id == task_id {
                if let Some(class) = experience.expected_action {
                    prototypes.insert(class, experience.features);
                }
            }
        }
        Ok(prototypes.into_iter().collect())
    }

    /// Menyimpan target dan prediksi; salah prediksi mendapat importance/replay lebih tinggi.
    pub fn record_feedback(
        &mut self,
        task_id: &str,
        features: &[f64],
        expected_action: Option<usize>,
        predicted_action: Option<usize>,
    ) -> Result<MemoryFeedbackWrite, SimError> {
        let correct = expected_action == predicted_action;
        let feedback = if correct { 1.0 } else { -1.0 };
        let importance = if correct { 1.0 } else { 2.0 };
        self.record_with_feedback(task_id, features, expected_action, feedback, importance)
    }

    /// Menyimpan feedback signed dari evaluator atau supervisor eksternal.
    pub fn record_with_feedback(
        &mut self,
        task_id: &str,
        features: &[f64],
        expected_action: Option<usize>,
        feedback: f64,
        importance: f64,
    ) -> Result<MemoryFeedbackWrite, SimError> {
        validate_task_id(task_id)?;
        validate_features(features)?;
        validate_feedback(feedback, importance)?;
        let conflicts =
            self.find_conflicts(features, expected_action, CONFLICT_SIMILARITY_THRESHOLD)?;
        let experience = StoredExperience {
            version: EXPERIENCE_VERSION,
            task_id: task_id.to_owned(),
            features: features.to_vec(),
            expected_action,
            feedback,
            importance,
            embedding: None,
        };
        let record_id = self.insert_experience(&experience)?;
        Ok(MemoryFeedbackWrite {
            record_id,
            prediction_correct: feedback >= 0.0,
            conflicts,
        })
    }

    /// Menyimpan pengalaman teks beserta embedding hashing deterministik 256 dimensi.
    pub fn record_text_feedback(
        &mut self,
        task_id: &str,
        text: &str,
        expected_action: Option<usize>,
        predicted_action: Option<usize>,
    ) -> Result<MemoryFeedbackWrite, SimError> {
        validate_task_id(task_id)?;
        let embedding = TextEmbedding::from_text(text, 256)?;
        let correct = expected_action == predicted_action;
        let experience = StoredExperience {
            version: EXPERIENCE_VERSION,
            task_id: task_id.to_owned(),
            features: embedding
                .values()
                .iter()
                .map(|value| f64::from(*value))
                .collect(),
            expected_action,
            feedback: if correct { 1.0 } else { -1.0 },
            importance: if correct { 1.0 } else { 2.0 },
            embedding: Some(embedding.values().to_vec()),
        };
        validate_experience(&experience)?;
        let conflicts = self.find_conflicts(
            &experience.features,
            expected_action,
            CONFLICT_SIMILARITY_THRESHOLD,
        )?;
        let record_id = self.insert_experience(&experience)?;
        Ok(MemoryFeedbackWrite {
            record_id,
            prediction_correct: correct,
            conflicts,
        })
    }

    /// Mengambil pengalaman teks berdasarkan kemiripan embedding yang tersimpan.
    pub fn recall_text_similar(
        &self,
        text: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecall>, SimError> {
        let query = TextEmbedding::from_text(text, 256)?;
        let query_features = query
            .values()
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        let mut recalls = self.recall_similar(&query_features, self.len())?;
        for recall in &mut recalls {
            let record = self
                .store
                .get_by_id(&recall.record_id)
                .ok_or(SimError::CorruptData {
                    subsystem: "memori pembelajaran",
                    reason: "record retrieval hilang".to_owned(),
                })?;
            let experience = decode_experience(record.value())?;
            if let Some(embedding) = experience.embedding {
                recall.similarity = query.cosine_similarity(&embedding)?;
                recall.retrieval_score = recall.similarity * (1.0 + recall.importance);
            }
        }
        recalls.retain(|recall| recall.similarity > 0.0);
        recalls.sort_by(|left, right| {
            right
                .retrieval_score
                .total_cmp(&left.retrieval_score)
                .then_with(|| left.record_id.cmp(&right.record_id))
        });
        recalls.truncate(limit);
        Ok(recalls)
    }

    /// Cosine similarity mengurutkan pengalaman; retrieval score tidak diklaim probabilitas.
    pub fn recall_similar(
        &self,
        query: &[f64],
        limit: usize,
    ) -> Result<Vec<MemoryRecall>, SimError> {
        self.recall_matching(query, limit, None)
    }

    /// Retrieval dibatasi pada task tertentu, untuk replay adaptif tanpa memasukkan task baru.
    pub fn recall_task_similar(
        &self,
        task_id: &str,
        query: &[f64],
        limit: usize,
    ) -> Result<Vec<MemoryRecall>, SimError> {
        validate_task_id(task_id)?;
        self.recall_matching(query, limit, Some(task_id))
    }

    fn recall_matching(
        &self,
        query: &[f64],
        limit: usize,
        task_filter: Option<&str>,
    ) -> Result<Vec<MemoryRecall>, SimError> {
        validate_features(query)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut recalled = Vec::new();
        recalled
            .try_reserve(self.len())
            .map_err(|_| SimError::AllocationFailed)?;
        for record in self.store.records() {
            if !record.key().starts_with(EXPERIENCE_PREFIX) {
                continue;
            }
            let experience = decode_experience(record.value())?;
            validate_experience(&experience)?;
            if task_filter.is_some_and(|task_id| experience.task_id != task_id) {
                continue;
            }
            let similarity = cosine_similarity(query, &experience.features);
            if similarity <= 0.0 {
                continue;
            }
            let error_priority = if experience.feedback < 0.0 { 2.0 } else { 1.0 };
            let retrieval_score = similarity * (1.0 + experience.importance) * error_priority;
            recalled.push(MemoryRecall {
                record_id: record.id().to_owned(),
                task_id: experience.task_id,
                features: experience.features,
                expected_action: experience.expected_action,
                feedback: experience.feedback,
                importance: experience.importance,
                similarity,
                retrieval_score,
            });
        }
        recalled.sort_by(|left, right| {
            right
                .retrieval_score
                .total_cmp(&left.retrieval_score)
                .then_with(|| left.record_id.cmp(&right.record_id))
        });
        recalled.truncate(limit);
        Ok(recalled)
    }

    /// Konflik ditemukan bila fitur sangat mirip tetapi target berbeda.
    pub fn find_conflicts(
        &self,
        features: &[f64],
        expected_action: Option<usize>,
        minimum_similarity: f64,
    ) -> Result<Vec<MemoryConflict>, SimError> {
        validate_features(features)?;
        if !minimum_similarity.is_finite() || !(0.0..=1.0).contains(&minimum_similarity) {
            return Err(SimError::InvalidInput {
                parameter: "minimum_similarity",
                reason: "harus finite dan berada pada rentang 0..1",
            });
        }
        Ok(self
            .recall_similar(features, self.len())?
            .into_iter()
            .filter(|memory| {
                memory.similarity >= minimum_similarity && memory.expected_action != expected_action
            })
            .map(|memory| MemoryConflict {
                record_id: memory.record_id,
                task_id: memory.task_id,
                previous_action: memory.expected_action,
                new_action: expected_action,
                similarity: memory.similarity,
            })
            .collect())
    }

    /// Konsolidasi menaikkan proteksi pengalaman agar tidak terhapus oleh decay.
    pub fn consolidate(&mut self, record_id: &str, increase: f64) -> Result<f64, SimError> {
        if !increase.is_finite() || increase <= 0.0 {
            return Err(SimError::InvalidInput {
                parameter: "consolidation_increase",
                reason: "harus finite dan positif",
            });
        }
        let record = self
            .store
            .get_by_id(record_id)
            .ok_or(SimError::InvalidInput {
                parameter: "record_id",
                reason: "pengalaman tidak ditemukan",
            })?;
        let key = record.key().to_owned();
        let mut experience = decode_experience(record.value())?;
        experience.importance = (experience.importance + increase).min(MAX_IMPORTANCE);
        let importance = experience.importance;
        self.store.put(key, encode_experience(&experience)?, &[])?;
        Ok(importance)
    }

    /// Melupakan hanya record tidak relevan dengan importance di bawah batas proteksi.
    pub fn decay_irrelevant(
        &mut self,
        query: &[f64],
        minimum_similarity: f64,
        decay_factor: f64,
        minimum_importance: f64,
        protected_importance: f64,
    ) -> Result<ForgettingReport, SimError> {
        validate_features(query)?;
        if !minimum_similarity.is_finite()
            || !(0.0..=1.0).contains(&minimum_similarity)
            || !decay_factor.is_finite()
            || !(0.0..1.0).contains(&decay_factor)
            || !minimum_importance.is_finite()
            || minimum_importance < 0.0
            || !protected_importance.is_finite()
            || protected_importance < minimum_importance
        {
            return Err(SimError::InvalidConfiguration {
                parameter: "forgetting_policy",
                reason: "threshold harus finite; decay 0..1 dan proteksi >= minimum importance",
            });
        }
        let records = self
            .store
            .records()
            .filter(|record| record.key().starts_with(EXPERIENCE_PREFIX))
            .cloned()
            .collect::<Vec<_>>();
        let mut report = ForgettingReport {
            examined: records.len(),
            protected: 0,
            decayed: 0,
            deleted: 0,
        };
        for record in records {
            let mut experience = decode_experience(record.value())?;
            let similarity = cosine_similarity(query, &experience.features);
            if similarity >= minimum_similarity || experience.importance >= protected_importance {
                report.protected += 1;
                continue;
            }
            experience.importance *= decay_factor;
            if experience.importance < minimum_importance {
                self.store.delete(record.key())?;
                report.deleted += 1;
            } else {
                self.store
                    .put(record.key(), encode_experience(&experience)?, &[])?;
                report.decayed += 1;
            }
        }
        Ok(report)
    }

    /// Retrieval menginjeksikan support sebagai arus LIF pada neuron aksi.
    pub fn infer_with_memory(
        &self,
        network: &mut LifNetwork,
        features: &[f64],
        action_neurons: &[usize],
        recall_limit: usize,
        memory_current_na: f64,
    ) -> Result<MemoryInference, SimError> {
        validate_network_features(network, features)?;
        validate_action_neurons(network, action_neurons)?;
        if !memory_current_na.is_finite() || memory_current_na < 0.0 {
            return Err(SimError::InvalidInput {
                parameter: "memory_current_na",
                reason: "harus finite dan tidak negatif",
            });
        }
        let recalled = self.recall_similar(features, recall_limit)?;
        self.infer_from_recall(
            network,
            features,
            action_neurons,
            memory_current_na,
            recalled,
            false,
        )
    }

    /// Inferensi menggunakan recall yang dibatasi task/context untuk meredam asosiasi konflik.
    pub fn infer_task_with_memory(
        &self,
        task_id: &str,
        network: &mut LifNetwork,
        features: &[f64],
        action_neurons: &[usize],
        recall_limit: usize,
        memory_current_na: f64,
    ) -> Result<MemoryInference, SimError> {
        validate_task_id(task_id)?;
        validate_network_features(network, features)?;
        validate_action_neurons(network, action_neurons)?;
        if !memory_current_na.is_finite() || memory_current_na < 0.0 {
            return Err(SimError::InvalidInput {
                parameter: "memory_current_na",
                reason: "harus finite dan tidak negatif",
            });
        }
        let recalled = self.recall_task_similar(task_id, features, recall_limit)?;
        self.infer_from_recall(
            network,
            features,
            action_neurons,
            memory_current_na,
            recalled,
            false,
        )
    }

    /// Inferensi robust untuk input bising; exemplar dipakai hanya jika similarity
    /// dan margin retrieval memadai sehingga noise tidak diperkuat sembarangan.
    pub fn infer_task_with_memory_robust(
        &self,
        task_id: &str,
        network: &mut LifNetwork,
        features: &[f64],
        action_neurons: &[usize],
        recall_limit: usize,
        memory_current_na: f64,
    ) -> Result<MemoryInference, SimError> {
        validate_task_id(task_id)?;
        validate_network_features(network, features)?;
        validate_action_neurons(network, action_neurons)?;
        if !memory_current_na.is_finite() || memory_current_na < 0.0 {
            return Err(SimError::InvalidInput {
                parameter: "memory_current_na",
                reason: "harus finite dan tidak negatif",
            });
        }
        let recalled = self.recall_task_similar(task_id, features, recall_limit)?;
        self.infer_from_recall(
            network,
            features,
            action_neurons,
            memory_current_na,
            recalled,
            true,
        )
    }

    /// Inferensi adaptif: memori hanya diinjeksikan jika similarity teratas melewati
    /// ambang validasi; jika tidak, jaringan berjalan sebagai kontrol tanpa memory.
    #[allow(clippy::too_many_arguments)]
    pub fn infer_task_with_memory_adaptive(
        &self,
        task_id: &str,
        network: &mut LifNetwork,
        features: &[f64],
        action_neurons: &[usize],
        recall_limit: usize,
        memory_current_na: f64,
        minimum_similarity: f64,
    ) -> Result<MemoryInference, SimError> {
        validate_task_id(task_id)?;
        validate_network_features(network, features)?;
        validate_action_neurons(network, action_neurons)?;
        if !minimum_similarity.is_finite() || !(0.0..=1.0).contains(&minimum_similarity) {
            return Err(SimError::InvalidInput {
                parameter: "minimum_similarity",
                reason: "harus finite dan berada pada rentang 0..1",
            });
        }
        let recalled = self.recall_task_similar(task_id, features, recall_limit)?;
        let use_memory = recalled
            .first()
            .is_some_and(|memory| memory.similarity() >= minimum_similarity);
        self.infer_from_recall(
            network,
            features,
            action_neurons,
            memory_current_na,
            if use_memory { recalled } else { Vec::new() },
            false,
        )
    }

    fn infer_from_recall(
        &self,
        network: &mut LifNetwork,
        features: &[f64],
        action_neurons: &[usize],
        memory_current_na: f64,
        recalled: Vec<MemoryRecall>,
        robust: bool,
    ) -> Result<MemoryInference, SimError> {
        let mut input = features.to_vec();
        let mut injected_currents_na = vec![0.0; network.neuron_count()];
        let mut support = vec![0.0_f64; action_neurons.len()];
        let confidence_boost = if robust {
            recalled.first().map_or(1.0, |best| {
                let second_similarity = recalled
                    .iter()
                    .skip(1)
                    .map(MemoryRecall::similarity)
                    .fold(0.0, f64::max);
                let margin = (best.similarity - second_similarity).max(0.0);
                if best.similarity >= 0.55 && margin >= 0.05 {
                    1.0 + 1.5 * best.similarity
                } else {
                    1.0
                }
            })
        } else {
            1.0
        };
        for memory in &recalled {
            if let Some(action) = memory.expected_action {
                let action_support = support.get_mut(action).ok_or(SimError::InvalidInput {
                    parameter: "memory_action",
                    reason: "aksi tersimpan di luar mapping action neuron",
                })?;
                *action_support = action_support.max(memory.similarity);
            }
        }
        for (action, similarity) in support.into_iter().enumerate() {
            let neuron = action_neurons[action];
            let current = memory_current_na * similarity * confidence_boost;
            input[neuron] += current;
            injected_currents_na[neuron] = current;
            if !input[neuron].is_finite() {
                return Err(SimError::NumericalOverflow {
                    operation: "arus retrieval memori",
                });
            }
        }
        let input_step = network.step(&input)?;
        let intermediate_step = network.step(&vec![0.0; network.neuron_count()])?;
        let propagated_step = network.step(&vec![0.0; network.neuron_count()])?;
        Ok(MemoryInference {
            input_step,
            intermediate_step,
            propagated_step,
            recalled,
            injected_currents_na,
        })
    }

    /// Replay berdasarkan similarity; contoh salah diprioritaskan oleh retrieval score.
    pub fn replay_relevant(
        &mut self,
        network: &mut LifNetwork,
        query: &[f64],
        action_neurons: &[usize],
        replay_limit: usize,
        teacher_current_na: f64,
    ) -> Result<ReplayReport, SimError> {
        self.replay_matching(
            network,
            query,
            action_neurons,
            replay_limit,
            teacher_current_na,
            None,
        )
    }

    /// Replay hanya dari task lama yang diminta, walau memori juga berisi pengalaman task baru.
    pub fn replay_task_relevant(
        &mut self,
        task_id: &str,
        network: &mut LifNetwork,
        query: &[f64],
        action_neurons: &[usize],
        replay_limit: usize,
        teacher_current_na: f64,
    ) -> Result<ReplayReport, SimError> {
        validate_task_id(task_id)?;
        self.replay_matching(
            network,
            query,
            action_neurons,
            replay_limit,
            teacher_current_na,
            Some(task_id),
        )
    }

    fn replay_matching(
        &mut self,
        network: &mut LifNetwork,
        query: &[f64],
        action_neurons: &[usize],
        replay_limit: usize,
        teacher_current_na: f64,
        task_filter: Option<&str>,
    ) -> Result<ReplayReport, SimError> {
        validate_network_features(network, query)?;
        validate_action_neurons(network, action_neurons)?;
        if !teacher_current_na.is_finite() || teacher_current_na <= 0.0 {
            return Err(SimError::InvalidInput {
                parameter: "teacher_current_na",
                reason: "harus finite dan positif",
            });
        }
        let recalled = self.recall_matching(query, replay_limit, task_filter)?;
        let mut evaluator = RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 1.0))?;
        let mut replayed_record_ids = Vec::new();
        let mut error_records_replayed = 0;
        let mut presentations = 0;
        let mut weight_updates = 0;
        for memory in recalled {
            error_records_replayed += usize::from(memory.feedback < 0.0);
            let old_weights = network
                .synapses()
                .iter()
                .map(|synapse| synapse.weight_na())
                .collect::<Vec<_>>();
            network.step(&memory.features)?;
            network.step(&vec![0.0; network.neuron_count()])?;
            let mut teacher = vec![0.0; network.neuron_count()];
            if let Some(action) = memory.expected_action {
                let neuron = *action_neurons.get(action).ok_or(SimError::InvalidInput {
                    parameter: "memory_action",
                    reason: "aksi tersimpan di luar mapping action neuron",
                })?;
                teacher[neuron] = teacher_current_na;
            }
            let response = network.step(&teacher)?;
            let observed = action_neurons
                .iter()
                .map(|neuron| response.spiking_neurons().contains(neuron))
                .collect::<Vec<_>>();
            evaluator.evaluate_and_apply(memory.expected_action, &observed, network)?;
            presentations += 1;
            weight_updates += network
                .synapses()
                .iter()
                .zip(old_weights)
                .filter(|(synapse, old_weight)| synapse.weight_na() != *old_weight)
                .count();
            self.consolidate(&memory.record_id, 0.1)?;
            replayed_record_ids.push(memory.record_id);
        }
        Ok(ReplayReport {
            replayed_record_ids,
            error_records_replayed,
            presentations,
            weight_updates,
        })
    }

    fn insert_experience(&mut self, experience: &StoredExperience) -> Result<String, SimError> {
        let key = loop {
            let key = format!("{EXPERIENCE_PREFIX}{:016x}", self.next_id);
            self.next_id = self
                .next_id
                .checked_add(1)
                .ok_or(SimError::CounterOverflow {
                    counter: "ID pengalaman memori",
                })?;
            if self.store.get(&key).is_none() {
                break key;
            }
        };
        self.store.put(
            key,
            encode_experience(experience)?,
            &["pengalaman".to_owned()],
        )
    }
}

fn encode_experience(experience: &StoredExperience) -> Result<Value, SimError> {
    serde_json::to_value(experience).map_err(|error| SimError::SerializationFailed {
        subsystem: "memori pembelajaran",
        reason: error.to_string(),
    })
}

fn decode_experience(value: &Value) -> Result<StoredExperience, SimError> {
    let experience: StoredExperience =
        serde_json::from_value(value.clone()).map_err(|error| SimError::CorruptData {
            subsystem: "memori pembelajaran",
            reason: error.to_string(),
        })?;
    if experience.version != EXPERIENCE_VERSION {
        return Err(SimError::UnsupportedDataVersion {
            subsystem: "memori pembelajaran",
            version: experience.version,
        });
    }
    Ok(experience)
}

fn validate_experience(experience: &StoredExperience) -> Result<(), SimError> {
    validate_task_id(&experience.task_id)?;
    validate_features(&experience.features)?;
    validate_feedback(experience.feedback, experience.importance)?;
    if let Some(embedding) = &experience.embedding {
        if embedding.is_empty() || embedding.iter().any(|value| !value.is_finite()) {
            return Err(SimError::CorruptData {
                subsystem: "memori pembelajaran",
                reason: "embedding pengalaman invalid".to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_task_id(task_id: &str) -> Result<(), SimError> {
    if task_id.trim().is_empty() || task_id.len() > 256 || task_id.chars().any(char::is_control) {
        return Err(SimError::InvalidInput {
            parameter: "task_id",
            reason: "harus non-kosong, tanpa karakter kontrol, dan <= 256 byte",
        });
    }
    Ok(())
}

fn validate_features(features: &[f64]) -> Result<(), SimError> {
    if features.is_empty() || features.iter().any(|feature| !feature.is_finite()) {
        return Err(SimError::InvalidInput {
            parameter: "features",
            reason: "harus non-kosong dan semua nilai finite",
        });
    }
    Ok(())
}

fn validate_feedback(feedback: f64, importance: f64) -> Result<(), SimError> {
    if !feedback.is_finite()
        || !(-1.0..=1.0).contains(&feedback)
        || !importance.is_finite()
        || !(0.0..=MAX_IMPORTANCE).contains(&importance)
    {
        return Err(SimError::InvalidInput {
            parameter: "feedback_or_importance",
            reason: "feedback harus -1..1 dan importance 0..10, keduanya finite",
        });
    }
    Ok(())
}

fn cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left, right) in left.iter().zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return f64::from(left_norm == right_norm);
    }
    (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(0.0, 1.0)
}

fn validate_network_features(network: &LifNetwork, features: &[f64]) -> Result<(), SimError> {
    validate_features(features)?;
    if features.len() != network.neuron_count() {
        return Err(SimError::InvalidInput {
            parameter: "features",
            reason: "panjang harus sama dengan jumlah neuron jaringan",
        });
    }
    Ok(())
}

fn validate_action_neurons(network: &LifNetwork, action_neurons: &[usize]) -> Result<(), SimError> {
    let mut unique = BTreeSet::new();
    if action_neurons.is_empty()
        || action_neurons
            .iter()
            .any(|neuron| *neuron >= network.neuron_count() || !unique.insert(*neuron))
    {
        return Err(SimError::InvalidInput {
            parameter: "action_neurons",
            reason: "harus berisi indeks neuron valid yang unik",
        });
    }
    Ok(())
}

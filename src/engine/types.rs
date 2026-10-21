use crate::language::{SemanticAnalysis, SemanticGapReport};
use crate::reasoning::{EvidenceContradiction, UncertaintyDecomposition};
use crate::{MemoryFeedbackWrite, MemoryRecall, NetworkStep, SimError};
use serde::{Deserialize, Serialize};

/// Parameter retrieval, perhatian temporal per detik, iterasi, dan abstention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    pub(super) iterations: usize,
    pub(super) recall_limit: usize,
    pub(super) memory_current_na: f64,
    pub(super) temporal_decay: f64,
    pub(super) minimum_confidence: f64,
    pub(super) context_window_capacity: usize,
    pub(super) adaptive_compute: bool,
}

impl ReasoningConfig {
    /// Membuat konfigurasi dengan 1 sampai 1024 langkah pemrosesan.
    pub fn new(
        iterations: usize,
        recall_limit: usize,
        memory_current_na: f64,
        temporal_decay: f64,
        minimum_confidence: f64,
    ) -> Result<Self, SimError> {
        if !(1..=1024).contains(&iterations) {
            return Err(SimError::InvalidConfiguration {
                parameter: "reasoning_iterations",
                reason: "harus berada pada rentang 1..=1024",
            });
        }
        if !memory_current_na.is_finite() || memory_current_na < 0.0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "memory_current_na",
                reason: "harus finite dan tidak negatif",
            });
        }
        if !temporal_decay.is_finite() || temporal_decay < 0.0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "temporal_decay",
                reason: "harus finite dan tidak negatif",
            });
        }
        if !minimum_confidence.is_finite() || !(0.0..=1.0).contains(&minimum_confidence) {
            return Err(SimError::InvalidConfiguration {
                parameter: "minimum_confidence",
                reason: "harus finite dan berada pada rentang 0..=1",
            });
        }
        Ok(Self {
            iterations,
            recall_limit,
            memory_current_na,
            temporal_decay,
            minimum_confidence,
            context_window_capacity: 0,
            adaptive_compute: false,
        })
    }

    /// Mengaktifkan buffer konteks dengan kapasitas maksimum 4096 transisi.
    pub fn with_context_window(mut self, capacity: usize) -> Result<Self, SimError> {
        if capacity > 4096 {
            return Err(SimError::InvalidConfiguration {
                parameter: "context_window_capacity",
                reason: "tidak boleh melebihi 4096 transisi",
            });
        }
        self.context_window_capacity = capacity;
        Ok(self)
    }

    /// Mengaktifkan budget langkah adaptif hingga batas `iterations`.
    #[must_use]
    pub const fn with_adaptive_compute(mut self, enabled: bool) -> Self {
        self.adaptive_compute = enabled;
        self
    }

    #[must_use]
    pub const fn iterations(self) -> usize {
        self.iterations
    }

    #[must_use]
    pub const fn recall_limit(self) -> usize {
        self.recall_limit
    }

    #[must_use]
    pub const fn memory_current_na(self) -> f64 {
        self.memory_current_na
    }

    #[must_use]
    pub const fn temporal_decay(self) -> f64 {
        self.temporal_decay
    }

    #[must_use]
    pub const fn minimum_confidence(self) -> f64 {
        self.minimum_confidence
    }

    #[must_use]
    pub const fn context_window_capacity(self) -> usize {
        self.context_window_capacity
    }

    #[must_use]
    pub const fn adaptive_compute(self) -> bool {
        self.adaptive_compute
    }
}

/// Satu transisi jaringan aktual beserta arus masukan yang dipakainya.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningStep {
    pub(super) iteration: usize,
    pub(super) currents_na: Vec<f64>,
    pub(super) network_step: NetworkStep,
}

impl ReasoningStep {
    #[must_use]
    pub const fn iteration(&self) -> usize {
        self.iteration
    }

    #[must_use]
    pub fn currents_na(&self) -> &[f64] {
        &self.currents_na
    }

    #[must_use]
    pub const fn network_step(&self) -> &NetworkStep {
        &self.network_step
    }
}

/// Bobot perhatian untuk satu pengalaman yang di-recall.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryAttention {
    pub(super) record_id: String,
    pub(super) similarity: f64,
    pub(super) weight: f64,
}

impl MemoryAttention {
    #[must_use]
    pub fn record_id(&self) -> &str {
        &self.record_id
    }

    #[must_use]
    pub const fn similarity(&self) -> f64 {
        self.similarity
    }

    /// Bobot ternormalisasi setelah faktor relevansi dan selisih timestamp.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

/// Trace lengkap dan metrik heuristik dari satu proses penalaran.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningResult {
    pub(super) selected_action: Option<usize>,
    pub(super) confidence: f64,
    pub(super) calibrated_confidence: Option<f64>,
    pub(super) uncertainty: f64,
    pub(super) evidence_strength: f64,
    pub(super) uncertainty_decomposition: UncertaintyDecomposition,
    pub(super) compute_budget: usize,
    pub(super) action_support: Vec<f64>,
    pub(super) steps: Vec<ReasoningStep>,
    pub(super) recalled: Vec<MemoryRecall>,
    pub(super) attention: Vec<MemoryAttention>,
    pub(super) contradictions: Vec<EvidenceContradiction>,
    pub(super) injected_currents_na: Vec<f64>,
}

impl ReasoningResult {
    #[must_use]
    pub const fn selected_action(&self) -> Option<usize> {
        self.selected_action
    }

    /// Skor heuristik dari evidence, konsistensi, dan entropy; bukan probabilitas.
    #[must_use]
    pub const fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Confidence dari Platt calibrator yang dipasang dengan data validasi, bila tersedia.
    #[must_use]
    pub const fn calibrated_confidence(&self) -> Option<f64> {
        self.calibrated_confidence
    }

    /// Entropy ternormalisasi dari distribusi dukungan aksi, pada rentang 0..=1.
    #[must_use]
    pub const fn uncertainty(&self) -> f64 {
        self.uncertainty
    }

    #[must_use]
    pub const fn evidence_strength(&self) -> f64 {
        self.evidence_strength
    }

    #[must_use]
    pub const fn uncertainty_decomposition(&self) -> UncertaintyDecomposition {
        self.uncertainty_decomposition
    }

    #[must_use]
    pub const fn compute_budget(&self) -> usize {
        self.compute_budget
    }

    #[must_use]
    pub fn action_support(&self) -> &[f64] {
        &self.action_support
    }

    #[must_use]
    pub fn steps(&self) -> &[ReasoningStep] {
        &self.steps
    }

    #[must_use]
    pub fn recalled(&self) -> &[MemoryRecall] {
        &self.recalled
    }

    #[must_use]
    pub fn attention(&self) -> &[MemoryAttention] {
        &self.attention
    }

    #[must_use]
    pub fn contradictions(&self) -> &[EvidenceContradiction] {
        &self.contradictions
    }

    #[must_use]
    pub fn injected_currents_na(&self) -> &[f64] {
        &self.injected_currents_na
    }

    #[must_use]
    pub const fn abstained(&self) -> bool {
        self.selected_action.is_none()
    }
}

/// Ringkasan update memory dan readout jaringan dari feedback berlabel.
#[derive(Debug, Clone, PartialEq)]
pub struct LearningOutcome {
    pub(super) memory_write: MemoryFeedbackWrite,
    pub(super) updated_synapses: usize,
}

impl LearningOutcome {
    #[must_use]
    pub const fn memory_write(&self) -> &MemoryFeedbackWrite {
        &self.memory_write
    }

    #[must_use]
    pub const fn updated_synapses(&self) -> usize {
        self.updated_synapses
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticLearningOutcome {
    analysis: SemanticAnalysis,
    gaps: SemanticGapReport,
    learning: Option<LearningOutcome>,
}

impl SemanticLearningOutcome {
    #[must_use]
    pub const fn analysis(&self) -> &SemanticAnalysis {
        &self.analysis
    }

    #[must_use]
    pub const fn gaps(&self) -> &SemanticGapReport {
        &self.gaps
    }

    #[must_use]
    pub const fn learning(&self) -> Option<&LearningOutcome> {
        self.learning.as_ref()
    }

    pub(super) const fn new(
        analysis: SemanticAnalysis,
        gaps: SemanticGapReport,
        learning: Option<LearningOutcome>,
    ) -> Self {
        Self {
            analysis,
            gaps,
            learning,
        }
    }
}

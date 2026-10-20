use serde::{Deserialize, Serialize};

/// Transisi penalaran aktual yang disimpan dalam context window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextFrame {
    pub(crate) context_id: String,
    pub(crate) currents_na: Vec<f64>,
    pub(crate) spiking_neurons: Vec<usize>,
    pub(crate) completed_step: u64,
}

impl ContextFrame {
    #[must_use]
    pub fn context_id(&self) -> &str {
        &self.context_id
    }

    #[must_use]
    pub fn currents_na(&self) -> &[f64] {
        &self.currents_na
    }

    #[must_use]
    pub fn spiking_neurons(&self) -> &[usize] {
        &self.spiking_neurons
    }

    #[must_use]
    pub const fn completed_step(&self) -> u64 {
        self.completed_step
    }
}

/// Bukti statistik asosiasi fitur-aksi, bukan jaminan hubungan kausal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalAssociation {
    pub(crate) feature_index: usize,
    pub(crate) action: usize,
    pub(crate) supporting_experiences: usize,
    pub(crate) contradicting_experiences: usize,
}

impl CausalAssociation {
    #[must_use]
    pub const fn feature_index(&self) -> usize {
        self.feature_index
    }

    #[must_use]
    pub const fn action(&self) -> usize {
        self.action
    }

    #[must_use]
    pub const fn supporting_experiences(&self) -> usize {
        self.supporting_experiences
    }

    #[must_use]
    pub const fn contradicting_experiences(&self) -> usize {
        self.contradicting_experiences
    }
}

/// Bukti memori yang sangat mirip tetapi menyimpan label aksi berbeda.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceContradiction {
    pub(crate) left_record_id: String,
    pub(crate) right_record_id: String,
    pub(crate) left_action: usize,
    pub(crate) right_action: usize,
    pub(crate) similarity: f64,
}

impl EvidenceContradiction {
    #[must_use]
    pub fn left_record_id(&self) -> &str {
        &self.left_record_id
    }

    #[must_use]
    pub fn right_record_id(&self) -> &str {
        &self.right_record_id
    }

    #[must_use]
    pub const fn actions(&self) -> (usize, usize) {
        (self.left_action, self.right_action)
    }

    #[must_use]
    pub const fn similarity(&self) -> f64 {
        self.similarity
    }
}

/// Ketidakpastian prediksi dan ketidakcukupan evidence sebagai komponen terpisah.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UncertaintyDecomposition {
    pub(crate) prediction: f64,
    pub(crate) evidence_gap: f64,
}

impl UncertaintyDecomposition {
    #[must_use]
    pub const fn prediction(self) -> f64 {
        self.prediction
    }

    #[must_use]
    pub const fn evidence_gap(self) -> f64 {
        self.evidence_gap
    }
}

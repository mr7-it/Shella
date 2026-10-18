use serde::{Deserialize, Serialize};

/// Hasil evaluasi satu respons jaringan.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RewardEvaluation {
    pub(super) expected_action: Option<usize>,
    pub(super) predicted_action: Option<usize>,
    pub(super) correct: bool,
    pub(super) reward: f64,
}

impl RewardEvaluation {
    /// Aksi target; `None` berarti tugas mengharapkan tidak ada output spike.
    #[must_use]
    pub const fn expected_action(self) -> Option<usize> {
        self.expected_action
    }

    /// Aksi yang diprediksi; `None` berarti tidak ada output spike.
    #[must_use]
    pub const fn predicted_action(self) -> Option<usize> {
        self.predicted_action
    }

    /// Apakah prediksi tepat sama dengan target.
    #[must_use]
    pub const fn correct(self) -> bool {
        self.correct
    }

    /// Reward signed yang dapat dikirimkan ke jaringan.
    #[must_use]
    pub const fn reward(self) -> f64 {
        self.reward
    }
}

/// Metrik kumulatif evaluator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct RewardMetrics {
    pub(super) samples: u64,
    pub(super) correct: u64,
    pub(super) no_response: u64,
    pub(super) total_reward: f64,
}

impl RewardMetrics {
    /// Jumlah respons yang dievaluasi.
    #[must_use]
    pub const fn samples(self) -> u64 {
        self.samples
    }

    /// Jumlah respons yang cocok dengan target.
    #[must_use]
    pub const fn correct(self) -> u64 {
        self.correct
    }

    /// Jumlah respons tanpa spike aktif.
    #[must_use]
    pub const fn no_response(self) -> u64 {
        self.no_response
    }

    /// Akurasi, atau `None` sebelum sampel pertama.
    #[must_use]
    pub fn accuracy(self) -> Option<f64> {
        (self.samples != 0).then(|| self.correct as f64 / self.samples as f64)
    }

    /// Jumlah reward signed yang diberikan evaluator.
    #[must_use]
    pub const fn total_reward(self) -> f64 {
        self.total_reward
    }

    /// Reward rata-rata, atau `None` sebelum sampel pertama.
    #[must_use]
    pub fn mean_reward(self) -> Option<f64> {
        (self.samples != 0).then(|| self.total_reward / self.samples as f64)
    }
}

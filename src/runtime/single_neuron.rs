use crate::{
    config::LifConfig,
    error::SimError,
    neuron::lif::{LifNeuron, StepOutcome},
};
use serde::{Deserialize, Serialize};

/// Runtime sederhana deterministik untuk satu neuron.
///
/// Runtime ini menjalankan satu neuron LIF. Setiap pemanggilan [`Self::step`]
/// memproses tepat satu arus konstan untuk interval waktu berikutnya. Runtime
/// ini terpisah dari `LifNetwork`, yang mengelola sinkronisasi multi-neuron.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LifRuntime {
    neuron: LifNeuron,
}

impl LifRuntime {
    /// Membuat runtime dengan neuron pada potensial istirahat.
    pub fn new(config: LifConfig) -> Result<Self, SimError> {
        Ok(Self {
            neuron: LifNeuron::new(config)?,
        })
    }

    /// Membuat runtime pada potensial awal khusus yang masih di bawah ambang.
    pub fn with_initial_potential(
        config: LifConfig,
        initial_potential_mv: f64,
    ) -> Result<Self, SimError> {
        Ok(Self {
            neuron: LifNeuron::with_initial_potential(config, initial_potential_mv)?,
        })
    }

    /// Memproses satu langkah dengan arus masukan dalam nanoampere.
    ///
    /// Kegagalan input, overflow, atau penghitung tidak mengubah keadaan neuron
    /// untuk langkah yang ditolak. Pemanggilan langkah yang sukses sebelumnya
    /// tetap tersimpan.
    pub fn step(&mut self, current_na: f64) -> Result<StepOutcome, SimError> {
        self.neuron.advance(current_na)
    }

    /// Menjalankan irisan arus secara berurutan.
    ///
    /// Jika suatu unsur gagal, langkah sukses sebelumnya dalam irisan tetap
    /// diterapkan. Tidak ada rollback batch. Ruang hasil dialokasikan sebelum
    /// langkah pertama, sehingga kegagalan reservasi tidak memajukan simulasi.
    pub fn run(&mut self, currents_na: &[f64]) -> Result<Vec<StepOutcome>, SimError> {
        let mut outcomes = Vec::new();
        outcomes
            .try_reserve(currents_na.len())
            .map_err(|_| SimError::AllocationFailed)?;
        for &current_na in currents_na {
            outcomes.push(self.step(current_na)?);
        }
        Ok(outcomes)
    }

    /// Akses hanya-baca ke neuron yang dijalankan.
    #[must_use]
    pub const fn neuron(&self) -> &LifNeuron {
        &self.neuron
    }

    /// Konfigurasi runtime.
    #[must_use]
    pub const fn config(&self) -> LifConfig {
        self.neuron.config()
    }

    pub(crate) fn validate_restored_state(&self) -> Result<(), SimError> {
        self.neuron.validate_restored_state()
    }
}

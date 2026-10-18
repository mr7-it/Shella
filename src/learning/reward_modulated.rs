use super::{StdpConfig, stdp::validate_positive};
use crate::SimError;
use serde::{Deserialize, Serialize};

/// Parameter STDP tiga faktor yang mengalikan jejak lokal dengan reward.
///
/// `learning_rate` harus positif dan finite. Reward eksternal bertanda dan
/// dianggap tanpa satuan; eligibility serta bobot memakai satuan bobot sinaps.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RewardModulatedStdpConfig {
    stdp: StdpConfig,
    eligibility_tau_steps: f64,
    learning_rate: f64,
}
impl RewardModulatedStdpConfig {
    /// Membuat konfigurasi; validasi dilakukan ketika mode diaktifkan.
    #[must_use]
    pub const fn new(stdp: StdpConfig, eligibility_tau_steps: f64, learning_rate: f64) -> Self {
        Self {
            stdp,
            eligibility_tau_steps,
            learning_rate,
        }
    }
    /// Memvalidasi aturan lokal, peluruhan eligibility, dan laju belajar.
    pub fn validate(self) -> Result<(), SimError> {
        self.stdp.validate()?;
        validate_positive("eligibility_tau_steps", self.eligibility_tau_steps)?;
        validate_positive("learning_rate", self.learning_rate)?;
        Ok(())
    }
    /// Aturan STDP lokal yang membentuk eligibility.
    #[must_use]
    pub const fn stdp(self) -> StdpConfig {
        self.stdp
    }
    /// Konstanta waktu eligibility dalam langkah jaringan.
    #[must_use]
    pub const fn eligibility_tau_steps(self) -> f64 {
        self.eligibility_tau_steps
    }
    /// Faktor pengali reward dan eligibility.
    #[must_use]
    pub const fn learning_rate(self) -> f64 {
        self.learning_rate
    }
}

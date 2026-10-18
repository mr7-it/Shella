use crate::error::SimError;
mod validation;
use serde::{Deserialize, Serialize};
use validation::{validate_finite, validate_positive};

/// Parameter neuron LIF.
///
/// Satuan yang digunakan adalah milidetik untuk waktu, milivolt untuk tegangan,
/// megaohm untuk resistansi, dan nanoampere untuk arus. Dengan konvensi ini,
/// hasil kali resistansi dan arus memiliki satuan milivolt.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LifConfig {
    dt_ms: f64,
    tau_m_ms: f64,
    resistance_mohm: f64,
    rest_potential_mv: f64,
    reset_potential_mv: f64,
    threshold_mv: f64,
    refractory_steps: usize,
}

impl LifConfig {
    /// Membuat konfigurasi; pemanggilannya harus diikuti validasi atau pembuatan runtime.
    #[must_use]
    pub const fn new(
        dt_ms: f64,
        tau_m_ms: f64,
        resistance_mohm: f64,
        rest_potential_mv: f64,
        reset_potential_mv: f64,
        threshold_mv: f64,
        refractory_steps: usize,
    ) -> Self {
        Self {
            dt_ms,
            tau_m_ms,
            resistance_mohm,
            rest_potential_mv,
            reset_potential_mv,
            threshold_mv,
            refractory_steps,
        }
    }

    /// Memvalidasi nilai hingga, tanda, serta hubungan ambang dan potensial.
    pub fn validate(self) -> Result<(), SimError> {
        validate_positive("dt_ms", self.dt_ms)?;
        validate_positive("tau_m_ms", self.tau_m_ms)?;
        validate_positive("resistance_mohm", self.resistance_mohm)?;
        validate_finite("rest_potential_mv", self.rest_potential_mv)?;
        validate_finite("reset_potential_mv", self.reset_potential_mv)?;
        validate_finite("threshold_mv", self.threshold_mv)?;

        if self.rest_potential_mv >= self.threshold_mv {
            return Err(SimError::InvalidConfiguration {
                parameter: "rest_potential_mv",
                reason: "harus lebih rendah daripada threshold_mv",
            });
        }
        if self.reset_potential_mv >= self.threshold_mv {
            return Err(SimError::InvalidConfiguration {
                parameter: "reset_potential_mv",
                reason: "harus lebih rendah daripada threshold_mv",
            });
        }
        Ok(())
    }

    /// Ukuran satu langkah simulasi dalam milidetik.
    #[must_use]
    pub const fn dt_ms(self) -> f64 {
        self.dt_ms
    }

    /// Konstanta waktu membran dalam milidetik.
    #[must_use]
    pub const fn tau_m_ms(self) -> f64 {
        self.tau_m_ms
    }

    /// Resistansi membran dalam megaohm.
    #[must_use]
    pub const fn resistance_mohm(self) -> f64 {
        self.resistance_mohm
    }

    /// Potensial istirahat dalam milivolt.
    #[must_use]
    pub const fn rest_potential_mv(self) -> f64 {
        self.rest_potential_mv
    }

    /// Potensial reset setelah spike dalam milivolt.
    #[must_use]
    pub const fn reset_potential_mv(self) -> f64 {
        self.reset_potential_mv
    }

    /// Ambang spike dalam milivolt.
    #[must_use]
    pub const fn threshold_mv(self) -> f64 {
        self.threshold_mv
    }

    /// Banyaknya langkah penuh dalam periode refraktori.
    #[must_use]
    pub const fn refractory_steps(self) -> usize {
        self.refractory_steps
    }
}

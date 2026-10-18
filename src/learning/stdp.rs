use crate::SimError;
use serde::{Deserialize, Serialize};

/// Parameter STDP pair-based dengan waktu dinyatakan dalam langkah simulasi.
///
/// `a_plus` dan `a_minus` adalah perubahan bobot per unit trace. Keduanya harus
/// positif; skala waktu juga positif dan finite.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StdpConfig {
    a_plus: f64,
    a_minus: f64,
    tau_plus_steps: f64,
    tau_minus_steps: f64,
}

impl StdpConfig {
    /// Membuat parameter STDP; validasi dilakukan saat aturan diaktifkan.
    #[must_use]
    pub const fn new(a_plus: f64, a_minus: f64, tau_plus_steps: f64, tau_minus_steps: f64) -> Self {
        Self {
            a_plus,
            a_minus,
            tau_plus_steps,
            tau_minus_steps,
        }
    }
    /// Memvalidasi amplitudo dan konstanta waktu.
    pub fn validate(self) -> Result<(), SimError> {
        validate_positive("a_plus", self.a_plus)?;
        validate_positive("a_minus", self.a_minus)?;
        validate_positive("tau_plus_steps", self.tau_plus_steps)?;
        validate_positive("tau_minus_steps", self.tau_minus_steps)?;
        Ok(())
    }
    /// Amplitudo potensiasi dalam satuan bobot per unit trace.
    #[must_use]
    pub const fn a_plus(self) -> f64 {
        self.a_plus
    }
    /// Amplitudo depresi dalam satuan bobot per unit trace.
    #[must_use]
    pub const fn a_minus(self) -> f64 {
        self.a_minus
    }
    /// Konstanta waktu potensiasi dalam langkah.
    #[must_use]
    pub const fn tau_plus_steps(self) -> f64 {
        self.tau_plus_steps
    }
    /// Konstanta waktu depresi dalam langkah.
    #[must_use]
    pub const fn tau_minus_steps(self) -> f64 {
        self.tau_minus_steps
    }
}

pub(super) fn validate_positive(parameter: &'static str, value: f64) -> Result<(), SimError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(SimError::InvalidConfiguration {
            parameter,
            reason: "harus berupa bilangan hingga yang lebih besar dari nol",
        });
    }
    Ok(())
}

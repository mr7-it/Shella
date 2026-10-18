use crate::SimError;
use serde::{Deserialize, Serialize};

/// Sinaps berarah dengan bobot arus bertanda dalam nanoampere.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Synapse {
    pub(super) source: usize,
    pub(super) target: usize,
    pub(super) weight_na: f64,
    pub(super) min_weight_na: f64,
    pub(super) max_weight_na: f64,
    pub(super) delay_steps: usize,
}
impl Synapse {
    /// Membuat sinaps tervalidasi; indeks neuron diperiksa saat ditambahkan ke jaringan.
    pub fn new(
        source: usize,
        target: usize,
        weight_na: f64,
        min_weight_na: f64,
        max_weight_na: f64,
        delay_steps: usize,
    ) -> Result<Self, SimError> {
        if delay_steps == 0 {
            return Err(SimError::InvalidInput {
                parameter: "delay_steps",
                reason: "harus sekurang-kurangnya satu langkah",
            });
        }
        if !weight_na.is_finite() || !min_weight_na.is_finite() || !max_weight_na.is_finite() {
            return Err(SimError::InvalidInput {
                parameter: "weight_na",
                reason: "bobot dan batasnya harus finite",
            });
        }
        if min_weight_na > max_weight_na || weight_na < min_weight_na || weight_na > max_weight_na {
            return Err(SimError::InvalidInput {
                parameter: "weight_na",
                reason: "bobot harus berada di dalam batas inklusif yang berurutan",
            });
        }
        Ok(Self {
            source,
            target,
            weight_na,
            min_weight_na,
            max_weight_na,
            delay_steps,
        })
    }
    /// Indeks neuron sumber.
    #[must_use]
    pub const fn source(&self) -> usize {
        self.source
    }
    /// Indeks neuron tujuan.
    #[must_use]
    pub const fn target(&self) -> usize {
        self.target
    }
    /// Bobot bertanda dalam nanoampere.
    #[must_use]
    pub const fn weight_na(&self) -> f64 {
        self.weight_na
    }
    /// Batas bawah bobot dalam nanoampere.
    #[must_use]
    pub const fn min_weight_na(&self) -> f64 {
        self.min_weight_na
    }
    /// Batas atas bobot dalam nanoampere.
    #[must_use]
    pub const fn max_weight_na(&self) -> f64 {
        self.max_weight_na
    }
    /// Penundaan transmisi dalam langkah simulasi.
    #[must_use]
    pub const fn delay_steps(&self) -> usize {
        self.delay_steps
    }
}

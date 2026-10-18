use crate::{config::LifConfig, error::SimError};
use serde::{Deserialize, Serialize};

/// Hasil satu transisi simulasi yang berhasil.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StepOutcome {
    /// Nomor langkah yang baru selesai, dimulai dari 1.
    pub completed_step: u64,
    /// Potensial membran setelah langkah, termasuk reset jika terjadi spike.
    pub membrane_potential_mv: f64,
    /// Bernilai benar jika ambang dilewati pada langkah ini.
    pub spiked: bool,
}

/// Keadaan satu neuron leaky integrate-and-fire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LifNeuron {
    config: LifConfig,
    membrane_potential_mv: f64,
    refractory_remaining: usize,
    completed_steps: u64,
    spike_count: u64,
    spike_steps: Vec<u64>,
}

/// Transisi yang telah dihitung tetapi belum diterapkan pada keadaan neuron.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LifStepPlan {
    pub(crate) outcome: StepOutcome,
    next_refractory_remaining: usize,
    next_spike_count: u64,
}

impl LifNeuron {
    /// Membuat neuron pada potensial istirahat.
    pub fn new(config: LifConfig) -> Result<Self, SimError> {
        config.validate()?;
        Ok(Self {
            config,
            membrane_potential_mv: config.rest_potential_mv(),
            refractory_remaining: 0,
            completed_steps: 0,
            spike_count: 0,
            spike_steps: Vec::new(),
        })
    }

    /// Membuat neuron dengan potensial awal khusus yang masih di bawah ambang.
    pub fn with_initial_potential(
        config: LifConfig,
        initial_potential_mv: f64,
    ) -> Result<Self, SimError> {
        config.validate()?;
        if !initial_potential_mv.is_finite() || initial_potential_mv >= config.threshold_mv() {
            return Err(SimError::InvalidInput {
                parameter: "initial_potential_mv",
                reason: "harus hingga dan lebih rendah daripada ambang",
            });
        }
        Ok(Self {
            config,
            membrane_potential_mv: initial_potential_mv,
            refractory_remaining: 0,
            completed_steps: 0,
            spike_count: 0,
            spike_steps: Vec::new(),
        })
    }

    /// Potensial membran saat ini dalam milivolt.
    #[must_use]
    pub const fn membrane_potential_mv(&self) -> f64 {
        self.membrane_potential_mv
    }

    /// Jumlah transisi yang berhasil diselesaikan.
    #[must_use]
    pub const fn completed_steps(&self) -> u64 {
        self.completed_steps
    }

    /// Jumlah langkah refraktori yang masih harus ditahan.
    #[must_use]
    pub const fn refractory_remaining(&self) -> usize {
        self.refractory_remaining
    }

    /// Jumlah spike yang tercatat.
    #[must_use]
    pub const fn spike_count(&self) -> u64 {
        self.spike_count
    }

    /// Indeks langkah selesai (basis satu) tempat spike terjadi.
    #[must_use]
    pub fn spike_steps(&self) -> &[u64] {
        &self.spike_steps
    }

    /// Konfigurasi tetap yang digunakan neuron.
    #[must_use]
    pub const fn config(&self) -> LifConfig {
        self.config
    }

    pub(crate) fn validate_restored_state(&self) -> Result<(), SimError> {
        self.config.validate()?;
        if !self.membrane_potential_mv.is_finite()
            || self.membrane_potential_mv >= self.config.threshold_mv()
            || self.refractory_remaining > self.config.refractory_steps()
            || self.spike_count != self.spike_steps.len() as u64
            || self
                .spike_steps
                .iter()
                .any(|step| *step == 0 || *step > self.completed_steps)
            || self.spike_steps.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(SimError::InvalidCheckpoint {
                reason: "state neuron tidak konsisten".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn prepare_advance(&mut self, current_na: f64) -> Result<LifStepPlan, SimError> {
        if !current_na.is_finite() {
            return Err(SimError::InvalidInput {
                parameter: "current_na",
                reason: "harus berupa bilangan hingga",
            });
        }
        let completed_step = self
            .completed_steps
            .checked_add(1)
            .ok_or(SimError::StepCounterOverflow)?;

        if self.refractory_remaining > 0 {
            return Ok(LifStepPlan {
                outcome: StepOutcome {
                    completed_step,
                    membrane_potential_mv: self.membrane_potential_mv,
                    spiked: false,
                },
                next_refractory_remaining: self.refractory_remaining - 1,
                next_spike_count: self.spike_count,
            });
        }

        let ratio = self.config.dt_ms() / self.config.tau_m_ms();
        let decay = (-ratio).exp();
        let integrated_fraction = -(-ratio).exp_m1();
        let voltage_from_rest = self.membrane_potential_mv - self.config.rest_potential_mv();
        let leak_component = voltage_from_rest * decay;
        let current_component = current_na * integrated_fraction;
        let driven_component = current_component * self.config.resistance_mohm();
        let leaked_voltage = self.config.rest_potential_mv() + leak_component;
        let integrated_voltage = leaked_voltage + driven_component;

        check_finite("selisih terhadap potensial istirahat", voltage_from_rest)?;
        check_finite("peluruhan tegangan", leak_component)?;
        check_finite("kontribusi arus", current_component)?;
        check_finite("konversi resistansi-arus", driven_component)?;
        check_finite("penjumlahan potensial bocor", leaked_voltage)?;
        check_finite("integrasi potensial", integrated_voltage)?;

        let spiked = integrated_voltage >= self.config.threshold_mv();
        let next_spike_count = if spiked {
            let count = self
                .spike_count
                .checked_add(1)
                .ok_or(SimError::SpikeCounterOverflow)?;
            self.spike_steps
                .try_reserve(1)
                .map_err(|_| SimError::AllocationFailed)?;
            count
        } else {
            self.spike_count
        };
        let next_potential = if spiked {
            self.config.reset_potential_mv()
        } else {
            integrated_voltage
        };
        let next_refractory = if spiked {
            self.config.refractory_steps()
        } else {
            0
        };

        Ok(LifStepPlan {
            outcome: StepOutcome {
                completed_step,
                membrane_potential_mv: next_potential,
                spiked,
            },
            next_refractory_remaining: next_refractory,
            next_spike_count,
        })
    }

    pub(crate) fn commit_advance(&mut self, plan: LifStepPlan) -> StepOutcome {
        if plan.outcome.spiked {
            self.spike_steps.push(plan.outcome.completed_step);
        }
        self.membrane_potential_mv = plan.outcome.membrane_potential_mv;
        self.refractory_remaining = plan.next_refractory_remaining;
        self.completed_steps = plan.outcome.completed_step;
        self.spike_count = plan.next_spike_count;
        plan.outcome
    }

    pub(crate) fn advance(&mut self, current_na: f64) -> Result<StepOutcome, SimError> {
        let plan = self.prepare_advance(current_na)?;
        Ok(self.commit_advance(plan))
    }
}

fn check_finite(operation: &'static str, value: f64) -> Result<(), SimError> {
    if !value.is_finite() {
        return Err(SimError::NumericalOverflow { operation });
    }
    Ok(())
}

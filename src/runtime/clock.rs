use crate::SimError;

/// Jam simulasi diskret dengan indeks langkah selesai berbasis satu.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimulationClock {
    completed_steps: u64,
}

impl SimulationClock {
    /// Membuat jam pada langkah awal nol.
    #[must_use]
    pub const fn new() -> Self {
        Self { completed_steps: 0 }
    }

    /// Indeks langkah selesai.
    #[must_use]
    pub const fn completed_steps(self) -> u64 {
        self.completed_steps
    }

    /// Menambah satu langkah secara atomik.
    pub fn tick(&mut self) -> Result<u64, SimError> {
        let next = self
            .completed_steps
            .checked_add(1)
            .ok_or(SimError::StepCounterOverflow)?;
        self.completed_steps = next;
        Ok(next)
    }

    /// Memulihkan jam dengan indeks yang disimpan.
    #[must_use]
    pub const fn from_completed_steps(completed_steps: u64) -> Self {
        Self { completed_steps }
    }
}

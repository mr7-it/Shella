use super::InputKind;
use crate::{NetworkStep, language::EncodedText};

/// Hasil satu input SDK setelah diproses oleh jaringan LIF.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    input_kind: InputKind,
    step: NetworkStep,
    encoded_text: Option<EncodedText>,
}
impl Output {
    pub(crate) const fn new(
        input_kind: InputKind,
        step: NetworkStep,
        encoded_text: Option<EncodedText>,
    ) -> Self {
        Self {
            input_kind,
            step,
            encoded_text,
        }
    }
    /// Jenis input sumber.
    #[must_use]
    pub const fn input_kind(&self) -> InputKind {
        self.input_kind
    }
    /// Hasil langkah LIF yang benar-benar dijalankan.
    #[must_use]
    pub const fn step(&self) -> &NetworkStep {
        &self.step
    }
    /// Metadata encoder bila input berupa teks.
    #[must_use]
    pub const fn encoded_text(&self) -> Option<&EncodedText> {
        self.encoded_text.as_ref()
    }
    /// Indeks neuron yang spike.
    #[must_use]
    pub fn spiking_neurons(&self) -> &[usize] {
        self.step.spiking_neurons()
    }
}

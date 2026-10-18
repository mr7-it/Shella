use super::{Vocabulary, tokenize};
use crate::SimError;

/// Representasi tekstual terstruktur untuk audit dan integrasi jaringan.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedText {
    tokens: Vec<String>,
    token_ids: Vec<u32>,
    currents_na: Vec<f64>,
}
impl EncodedText {
    /// Token hasil normalisasi dan segmentasi.
    #[must_use]
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
    /// ID vocabulary per token.
    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    /// Vektor arus hasil proyeksi hash ID token ke indeks neuron.
    #[must_use]
    pub fn currents_na(&self) -> &[f64] {
        &self.currents_na
    }
}

/// Encoder bag-of-token deterministik, bukan model semantik atau decoder bahasa.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextEncoder {
    vocabulary: Vocabulary,
}
impl TextEncoder {
    /// Membuat encoder dari vocabulary yang telah dilatih.
    pub fn new(vocabulary: Vocabulary) -> Result<Self, SimError> {
        vocabulary.validate()?;
        Ok(Self { vocabulary })
    }
    /// Referensi vocabulary encoder.
    #[must_use]
    pub const fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Encode teks dan proyeksikan hitungan token ke arus positif per neuron.
    ///
    /// Proyeksi `token_id % neuron_count` bersifat deterministik dan dapat memiliki
    /// collision; ia hanya representasi sparse sederhana, bukan embedding semantik.
    pub fn encode(
        &self,
        text: &str,
        neuron_count: usize,
        current_per_token_na: f64,
    ) -> Result<EncodedText, SimError> {
        if neuron_count == 0 {
            return Err(SimError::InvalidInput {
                parameter: "neuron_count",
                reason: "harus lebih besar dari nol",
            });
        }
        if !current_per_token_na.is_finite() || current_per_token_na <= 0.0 {
            return Err(SimError::InvalidInput {
                parameter: "current_per_token_na",
                reason: "harus finite dan positif",
            });
        }
        let tokens = tokenize(text)?;
        if tokens.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "text",
                reason: "harus memuat sekurang-kurangnya satu token",
            });
        }
        let token_ids: Vec<u32> = tokens
            .iter()
            .map(|token| self.vocabulary.id(token))
            .collect();
        let mut currents_na = vec![0.0; neuron_count];
        for id in &token_ids {
            let index = (*id as usize) % neuron_count;
            currents_na[index] += current_per_token_na;
            if !currents_na[index].is_finite() {
                return Err(SimError::NumericalOverflow {
                    operation: "akumulasi arus teks",
                });
            }
        }
        Ok(EncodedText {
            tokens,
            token_ids,
            currents_na,
        })
    }
}

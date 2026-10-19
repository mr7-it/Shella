use crate::SimError;

/// Embedding deterministik berbasis hashing token dan character n-gram.
/// Ini adalah representasi lexical yang dapat direproduksi, bukan klaim embedding semantik penuh.
#[derive(Debug, Clone, PartialEq)]
pub struct TextEmbedding {
    dimensions: usize,
    values: Vec<f32>,
}

impl TextEmbedding {
    pub fn from_text(text: &str, dimensions: usize) -> Result<Self, SimError> {
        if text.trim().is_empty() || dimensions == 0 || dimensions > 16_384 {
            return Err(SimError::InvalidInput {
                parameter: "embedding",
                reason: "teks harus non-kosong dan dimensi berada pada 1..=16384",
            });
        }
        let mut values = vec![0.0_f32; dimensions];
        let normalized = text.to_lowercase();
        for token in normalized.split_whitespace() {
            add_hashed(&mut values, token.as_bytes(), 1.0);
            let bytes = token.as_bytes();
            for window in bytes.windows(3) {
                add_hashed(&mut values, window, 0.35);
            }
        }
        let norm = values
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(SimError::InvalidInput {
                parameter: "embedding",
                reason: "teks tidak menghasilkan vektor embedding",
            });
        }
        for value in &mut values {
            *value = (*value / norm as f32).clamp(-1.0, 1.0);
        }
        Ok(Self { dimensions, values })
    }

    #[must_use]
    pub const fn dimensions(&self) -> usize {
        self.dimensions
    }

    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn cosine_similarity(&self, other: &[f32]) -> Result<f64, SimError> {
        if other.len() != self.dimensions || other.iter().any(|value| !value.is_finite()) {
            return Err(SimError::InvalidInput {
                parameter: "embedding_query",
                reason: "dimensi embedding harus sama dan semua nilai finite",
            });
        }
        let dot = self
            .values
            .iter()
            .zip(other)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum::<f64>();
        let norm = other
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            return Ok(0.0);
        }
        Ok((dot / norm).clamp(0.0, 1.0))
    }
}

fn add_hashed(values: &mut [f32], bytes: &[u8], weight: f32) {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |state, byte| {
        (state ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    let index = (hash as usize) % values.len();
    let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
    values[index] += sign * weight;
}

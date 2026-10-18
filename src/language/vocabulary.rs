use super::tokenize;
use crate::SimError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const UNKNOWN_TOKEN: &str = "<unk>";

/// Vocabulary terlatih dengan ID 0 khusus untuk token yang tidak dikenal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vocabulary {
    tokens: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
}

impl Vocabulary {
    /// Melatih vocabulary hanya dari korpus pemanggil; urutan seri leksikografis.
    pub fn fit<I, S>(
        corpus: I,
        minimum_frequency: u64,
        maximum_tokens: usize,
    ) -> Result<Self, SimError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if minimum_frequency == 0 || maximum_tokens == 0 || maximum_tokens > u32::MAX as usize {
            return Err(SimError::InvalidConfiguration {
                parameter: "vocabulary_size",
                reason: "frekuensi dan kapasitas harus positif dan dapat direpresentasikan oleh u32",
            });
        }
        let mut counts = BTreeMap::<String, u64>::new();
        for text in corpus {
            for token in tokenize(text.as_ref())? {
                let count = counts.entry(token).or_default();
                *count = count.checked_add(1).ok_or(SimError::CounterOverflow {
                    counter: "frekuensi token",
                })?;
            }
        }
        let mut entries: Vec<_> = counts
            .into_iter()
            .filter(|(_, count)| *count >= minimum_frequency)
            .collect();
        entries.sort_by(|(left_token, left_count), (right_token, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_token.cmp(right_token))
        });
        let available = maximum_tokens.saturating_sub(1);
        entries.truncate(available);
        let mut tokens = Vec::with_capacity(entries.len() + 1);
        tokens.push(UNKNOWN_TOKEN.to_owned());
        tokens.extend(entries.into_iter().map(|(token, _)| token));
        let mut token_to_id = BTreeMap::new();
        for (index, token) in tokens.iter().enumerate() {
            token_to_id.insert(
                token.clone(),
                u32::try_from(index).map_err(|_| SimError::CounterOverflow {
                    counter: "ID token",
                })?,
            );
        }
        Ok(Self {
            tokens,
            token_to_id,
        })
    }

    /// Encode token menjadi ID; token di luar vocabulary menggunakan ID 0.
    #[must_use]
    pub fn id(&self, token: &str) -> u32 {
        self.token_to_id.get(token).copied().unwrap_or(0)
    }
    /// Mendapatkan ejaan token dari ID.
    #[must_use]
    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }
    /// Jumlah token termasuk token unknown.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    /// Apakah hanya token unknown tersedia.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    /// Token berurutan ID.
    #[must_use]
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Encode teks menjadi ID per token.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, SimError> {
        Ok(tokenize(text)?.iter().map(|token| self.id(token)).collect())
    }

    /// Validasi invariant struktur saat vocabulary dipulihkan dari penyimpanan.
    pub(crate) fn validate(&self) -> Result<(), SimError> {
        if self.tokens.first().map(String::as_str) != Some(UNKNOWN_TOKEN)
            || self.token_to_id.len() != self.tokens.len()
            || self.tokens.iter().enumerate().any(|(index, token)| {
                self.token_to_id.get(token) != u32::try_from(index).ok().as_ref()
            })
        {
            return Err(SimError::InvalidInput {
                parameter: "vocabulary",
                reason: "pemetaan token dan ID tidak konsisten",
            });
        }
        Ok(())
    }
}

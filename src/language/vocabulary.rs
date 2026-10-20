use super::tokenize;
use crate::{SimError, WordID};
use hashbrown::HashMap;
use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

const UNKNOWN_TOKEN: &str = "<unk>";
const MAX_VOCABULARY_BYTES: u64 = 64 * 1024 * 1024;

/// Vocabulary terlatih dengan ID 0 khusus untuk token yang tidak dikenal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Vocabulary {
    tokens: Vec<String>,
    word_ids: Vec<WordID>,
    next_word_id: u64,
    #[serde(skip)]
    token_to_id: HashMap<String, u32>,
    #[serde(skip)]
    id_to_index: HashMap<WordID, usize>,
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
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut id_to_index = HashMap::with_capacity(tokens.len());
        let mut word_ids = Vec::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| SimError::CounterOverflow {
                counter: "ID token",
            })?;
            token_to_id.insert(token.clone(), id);
            let word_id = WordID::new(u64::from(id));
            id_to_index.insert(word_id, index);
            word_ids.push(word_id);
        }
        Ok(Self {
            next_word_id: u64::try_from(tokens.len())
                .map_err(|_| SimError::CounterOverflow { counter: "ID kata" })?,
            tokens,
            word_ids,
            token_to_id,
            id_to_index,
        })
    }

    /// Lookup token kanonis; token di luar vocabulary menggunakan ID 0.
    #[must_use]
    pub fn id(&self, token: &str) -> u32 {
        self.token_to_id.get(token).copied().unwrap_or(0)
    }

    /// ID bertipe untuk token kanonis; kata di luar vocabulary menggunakan ID 0.
    #[must_use]
    pub fn word_id(&self, token: &str) -> WordID {
        self.word_ids
            .get(self.id(token) as usize)
            .copied()
            .unwrap_or_else(|| WordID::new(0))
    }

    /// Cari satu kata setelah normalisasi; frasa dan input tanpa token tidak cocok.
    pub fn lookup_word(&self, text: &str) -> Result<Option<WordID>, SimError> {
        if text == UNKNOWN_TOKEN {
            return Ok(Some(WordID::new(0)));
        }
        let tokens = tokenize(text)?;
        if tokens.len() != 1 {
            return Ok(None);
        }
        Ok(self
            .token_to_id
            .get(&tokens[0])
            .and_then(|index| self.word_ids.get(*index as usize))
            .copied())
    }

    /// Mendapatkan ejaan kata berdasarkan ID stabil yang tersimpan.
    #[must_use]
    pub fn word(&self, id: WordID) -> Option<&str> {
        self.id_to_index
            .get(&id)
            .and_then(|index| self.tokens.get(*index))
            .map(String::as_str)
    }

    /// Daftar ID kata sesuai urutan vocabulary yang persisten.
    #[must_use]
    pub fn word_ids(&self) -> &[WordID] {
        &self.word_ids
    }

    /// Menambahkan satu kata baru dengan ID monotonik yang tidak berubah saat restart.
    /// Kata yang telah dikenal mempertahankan ID sebelumnya.
    pub fn intern_word(&mut self, text: &str) -> Result<WordID, SimError> {
        let mut tokens = tokenize(text)?;
        if tokens.len() != 1 {
            return Err(SimError::InvalidInput {
                parameter: "word",
                reason: "harus menghasilkan tepat satu token",
            });
        }
        let token = tokens.pop().ok_or(SimError::InvalidInput {
            parameter: "word",
            reason: "harus menghasilkan tepat satu token",
        })?;
        if let Some(index) = self.token_to_id.get(&token).copied() {
            return self
                .word_ids
                .get(index as usize)
                .copied()
                .ok_or(SimError::InvalidInput {
                    parameter: "vocabulary",
                    reason: "indeks kata tidak konsisten",
                });
        }
        if self.tokens.len() >= u32::MAX as usize {
            return Err(SimError::CounterOverflow {
                counter: "ID token",
            });
        }
        let next_word_id = self
            .next_word_id
            .checked_add(1)
            .ok_or(SimError::CounterOverflow { counter: "ID kata" })?;
        let token_index =
            u32::try_from(self.tokens.len()).map_err(|_| SimError::CounterOverflow {
                counter: "ID token",
            })?;
        self.tokens
            .try_reserve(1)
            .map_err(|_| SimError::AllocationFailed)?;
        self.word_ids
            .try_reserve(1)
            .map_err(|_| SimError::AllocationFailed)?;
        self.token_to_id
            .try_reserve(1)
            .map_err(|_| SimError::AllocationFailed)?;
        self.id_to_index
            .try_reserve(1)
            .map_err(|_| SimError::AllocationFailed)?;
        let word_id = WordID::new(self.next_word_id);
        self.tokens.push(token.clone());
        self.word_ids.push(word_id);
        self.id_to_index.insert(word_id, self.tokens.len() - 1);
        self.token_to_id.insert(token, token_index);
        self.next_word_id = next_word_id;
        Ok(word_id)
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

    /// Encode teks menjadi ID kata stabil setelah normalisasi dan tokenisasi.
    pub fn encode_word_ids(&self, text: &str) -> Result<Vec<WordID>, SimError> {
        Ok(tokenize(text)?
            .iter()
            .map(|token| self.word_id(token))
            .collect())
    }

    /// Decode ID stabil ke ejaan kanonis vocabulary; `WordID(0)` menjadi `<unk>`.
    ///
    /// Huruf besar, tanda baca, dan bentuk Unicode sebelum normalisasi tidak
    /// dipertahankan oleh tokenisasi; kata yang tersimpan dikembalikan apa adanya.
    pub fn decode_word_ids(&self, ids: &[WordID]) -> Result<Vec<String>, SimError> {
        ids.iter()
            .map(|id| {
                self.word(*id)
                    .map(str::to_owned)
                    .ok_or(SimError::InvalidInput {
                        parameter: "word_id",
                        reason: "ID tidak dikenal oleh vocabulary",
                    })
            })
            .collect()
    }

    /// Simpan vocabulary sebagai JSON tervalidasi melalui atomic replace.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SimError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| SimError::SerializationFailed {
            subsystem: "vocabulary",
            reason: error.to_string(),
        })?;
        if bytes.len() as u64 > MAX_VOCABULARY_BYTES {
            return Err(SimError::CorruptData {
                subsystem: "vocabulary",
                reason: "snapshot melebihi batas ukuran 64 MiB".to_owned(),
            });
        }
        crate::checkpoint::atomic_write(
            path.as_ref(),
            &bytes,
            "vocabulary",
            "simpan vocabulary atomik",
        )
    }

    /// Pulihkan vocabulary JSON dengan batas ukuran dan validasi mapping dua arah.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SimError> {
        let file = File::open(path).map_err(|error| SimError::Io {
            operation: "buka vocabulary",
            kind: error.kind(),
        })?;
        let length = file
            .metadata()
            .map_err(|error| SimError::Io {
                operation: "baca metadata vocabulary",
                kind: error.kind(),
            })?
            .len();
        if length > MAX_VOCABULARY_BYTES {
            return Err(SimError::CorruptData {
                subsystem: "vocabulary",
                reason: "snapshot melebihi batas ukuran 64 MiB".to_owned(),
            });
        }
        let mut bytes = Vec::with_capacity(length as usize);
        file.take(MAX_VOCABULARY_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| SimError::Io {
                operation: "baca vocabulary",
                kind: error.kind(),
            })?;
        if bytes.len() as u64 > MAX_VOCABULARY_BYTES {
            return Err(SimError::CorruptData {
                subsystem: "vocabulary",
                reason: "snapshot melebihi batas ukuran 64 MiB".to_owned(),
            });
        }
        serde_json::from_slice(&bytes).map_err(|error| SimError::CorruptData {
            subsystem: "vocabulary",
            reason: error.to_string(),
        })
    }

    /// Validasi invariant struktur saat vocabulary dipulihkan dari penyimpanan.
    pub(crate) fn validate(&self) -> Result<(), SimError> {
        if self.tokens.first().map(String::as_str) != Some(UNKNOWN_TOKEN)
            || self.token_to_id.len() != self.tokens.len()
            || self.id_to_index.len() != self.word_ids.len()
            || self.word_ids.len() != self.tokens.len()
            || self
                .word_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.word_ids.len()
            || self.word_ids.first().copied() != Some(WordID::new(0))
            || self.next_word_id == 0
            || self
                .word_ids
                .iter()
                .any(|word_id| word_id.get() >= self.next_word_id)
            || self.tokens.iter().enumerate().any(|(index, token)| {
                self.token_to_id.get(token) != u32::try_from(index).ok().as_ref()
                    || self.id_to_index.get(&self.word_ids[index]) != Some(&index)
            })
            || self.tokens.iter().skip(1).any(|token| {
                tokenize(token).map_or(true, |normalized| {
                    normalized.len() != 1 || normalized.first() != Some(token)
                })
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

impl<'de> Deserialize<'de> for Vocabulary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct StoredVocabulary {
            tokens: Vec<String>,
            #[serde(default)]
            word_ids: Vec<WordID>,
            #[serde(default)]
            next_word_id: u64,
            #[serde(default)]
            token_to_id: BTreeMap<String, u32>,
        }

        let stored = StoredVocabulary::deserialize(deserializer)?;
        let word_ids = if stored.word_ids.is_empty() {
            stored
                .tokens
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    u64::try_from(index)
                        .map(WordID::new)
                        .map_err(|_| D::Error::custom("jumlah kata melampaui ID yang didukung"))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            stored.word_ids
        };
        let next_word_id = if stored.next_word_id == 0 {
            word_ids
                .iter()
                .map(|id| id.get())
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| D::Error::custom("ID kata meluap saat migrasi"))?
        } else {
            stored.next_word_id
        };
        let mut token_to_id = HashMap::with_capacity(stored.tokens.len());
        let mut id_to_index = HashMap::with_capacity(stored.tokens.len());
        for (index, token) in stored.tokens.iter().enumerate() {
            let id = u32::try_from(index)
                .map_err(|_| D::Error::custom("jumlah kata melampaui ID vocabulary"))?;
            if token_to_id.insert(token.clone(), id).is_some() {
                return Err(D::Error::custom("token vocabulary duplikat"));
            }
            if id_to_index.insert(word_ids[index], index).is_some() {
                return Err(D::Error::custom("WordID vocabulary duplikat"));
            }
        }
        if !stored.token_to_id.is_empty()
            && (stored.token_to_id.len() != token_to_id.len()
                || stored.token_to_id.iter().any(|(token, id)| {
                    token_to_id
                        .get(token)
                        .is_none_or(|stored_id| stored_id != id)
                }))
        {
            return Err(D::Error::custom(
                "pemetaan token legacy tidak konsisten dengan urutan vocabulary",
            ));
        }
        let vocabulary = Self {
            tokens: stored.tokens,
            word_ids,
            next_word_id,
            token_to_id,
            id_to_index,
        };
        vocabulary
            .validate()
            .map_err(|error| D::Error::custom(error.to_string()))?;
        Ok(vocabulary)
    }
}

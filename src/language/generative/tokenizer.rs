use super::{GenerativeError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use unicode_normalization::UnicodeNormalization;

pub const PAD_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
const FIRST_CHARACTER_ID: u32 = 4;
const MAX_CHARACTER_VOCABULARY: usize = 4096;

/// Tokenisasi karakter untuk LM; ruang ID terpisah dari Vocabulary/WordID leksikal.
/// Semua code point, spasi, angka, dan tanda baca dipertahankan setelah NFC/lowercase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CharacterTokenizer {
    characters: Vec<char>,
}

impl CharacterTokenizer {
    /// Membangun vocabulary hanya dari teks training yang sudah lolos lisensi dan split.
    pub fn fit<'a>(training_texts: impl IntoIterator<Item = &'a str>) -> Result<Self> {
        let mut characters = BTreeSet::new();
        for text in training_texts {
            for character in normalize_generation_text(text)?.chars() {
                characters.insert(character);
            }
        }
        if characters.is_empty() {
            return Err(GenerativeError::InvalidInput {
                parameter: "training_texts",
                reason: "korpus training tidak berisi karakter setelah normalisasi",
            });
        }
        if characters.len() > MAX_CHARACTER_VOCABULARY {
            return Err(GenerativeError::InvalidInput {
                parameter: "training_texts",
                reason: "vocabulary karakter melebihi batas 4096",
            });
        }
        Ok(Self {
            characters: characters.into_iter().collect(),
        })
    }

    pub fn vocabulary_size(&self) -> usize {
        FIRST_CHARACTER_ID as usize + self.characters.len()
    }

    pub fn characters(&self) -> &[char] {
        &self.characters
    }

    /// Mengodekan satu dokumen dengan BOS/EOS untuk teacher forcing.
    pub fn encode_document(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = self.encode_characters(text)?;
        ids.insert(0, BOS_ID);
        ids.push(EOS_ID);
        Ok(ids)
    }

    /// Mengodekan prompt dengan BOS; tidak menambahkan EOS agar model dapat melanjutkan.
    pub fn encode_prompt(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = self.encode_characters(text)?;
        ids.insert(0, BOS_ID);
        Ok(ids)
    }

    /// OOV dipetakan ke UNK. Segmentasi karakter mempertahankan bentuk kata langka/imbuhan,
    /// tetapi tidak berarti model memahami karakter atau kata yang belum dipelajari.
    pub fn encode_characters(&self, text: &str) -> Result<Vec<u32>> {
        let normalized = normalize_generation_text(text)?;
        Ok(normalized
            .chars()
            .map(|character| {
                self.characters
                    .binary_search(&character)
                    .map_or(UNK_ID, |index| FIRST_CHARACTER_ID + index as u32)
            })
            .collect())
    }

    /// Mengubah ID kembali menjadi teks; EOS menghentikan decoding dan special token lain
    /// tidak ditampilkan selain UNK yang tampil sebagai U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut output = String::new();
        for &id in ids {
            match id {
                PAD_ID | BOS_ID => {}
                EOS_ID => break,
                UNK_ID => output.push('\u{fffd}'),
                value if value >= FIRST_CHARACTER_ID => {
                    let index = (value - FIRST_CHARACTER_ID) as usize;
                    let character =
                        self.characters
                            .get(index)
                            .ok_or(GenerativeError::InvalidInput {
                                parameter: "token_ids",
                                reason: "ID karakter di luar vocabulary tokenizer",
                            })?;
                    output.push(*character);
                }
                _ => {
                    return Err(GenerativeError::InvalidInput {
                        parameter: "token_ids",
                        reason: "ID token khusus tidak dikenal",
                    });
                }
            }
        }
        Ok(output)
    }

    pub fn validate(&self) -> Result<()> {
        if self.characters.is_empty()
            || self.characters.len() > MAX_CHARACTER_VOCABULARY
            || self.characters.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "tokenizer",
                reason: "tabel karakter kosong, duplikat, tidak terurut, atau terlalu besar",
            });
        }
        if self.characters.iter().copied().any(char::is_control) {
            return Err(GenerativeError::InvalidInput {
                parameter: "tokenizer",
                reason: "karakter kontrol tidak diizinkan dalam vocabulary",
            });
        }
        Ok(())
    }
}

pub(crate) fn normalize_generation_text(text: &str) -> Result<String> {
    if text.len() > 1_000_000 {
        return Err(GenerativeError::InvalidInput {
            parameter: "text",
            reason: "teks melebihi batas 1 MiB",
        });
    }
    let normalized = text.nfc().collect::<String>().to_lowercase();
    let mut output = String::with_capacity(normalized.len());
    let mut previous_was_space = true;
    for character in normalized.chars() {
        if character.is_whitespace() {
            if !previous_was_space {
                output.push(' ');
                previous_was_space = true;
            }
        } else if !character.is_control() {
            output.push(character);
            previous_was_space = false;
        }
    }
    if output.ends_with(' ') {
        output.pop();
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_indonesian_punctuation_numbers_and_unicode() {
        let tokenizer = CharacterTokenizer::fit(["Saya membeli 2 buku—baru!"]).unwrap();
        let encoded = tokenizer
            .encode_characters("Saya membeli 2 buku—baru!")
            .unwrap();
        assert_eq!(
            tokenizer.decode(&encoded).unwrap(),
            "saya membeli 2 buku—baru!"
        );
        assert_eq!(tokenizer.encode_characters("q").unwrap()[0], UNK_ID);
    }

    #[test]
    fn document_and_prompt_boundaries_differ() {
        let tokenizer = CharacterTokenizer::fit(["halo"]).unwrap();
        assert_eq!(
            tokenizer.encode_document("halo").unwrap().first(),
            Some(&BOS_ID)
        );
        assert_eq!(
            tokenizer.encode_document("halo").unwrap().last(),
            Some(&EOS_ID)
        );
        let prompt = tokenizer.encode_prompt("halo").unwrap();
        let characters = tokenizer.encode_characters("halo").unwrap();
        assert_eq!(prompt.last(), characters.last());
    }

    #[test]
    fn invalid_token_ids_and_empty_tokenizers_are_rejected() {
        assert!(CharacterTokenizer::fit(std::iter::empty()).is_err());
        let tokenizer = CharacterTokenizer::fit(["angka 123!"]).unwrap();
        assert!(tokenizer.decode(&[u32::MAX]).is_err());
        assert!(tokenizer.validate().is_ok());
    }
}

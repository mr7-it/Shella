use crate::{LifNetwork, NeuronID, SimError, SubwordID, Synapse, SynapseID, WordID};
use hashbrown::HashMap;
use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};
use std::collections::{BTreeMap, BTreeSet};

use super::{Vocabulary, tokenize};

const LANGUAGE_NETWORK_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SubwordRecord {
    id: SubwordID,
    text: String,
}

/// Relasi kata dan sinaps LIF aktual yang menghubungkan dua representasi kata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalConnection {
    source_word: WordID,
    target_word: WordID,
    source_neuron: NeuronID,
    target_neuron: NeuronID,
    synapse_id: SynapseID,
}

impl LexicalConnection {
    #[must_use]
    pub const fn source_word(&self) -> WordID {
        self.source_word
    }

    #[must_use]
    pub const fn target_word(&self) -> WordID {
        self.target_word
    }

    #[must_use]
    pub const fn source_neuron(&self) -> NeuronID {
        self.source_neuron
    }

    #[must_use]
    pub const fn target_neuron(&self) -> NeuronID {
        self.target_neuron
    }

    #[must_use]
    pub const fn synapse_id(&self) -> SynapseID {
        self.synapse_id
    }
}

/// Vocabulary ber-ID persisten yang diproyeksikan ke populasi LIF aktual.
///
/// Sebuah kata dapat memakai banyak neuron; arus token dibagi rata pada neuron
/// representasinya. Sinaps lexical adalah sinaps jaringan sebenarnya, sehingga
/// propagasi LIF dan STDP tetap memakai implementasi jaringan yang sama.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LanguageNetwork {
    version: u32,
    network: LifNetwork,
    vocabulary: Vocabulary,
    subwords: Vec<SubwordRecord>,
    word_subwords: BTreeMap<WordID, Vec<SubwordID>>,
    word_neurons: BTreeMap<WordID, Vec<NeuronID>>,
    connections: Vec<LexicalConnection>,
    next_subword_id: u64,
}

impl<'de> Deserialize<'de> for LanguageNetwork {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct StoredLanguageNetwork {
            version: u32,
            network: LifNetwork,
            vocabulary: Vocabulary,
            subwords: Vec<SubwordRecord>,
            word_subwords: BTreeMap<WordID, Vec<SubwordID>>,
            word_neurons: BTreeMap<WordID, Vec<NeuronID>>,
            connections: Vec<LexicalConnection>,
            next_subword_id: u64,
        }

        let stored = StoredLanguageNetwork::deserialize(deserializer)?;
        let mut state = Self {
            version: stored.version,
            network: stored.network,
            vocabulary: stored.vocabulary,
            subwords: stored.subwords,
            word_subwords: stored.word_subwords,
            word_neurons: stored.word_neurons,
            connections: stored.connections,
            next_subword_id: stored.next_subword_id,
        };
        state
            .network
            .migrate_restored_identity_state()
            .map_err(D::Error::custom)?;
        state.validate().map_err(D::Error::custom)?;
        Ok(state)
    }
}

impl LanguageNetwork {
    pub fn new(mut network: LifNetwork, vocabulary: Vocabulary) -> Result<Self, SimError> {
        vocabulary.validate()?;
        network.migrate_restored_identity_state()?;
        let mut subword_lookup = HashMap::new();
        let mut subwords = Vec::new();
        let mut word_subwords = BTreeMap::new();
        let mut next_subword_id = 1_u64;

        for (index, token) in vocabulary.tokens().iter().enumerate().skip(1) {
            let word_id = vocabulary.word_id(token);
            let mut token_subwords = Vec::new();
            for character in token.chars() {
                let character = character.to_string();
                let id = if let Some(id) = subword_lookup.get(&character) {
                    *id
                } else {
                    let id = SubwordID::new(next_subword_id);
                    next_subword_id =
                        next_subword_id
                            .checked_add(1)
                            .ok_or(SimError::CounterOverflow {
                                counter: "ID subword",
                            })?;
                    subwords
                        .try_reserve(1)
                        .map_err(|_| SimError::AllocationFailed)?;
                    subword_lookup.insert(character.clone(), id);
                    subwords.push(SubwordRecord {
                        id,
                        text: character,
                    });
                    id
                };
                token_subwords.push(id);
            }
            if index > 0 {
                word_subwords.insert(word_id, token_subwords);
            }
        }

        let language_network = Self {
            version: LANGUAGE_NETWORK_VERSION,
            network,
            vocabulary,
            subwords,
            word_subwords,
            word_neurons: BTreeMap::new(),
            connections: Vec::new(),
            next_subword_id,
        };
        language_network.validate()?;
        Ok(language_network)
    }

    #[must_use]
    pub const fn network(&self) -> &LifNetwork {
        &self.network
    }

    /// Mutasi jaringan tetap melewati validasi, scheduler, LIF, dan STDP yang sama.
    pub fn network_mut(&mut self) -> &mut LifNetwork {
        &mut self.network
    }

    #[must_use]
    pub const fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    #[must_use]
    pub fn subword(&self, id: SubwordID) -> Option<&str> {
        self.subwords
            .iter()
            .find(|record| record.id == id)
            .map(|record| record.text.as_str())
    }

    #[must_use]
    pub fn word_subwords(&self, id: WordID) -> Option<&[SubwordID]> {
        self.word_subwords.get(&id).map(Vec::as_slice)
    }

    /// Menambahkan kata baru tanpa mengubah ID kata atau subword yang telah ada.
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
        if self.vocabulary.id(&token) != 0 {
            return Ok(self.vocabulary.word_id(&token));
        }

        let mut ids = Vec::new();
        ids.try_reserve_exact(token.chars().count())
            .map_err(|_| SimError::AllocationFailed)?;
        let mut additions = Vec::new();
        additions
            .try_reserve(token.chars().count())
            .map_err(|_| SimError::AllocationFailed)?;
        let mut next_subword_id = self.next_subword_id;
        for character in token.chars() {
            let character = character.to_string();
            let existing = self
                .subwords
                .iter()
                .chain(&additions)
                .find(|entry| entry.text == character);
            let id = if let Some(record) = existing {
                record.id
            } else {
                let id = SubwordID::new(next_subword_id);
                next_subword_id =
                    next_subword_id
                        .checked_add(1)
                        .ok_or(SimError::CounterOverflow {
                            counter: "ID subword",
                        })?;
                additions.push(SubwordRecord {
                    id,
                    text: character,
                });
                id
            };
            ids.push(id);
        }
        self.subwords
            .try_reserve(additions.len())
            .map_err(|_| SimError::AllocationFailed)?;
        let word_id = self.vocabulary.intern_word(&token)?;
        self.subwords.extend(additions);
        self.next_subword_id = next_subword_id;
        self.word_subwords.insert(word_id, ids);
        Ok(word_id)
    }

    /// Bind kata ke satu atau lebih neuron; ID neuron harus berasal dari jaringannya.
    pub fn bind_word(&mut self, word_id: WordID, neurons: &[NeuronID]) -> Result<(), SimError> {
        if self.vocabulary.word(word_id).is_none() || neurons.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "word_neurons",
                reason: "kata harus dikenal dan memiliki sekurang-kurangnya satu neuron",
            });
        }
        let mut unique = BTreeSet::new();
        if neurons
            .iter()
            .any(|neuron| self.network.neuron_index(*neuron).is_none() || !unique.insert(*neuron))
        {
            return Err(SimError::InvalidInput {
                parameter: "word_neurons",
                reason: "ID neuron harus valid dan tidak boleh duplikat",
            });
        }
        if self.connections.iter().any(|connection| {
            connection.source_word == word_id || connection.target_word == word_id
        }) {
            return Err(SimError::InvalidInput {
                parameter: "word_neurons",
                reason: "representasi kata dengan koneksi synapse tidak dapat diganti",
            });
        }
        self.word_neurons.insert(word_id, neurons.to_vec());
        Ok(())
    }

    #[must_use]
    pub fn neurons_for_word(&self, word_id: WordID) -> Option<&[NeuronID]> {
        self.word_neurons.get(&word_id).map(Vec::as_slice)
    }

    /// Membuat koneksi all-to-all antar representasi terdistribusi dua kata.
    /// Operasi bersifat atomik: jaringan berubah hanya bila semua sinaps valid.
    pub fn connect_words(
        &mut self,
        source_word: WordID,
        target_word: WordID,
        weight_na: f64,
        min_weight_na: f64,
        max_weight_na: f64,
        delay_steps: usize,
    ) -> Result<Vec<SynapseID>, SimError> {
        let source_neurons = self
            .word_neurons
            .get(&source_word)
            .ok_or(SimError::InvalidInput {
                parameter: "source_word",
                reason: "kata sumber belum memiliki representasi neuron",
            })?;
        let target_neurons = self
            .word_neurons
            .get(&target_word)
            .ok_or(SimError::InvalidInput {
                parameter: "target_word",
                reason: "kata tujuan belum memiliki representasi neuron",
            })?;
        let count = source_neurons
            .len()
            .checked_mul(target_neurons.len())
            .ok_or(SimError::AllocationFailed)?;
        let mut next_network = self.network.clone();
        let mut additions = Vec::new();
        additions
            .try_reserve_exact(count)
            .map_err(|_| SimError::AllocationFailed)?;

        for source_neuron in source_neurons {
            for target_neuron in target_neurons {
                let source_index =
                    next_network
                        .neuron_index(*source_neuron)
                        .ok_or(SimError::InvalidTopology {
                            reason: "ID neuron sumber tidak ditemukan",
                        })?;
                let target_index =
                    next_network
                        .neuron_index(*target_neuron)
                        .ok_or(SimError::InvalidTopology {
                            reason: "ID neuron tujuan tidak ditemukan",
                        })?;
                next_network.add_synapse(Synapse::new(
                    source_index,
                    target_index,
                    weight_na,
                    min_weight_na,
                    max_weight_na,
                    delay_steps,
                )?)?;
                let synapse_id = next_network
                    .synapse_id(next_network.synapse_count() - 1)
                    .ok_or(SimError::InvalidTopology {
                        reason: "sinaps baru tidak memiliki ID stabil",
                    })?;
                additions.push(LexicalConnection {
                    source_word,
                    target_word,
                    source_neuron: *source_neuron,
                    target_neuron: *target_neuron,
                    synapse_id,
                });
            }
        }

        let mut all_connections = self.connections.clone();
        all_connections
            .try_reserve(additions.len())
            .map_err(|_| SimError::AllocationFailed)?;
        let ids = additions
            .iter()
            .map(LexicalConnection::synapse_id)
            .collect();
        all_connections.extend(additions);
        self.network = next_network;
        self.connections = all_connections;
        Ok(ids)
    }

    #[must_use]
    pub fn connections(&self) -> &[LexicalConnection] {
        &self.connections
    }

    /// Bobot sinaps aktual yang dapat berubah melalui STDP jaringan.
    #[must_use]
    pub fn connection_weight_na(&self, connection: &LexicalConnection) -> Option<f64> {
        self.network
            .synapse_by_id(connection.synapse_id)
            .map(|synapse| synapse.weight_na())
    }

    /// Proyeksikan setiap kata ke arus yang dibagi ke seluruh neuron representasinya.
    pub fn encode_currents(
        &self,
        text: &str,
        current_per_word_na: f64,
    ) -> Result<Vec<f64>, SimError> {
        if !current_per_word_na.is_finite() || current_per_word_na <= 0.0 {
            return Err(SimError::InvalidInput {
                parameter: "current_per_word_na",
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
        let mut currents = vec![0.0; self.network.neuron_count()];
        for token in tokens {
            let word_id = self.vocabulary.word_id(&token);
            let neurons = self
                .word_neurons
                .get(&word_id)
                .ok_or(SimError::InvalidInput {
                    parameter: "word_neurons",
                    reason: "kata atau token unknown belum memiliki representasi neuron",
                })?;
            let current = current_per_word_na / neurons.len() as f64;
            for neuron in neurons {
                let index =
                    self.network
                        .neuron_index(*neuron)
                        .ok_or(SimError::InvalidTopology {
                            reason: "representasi kata merujuk ID neuron yang tidak ada",
                        })?;
                currents[index] += current;
                if !currents[index].is_finite() {
                    return Err(SimError::NumericalOverflow {
                        operation: "proyeksi kata ke arus LIF",
                    });
                }
            }
        }
        Ok(currents)
    }

    /// Encode only non-stopword tokens using the registered Indonesian stopword list.
    pub fn encode_currents_without_stopwords(
        &self,
        text: &str,
        current_per_word_na: f64,
        stopwords: &super::StopwordSet,
    ) -> Result<Vec<f64>, SimError> {
        let tokens = stopwords.filter_text(text)?;
        if tokens.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "text",
                reason: "tidak ada token tersisa setelah stopword dihapus",
            });
        }
        self.encode_currents(&tokens.join(" "), current_per_word_na)
    }

    /// Menjalankan input teks melalui jaringan LIF dan konfigurasi STDP aktif.
    pub fn step_text(
        &mut self,
        text: &str,
        current_per_word_na: f64,
    ) -> Result<crate::NetworkStep, SimError> {
        let currents = self.encode_currents(text, current_per_word_na)?;
        self.network.step(&currents)
    }

    /// Remove verified stopwords, project remaining terms, and advance the LIF network.
    pub fn step_text_without_stopwords(
        &mut self,
        text: &str,
        current_per_word_na: f64,
        stopwords: &super::StopwordSet,
    ) -> Result<crate::NetworkStep, SimError> {
        let currents =
            self.encode_currents_without_stopwords(text, current_per_word_na, stopwords)?;
        self.network.step(&currents)
    }

    pub(crate) fn validate(&self) -> Result<(), SimError> {
        self.vocabulary.validate()?;
        self.network.validate_restored_state()?;
        if self.version != LANGUAGE_NETWORK_VERSION || self.next_subword_id == 0 {
            return Err(SimError::InvalidCheckpoint {
                reason: "versi atau penghitung ID language network invalid".to_owned(),
            });
        }
        let mut subword_ids = BTreeSet::new();
        let mut subword_text = BTreeSet::new();
        if self.subwords.iter().any(|subword| {
            subword.id.get() == 0
                || subword.id.get() >= self.next_subword_id
                || subword.text.chars().count() != 1
                || !subword_ids.insert(subword.id)
                || !subword_text.insert(subword.text.as_str())
        }) {
            return Err(SimError::InvalidCheckpoint {
                reason: "ID atau isi subword duplikat/tidak valid".to_owned(),
            });
        }
        for word_id in self.vocabulary.word_ids().iter().copied().skip(1) {
            let Some(subwords) = self.word_subwords.get(&word_id) else {
                return Err(SimError::InvalidCheckpoint {
                    reason: "word tidak memiliki segmentasi subword".to_owned(),
                });
            };
            let expected = self
                .vocabulary
                .word(word_id)
                .ok_or(SimError::InvalidCheckpoint {
                    reason: "ID word tidak ditemukan pada vocabulary".to_owned(),
                })?
                .chars()
                .map(|character| character.to_string())
                .collect::<Vec<_>>();
            let actual = subwords
                .iter()
                .map(|id| {
                    self.subwords
                        .iter()
                        .find(|record| record.id == *id)
                        .map(|record| record.text.as_str())
                        .ok_or(SimError::InvalidCheckpoint {
                            reason: "segmentasi word merujuk ID subword yang hilang".to_owned(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if subwords.iter().any(|id| !subword_ids.contains(id))
                || actual
                    .iter()
                    .copied()
                    .ne(expected.iter().map(String::as_str))
            {
                return Err(SimError::InvalidCheckpoint {
                    reason: "segmentasi word merujuk ID subword invalid atau tidak cocok"
                        .to_owned(),
                });
            }
        }
        if self.word_subwords.keys().any(|id| {
            self.vocabulary.word(*id).is_none() || *id == self.vocabulary.word_id("<unk>")
        }) {
            return Err(SimError::InvalidCheckpoint {
                reason: "segmentasi memuat word di luar vocabulary".to_owned(),
            });
        }
        for (word_id, neurons) in &self.word_neurons {
            if self.vocabulary.word(*word_id).is_none()
                || neurons.is_empty()
                || neurons
                    .iter()
                    .any(|id| self.network.neuron_index(*id).is_none())
                || neurons.iter().copied().collect::<BTreeSet<_>>().len() != neurons.len()
            {
                return Err(SimError::InvalidCheckpoint {
                    reason: "representasi neuron untuk word invalid".to_owned(),
                });
            }
        }
        let mut connection_ids = BTreeSet::new();
        for connection in &self.connections {
            let Some(synapse) = self.network.synapse_by_id(connection.synapse_id) else {
                return Err(SimError::InvalidCheckpoint {
                    reason: "relasi kata merujuk sinaps yang tidak ada".to_owned(),
                });
            };
            if !self
                .word_neurons
                .get(&connection.source_word)
                .is_some_and(|neurons| neurons.contains(&connection.source_neuron))
                || !self
                    .word_neurons
                    .get(&connection.target_word)
                    .is_some_and(|neurons| neurons.contains(&connection.target_neuron))
                || self.network.neuron_id(synapse.source()) != Some(connection.source_neuron)
                || self.network.neuron_id(synapse.target()) != Some(connection.target_neuron)
                || !connection_ids.insert(connection.synapse_id)
            {
                return Err(SimError::InvalidCheckpoint {
                    reason: "relasi kata/sinaps tidak konsisten".to_owned(),
                });
            }
        }
        Ok(())
    }
}

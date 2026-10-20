use super::{DatasetId, DatasetProvenance, LicenseRegistry, datasets::dataset_error};
use crate::{
    SimError, WordID,
    language::{normalize_text, tokenize},
};
use csv::{ReaderBuilder, StringRecord};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

const MAX_NUSAX_FILE_BYTES: u64 = 8 * 1024 * 1024;
const LANGUAGES: [&str; 12] = [
    "indonesian",
    "acehnese",
    "banjarese",
    "english",
    "madurese",
    "ngaju",
    "sundanese",
    "balinese",
    "buginese",
    "javanese",
    "minangkabau",
    "toba_batak",
];
const LEXICON_LANGUAGES: [&str; 11] = [
    "acehnese",
    "balinese",
    "banjarese",
    "buginese",
    "english",
    "javanese",
    "madurese",
    "minangkabau",
    "ngaju",
    "sundanese",
    "toba_batak",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NusaXSplit {
    Train,
    Validation,
    Test,
}

impl NusaXSplit {
    const fn filename(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "valid",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NusaXSentimentLabel {
    Positive,
    Neutral,
    Negative,
}

impl NusaXSentimentLabel {
    fn parse(value: &str) -> Result<Self, SimError> {
        match value {
            "positive" => Ok(Self::Positive),
            "neutral" => Ok(Self::Neutral),
            "negative" => Ok(Self::Negative),
            _ => Err(dataset_error("label sentimen NusaX tidak dikenal")),
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Positive => 0,
            Self::Neutral => 1,
            Self::Negative => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NusaXParallelSentence {
    pub id: usize,
    pub translations: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NusaXParallelCorpus {
    pub source: DatasetProvenance,
    pub split: NusaXSplit,
    pub sentences: Vec<NusaXParallelSentence>,
}

impl NusaXParallelCorpus {
    pub fn translation(
        &self,
        index: usize,
        source_language: &str,
        target_language: &str,
    ) -> Result<(&str, &str), SimError> {
        let sentence = self.sentences.get(index).ok_or(SimError::InvalidInput {
            parameter: "sentence_index",
            reason: "indeks berada di luar korpus NusaX",
        })?;
        let source = sentence
            .translations
            .get(source_language)
            .ok_or(SimError::InvalidInput {
                parameter: "source_language",
                reason: "bahasa tidak tersedia pada korpus NusaX",
            })?;
        let target = sentence
            .translations
            .get(target_language)
            .ok_or(SimError::InvalidInput {
                parameter: "target_language",
                reason: "bahasa tidak tersedia pada korpus NusaX",
            })?;
        Ok((source, target))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NusaXWordPair {
    pub indonesian: Vec<WordID>,
    pub translation: Vec<WordID>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NusaXSentimentExample {
    pub id: usize,
    pub text: String,
    pub label: NusaXSentimentLabel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NusaXSentimentCorpus {
    pub source: DatasetProvenance,
    pub language: String,
    pub split: NusaXSplit,
    pub examples: Vec<NusaXSentimentExample>,
}

impl NusaXSentimentCorpus {
    pub fn tokenize(&self, example_index: usize) -> Result<Vec<String>, SimError> {
        let example = self
            .examples
            .get(example_index)
            .ok_or(SimError::InvalidInput {
                parameter: "example_index",
                reason: "indeks berada di luar korpus sentimen NusaX",
            })?;
        tokenize(&example.text)
    }

    /// Evaluate caller-provided predictions on validation or test data.
    /// This measures predictions only; it does not train a classifier.
    pub fn evaluate_predictions(
        &self,
        predictions: &[NusaXSentimentLabel],
    ) -> Result<NusaXSentimentMetrics, SimError> {
        if self.split == NusaXSplit::Train {
            return Err(SimError::InvalidInput {
                parameter: "dataset_split",
                reason: "evaluasi sentimen menolak split train",
            });
        }
        if predictions.len() != self.examples.len() {
            return Err(SimError::InvalidInput {
                parameter: "predictions",
                reason: "jumlah prediksi harus sama dengan jumlah contoh",
            });
        }
        if predictions.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "predictions",
                reason: "evaluasi membutuhkan setidaknya satu contoh",
            });
        }
        let mut confusion = [[0; 3]; 3];
        for (example, prediction) in self.examples.iter().zip(predictions) {
            confusion[example.label.index()][prediction.index()] += 1;
        }
        let correct = (0..3).map(|index| confusion[index][index]).sum::<usize>();
        Ok(NusaXSentimentMetrics {
            samples: predictions.len(),
            accuracy: correct as f64 / predictions.len() as f64,
            confusion,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NusaXSentimentMetrics {
    pub samples: usize,
    pub accuracy: f64,
    pub confusion: [[usize; 3]; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NusaXLexiconEntry {
    pub id: usize,
    pub indonesian: String,
    pub translation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NusaXLexicon {
    pub source: DatasetProvenance,
    pub language: String,
    pub entries: Vec<NusaXLexiconEntry>,
    #[serde(skip)]
    by_indonesian: HashMap<String, Vec<usize>>,
}

impl<'de> Deserialize<'de> for NusaXLexicon {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct StoredLexicon {
            source: DatasetProvenance,
            language: String,
            entries: Vec<NusaXLexiconEntry>,
        }

        let stored = StoredLexicon::deserialize(deserializer)?;
        let mut lexicon = Self {
            source: stored.source,
            language: stored.language,
            entries: stored.entries,
            by_indonesian: HashMap::new(),
        };
        lexicon.rebuild_index().map_err(serde::de::Error::custom)?;
        Ok(lexicon)
    }
}

impl NusaXLexicon {
    pub fn lookup_indonesian(&self, term: &str) -> Result<Vec<&str>, SimError> {
        let term = normalize_text(term)?;
        Ok(self
            .by_indonesian
            .get(&term)
            .into_iter()
            .flatten()
            .filter_map(|index| self.entries.get(*index))
            .map(|entry| entry.translation.as_str())
            .collect())
    }

    /// Interns each Indonesian/local-language pair in the actual language
    /// network. It does not bind neurons, create synapses, or train a model.
    pub fn intern_terms(
        &self,
        network: &mut super::LanguageNetwork,
    ) -> Result<Vec<NusaXWordPair>, SimError> {
        let mut staged = network.clone();
        let mut word_pairs = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let indonesian = tokenize(&entry.indonesian)?
                .iter()
                .map(|token| staged.intern_word(token))
                .collect::<Result<Vec<_>, _>>()?;
            let translation = tokenize(&entry.translation)?
                .iter()
                .map(|token| staged.intern_word(token))
                .collect::<Result<Vec<_>, _>>()?;
            if indonesian.is_empty() || translation.is_empty() {
                return Err(dataset_error("entri lexicon NusaX tidak memiliki token"));
            }
            word_pairs.push(NusaXWordPair {
                indonesian,
                translation,
            });
        }
        *network = staged;
        Ok(word_pairs)
    }

    pub fn to_json(&self) -> Result<String, SimError> {
        serde_json::to_string(self).map_err(|error| SimError::SerializationFailed {
            subsystem: "NusaX lexicon",
            reason: error.to_string(),
        })
    }

    pub fn from_json(json: &str, registry: &LicenseRegistry) -> Result<Self, SimError> {
        let lexicon: Self =
            serde_json::from_str(json).map_err(|error| dataset_error(error.to_string()))?;
        let expected = registry.provenance(DatasetId::NusaX)?;
        if lexicon.source != expected || !LEXICON_LANGUAGES.contains(&lexicon.language.as_str()) {
            return Err(dataset_error(
                "metadata sumber atau bahasa lexicon NusaX tidak valid",
            ));
        }
        Ok(lexicon)
    }

    fn rebuild_index(&mut self) -> Result<(), SimError> {
        self.by_indonesian.clear();
        let mut seen_pairs = HashSet::new();
        let mut seen_ids = HashSet::new();
        for (index, entry) in self.entries.iter().enumerate() {
            let indonesian = normalize_text(&entry.indonesian)?;
            let translation = normalize_text(&entry.translation)?;
            if indonesian.is_empty()
                || translation.is_empty()
                || !seen_ids.insert(entry.id)
                || !seen_pairs.insert((indonesian.clone(), translation))
            {
                return Err(dataset_error("entri duplikat dalam lexicon NusaX"));
            }
            self.by_indonesian
                .entry(indonesian)
                .or_default()
                .push(index);
        }
        if self.entries.is_empty() {
            return Err(dataset_error("lexicon NusaX kosong"));
        }
        Ok(())
    }
}

impl super::DatasetLoader {
    /// Parse and verify the pinned multilingual NusaX machine-translation CSV.
    pub fn load_nusax_mt_csv(
        path: impl AsRef<Path>,
        split: NusaXSplit,
        registry: &LicenseRegistry,
    ) -> Result<NusaXParallelCorpus, SimError> {
        let relative_path = format!("datasets/mt/{}.csv", split.filename());
        let (bytes, source) = read_verified_nusax(path.as_ref(), &relative_path, registry)?;
        let mut reader = csv_reader(&bytes);
        let headers = reader
            .headers()
            .map_err(|error| dataset_error(format!("header MT NusaX invalid: {error}")))?
            .clone();
        if headers.len() != LANGUAGES.len() + 1
            || headers.get(0) != Some("")
            || !LANGUAGES
                .iter()
                .enumerate()
                .all(|(index, language)| headers.get(index + 1) == Some(*language))
        {
            return Err(dataset_error("skema header MT NusaX tidak cocok"));
        }
        let mut ids = HashSet::new();
        let mut sentences = Vec::new();
        for record in reader.records() {
            let record =
                record.map_err(|error| dataset_error(format!("baris MT invalid: {error}")))?;
            require_columns(&record, LANGUAGES.len() + 1)?;
            let id = parse_id(&record, 0)?;
            if !ids.insert(id) {
                return Err(dataset_error("ID duplikat pada dataset MT NusaX"));
            }
            let mut translations = BTreeMap::new();
            for (index, language) in LANGUAGES.iter().enumerate() {
                let text = required_text(&record, index + 1, "terjemahan")?;
                translations.insert((*language).to_owned(), text);
            }
            sentences.push(NusaXParallelSentence { id, translations });
        }
        ensure_nonempty(&sentences)?;
        Ok(NusaXParallelCorpus {
            source,
            split,
            sentences,
        })
    }

    /// Parse and verify NusaX sentiment examples for one language and split.
    pub fn load_nusax_sentiment_csv(
        path: impl AsRef<Path>,
        language: &str,
        split: NusaXSplit,
        registry: &LicenseRegistry,
    ) -> Result<NusaXSentimentCorpus, SimError> {
        if !LANGUAGES.contains(&language) {
            return Err(SimError::InvalidInput {
                parameter: "language",
                reason: "bahasa tidak tersedia pada dataset sentimen NusaX",
            });
        }
        let relative_path = format!("datasets/sentiment/{language}/{}.csv", split.filename());
        let (bytes, source) = read_verified_nusax(path.as_ref(), &relative_path, registry)?;
        let mut reader = csv_reader(&bytes);
        let headers = reader
            .headers()
            .map_err(|error| dataset_error(format!("header sentimen invalid: {error}")))?
            .clone();
        if headers.iter().collect::<Vec<_>>() != ["id", "text", "label"] {
            return Err(dataset_error("skema header sentimen NusaX tidak cocok"));
        }
        let mut ids = HashSet::new();
        let mut examples = Vec::new();
        for record in reader.records() {
            let record = record
                .map_err(|error| dataset_error(format!("baris sentimen invalid: {error}")))?;
            require_columns(&record, 3)?;
            let id = parse_id(&record, 0)?;
            if !ids.insert(id) {
                return Err(dataset_error("ID duplikat pada dataset sentimen NusaX"));
            }
            let text = required_text(&record, 1, "teks")?;
            if normalize_text(&text)?.is_empty() {
                return Err(dataset_error("teks sentimen kosong setelah normalisasi"));
            }
            examples.push(NusaXSentimentExample {
                id,
                text,
                label: NusaXSentimentLabel::parse(&record[2])?,
            });
        }
        ensure_nonempty(&examples)?;
        Ok(NusaXSentimentCorpus {
            source,
            language: language.to_owned(),
            split,
            examples,
        })
    }

    /// Parse and verify a pinned NusaX Indonesian-to-language lexicon CSV.
    pub fn load_nusax_lexicon_csv(
        path: impl AsRef<Path>,
        language: &str,
        registry: &LicenseRegistry,
    ) -> Result<NusaXLexicon, SimError> {
        if !LEXICON_LANGUAGES.contains(&language) {
            return Err(SimError::InvalidInput {
                parameter: "language",
                reason: "bahasa tidak tersedia pada lexicon NusaX",
            });
        }
        let relative_path = format!("datasets/lexicon/{language}.csv");
        let (bytes, source) = read_verified_nusax(path.as_ref(), &relative_path, registry)?;
        let mut reader = csv_reader(&bytes);
        let headers = reader
            .headers()
            .map_err(|error| dataset_error(format!("header lexicon invalid: {error}")))?
            .clone();
        if headers.len() != 3
            || headers.get(0) != Some("")
            || headers.get(1) != Some("indonesian")
            || headers.get(2) != Some(language)
        {
            return Err(dataset_error("skema header lexicon NusaX tidak cocok"));
        }
        let mut ids = HashSet::new();
        let mut entries = Vec::new();
        for record in reader.records() {
            let record =
                record.map_err(|error| dataset_error(format!("baris lexicon invalid: {error}")))?;
            require_columns(&record, 3)?;
            let id = parse_id(&record, 0)?;
            if !ids.insert(id) {
                return Err(dataset_error("ID duplikat pada lexicon NusaX"));
            }
            entries.push(NusaXLexiconEntry {
                id,
                indonesian: required_text(&record, 1, "kata bahasa Indonesia")?,
                translation: required_text(&record, 2, "terjemahan lexicon")?,
            });
        }
        let mut lexicon = NusaXLexicon {
            source,
            language: language.to_owned(),
            entries,
            by_indonesian: HashMap::new(),
        };
        lexicon.rebuild_index()?;
        Ok(lexicon)
    }
}

fn read_verified_nusax(
    path: &Path,
    relative_path: &str,
    registry: &LicenseRegistry,
) -> Result<(Vec<u8>, DatasetProvenance), SimError> {
    let file =
        File::open(path).map_err(|error| super::datasets::io_error("buka dataset NusaX", error))?;
    let length = file
        .metadata()
        .map_err(|error| super::datasets::io_error("baca metadata dataset NusaX", error))?
        .len();
    if length > MAX_NUSAX_FILE_BYTES {
        return Err(dataset_error("file NusaX melampaui batas 8 MiB"));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(MAX_NUSAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| super::datasets::io_error("baca dataset NusaX", error))?;
    if bytes.len() as u64 > MAX_NUSAX_FILE_BYTES {
        return Err(dataset_error("file NusaX melampaui batas 8 MiB"));
    }
    registry.verify_file(DatasetId::NusaX, relative_path, &bytes)?;
    Ok((bytes, registry.provenance(DatasetId::NusaX)?))
}

fn csv_reader(bytes: &[u8]) -> csv::Reader<&[u8]> {
    ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(bytes)
}

fn require_columns(record: &StringRecord, expected: usize) -> Result<(), SimError> {
    if record.len() == expected {
        Ok(())
    } else {
        Err(dataset_error("jumlah kolom CSV NusaX tidak cocok"))
    }
}

fn parse_id(record: &StringRecord, index: usize) -> Result<usize, SimError> {
    record
        .get(index)
        .ok_or_else(|| dataset_error("ID CSV tidak tersedia"))?
        .parse::<usize>()
        .map_err(|_| dataset_error("ID CSV NusaX harus berupa bilangan bulat non-negatif"))
}

fn required_text(
    record: &StringRecord,
    index: usize,
    field: &'static str,
) -> Result<String, SimError> {
    let value = record
        .get(index)
        .ok_or_else(|| dataset_error("kolom CSV NusaX tidak tersedia"))?;
    if value.trim().is_empty() {
        return Err(dataset_error(format!("{field} kosong dalam CSV NusaX")));
    }
    Ok(value.to_owned())
}

fn ensure_nonempty<T>(records: &[T]) -> Result<(), SimError> {
    if records.is_empty() {
        Err(dataset_error("dataset NusaX kosong"))
    } else {
        Ok(())
    }
}

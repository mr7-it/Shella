mod checkpoint;
mod dataset;
mod generation;
mod model;
mod tokenizer;
mod training;

pub use dataset::{
    CorpusManifest, CorpusRecord, CorpusSourceManifest, CorpusSplit, DatasetSplitConfig,
    SplitManifest,
};
pub use generation::{
    GeneratedText, GenerationConfig, GenerationRepetitionMetrics, GenerationStrategy,
};
pub use model::{ComputeDevice, TransformerConfig};
pub use tokenizer::{BOS_ID, CharacterTokenizer, EOS_ID, PAD_ID, UNK_ID};
pub use training::{EpochMetrics, EvaluationMetrics, TrainingConfig, TrainingReport};

use crate::{SimError, language::LicenseRegistry};
use std::{fmt, path::Path};

pub type Result<T> = std::result::Result<T, GenerativeError>;

#[derive(Debug)]
pub enum GenerativeError {
    InvalidConfig {
        parameter: &'static str,
        reason: &'static str,
    },
    InvalidInput {
        parameter: &'static str,
        reason: &'static str,
    },
    Backend {
        operation: &'static str,
        reason: String,
    },
    Dataset(SimError),
    Checkpoint(String),
    Io {
        operation: &'static str,
        kind: std::io::ErrorKind,
    },
    Serialization(String),
    Numerical {
        operation: &'static str,
    },
}

impl fmt::Display for GenerativeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { parameter, reason } => {
                write!(
                    formatter,
                    "konfigurasi generatif {parameter} tidak sah: {reason}"
                )
            }
            Self::InvalidInput { parameter, reason } => {
                write!(
                    formatter,
                    "masukan generatif {parameter} tidak sah: {reason}"
                )
            }
            Self::Backend { operation, reason } => {
                write!(formatter, "operasi neural {operation} gagal: {reason}")
            }
            Self::Dataset(error) => write!(formatter, "dataset neural tidak valid: {error}"),
            Self::Checkpoint(reason) => {
                write!(formatter, "checkpoint model generatif invalid: {reason}")
            }
            Self::Io { operation, kind } => write!(formatter, "I/O {operation} gagal: {kind}"),
            Self::Serialization(reason) => write!(formatter, "serialisasi model gagal: {reason}"),
            Self::Numerical { operation } => {
                write!(formatter, "hasil numerik tidak finite pada {operation}")
            }
        }
    }
}

impl std::error::Error for GenerativeError {}

impl From<candle_core::Error> for GenerativeError {
    fn from(error: candle_core::Error) -> Self {
        Self::Backend {
            operation: "tensor/autograd",
            reason: error.to_string(),
        }
    }
}

impl From<serde_json::Error> for GenerativeError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error.to_string())
    }
}

impl From<std::io::Error> for GenerativeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io {
            operation: "file model generatif",
            kind: error.kind(),
        }
    }
}

/// Decoder-only Transformer kecil berbasis Candle; vocabulary karakter milik model ini
/// terpisah dari WordID leksikal Shella dan tidak mengubah semantic/LearningMemory.
pub struct GenerativeLanguageModel {
    pub(crate) tokenizer: CharacterTokenizer,
    pub(crate) backend: model::CausalLanguageModel,
    pub(crate) corpus_manifest: Option<CorpusManifest>,
    pub(crate) training_report: Option<TrainingReport>,
    pub(crate) checkpoint_sha256: Option<String>,
}

impl GenerativeLanguageModel {
    pub fn new(config: TransformerConfig, tokenizer: CharacterTokenizer) -> Result<Self> {
        tokenizer.validate()?;
        let backend = model::CausalLanguageModel::new(config, tokenizer.vocabulary_size())?;
        Ok(Self {
            tokenizer,
            backend,
            corpus_manifest: None,
            training_report: None,
            checkpoint_sha256: None,
        })
    }

    /// Membuat tokenizer hanya dari train split lalu mengikat manifest sumber ke model.
    pub fn from_corpus(config: TransformerConfig, corpus: &CorpusSplit) -> Result<Self> {
        corpus.validate_disjoint_ids()?;
        let tokenizer =
            CharacterTokenizer::fit(corpus.training.iter().map(|record| record.text.as_str()))?;
        let mut model = Self::new(config, tokenizer)?;
        model.corpus_manifest = Some(corpus.manifest.clone());
        Ok(model)
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.backend.config
    }

    pub fn tokenizer(&self) -> &CharacterTokenizer {
        &self.tokenizer
    }

    pub fn corpus_manifest(&self) -> Option<&CorpusManifest> {
        self.corpus_manifest.as_ref()
    }

    pub fn training_report(&self) -> Option<&TrainingReport> {
        self.training_report.as_ref()
    }

    pub fn parameter_count(&self) -> usize {
        self.backend
            .variables()
            .iter()
            .map(|variable| variable.as_tensor().elem_count())
            .sum()
    }

    pub fn train(
        &mut self,
        corpus: &CorpusSplit,
        config: TrainingConfig,
    ) -> Result<TrainingReport> {
        let report = training::train_model(self, corpus, config)?;
        self.training_report = Some(report.clone());
        Ok(report)
    }

    pub fn evaluate(
        &self,
        records: &[CorpusRecord],
        batch_size: usize,
    ) -> Result<EvaluationMetrics> {
        training::evaluate_model(self, records, batch_size)
    }

    pub fn generate(&self, prompt: &str, config: GenerationConfig) -> Result<GeneratedText> {
        generation::generate(self, prompt, config)
    }

    /// Menyimpan bobot + arsitektur/tokenizer + provenance dan metrik training atomik.
    pub fn save_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<String> {
        let fingerprint = checkpoint::save(self, path.as_ref())?;
        self.checkpoint_sha256 = Some(fingerprint.clone());
        Ok(fingerprint)
    }

    /// Memvalidasi versi, checksum, ukuran, config, tokenizer, dan tensor saat memuat checkpoint.
    pub fn load_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        checkpoint::load(path.as_ref())
    }

    pub fn checkpoint_sha256(&self) -> Option<&str> {
        self.checkpoint_sha256.as_deref()
    }

    /// Memuat split train NusaX yang lisensinya tercatat di registry Shella.
    pub fn load_nusax_training_corpus(
        csv_path: impl AsRef<Path>,
        split_config: DatasetSplitConfig,
    ) -> Result<CorpusSplit> {
        let registry = LicenseRegistry::verified_sources();
        CorpusSplit::load_nusax_training(csv_path, &registry, split_config)
    }
}

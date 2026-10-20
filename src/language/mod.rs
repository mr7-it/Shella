//! Pipeline representasi tekstual lokal; tidak melakukan pemahaman atau generasi bahasa.
mod encoder;
mod indonesian;
mod network;
mod normalization;
mod tokenizer;
mod vocabulary;

pub use encoder::{EncodedText, TextEncoder};
pub use indonesian::{
    DatasetId, DatasetLoadReport, DatasetLoader, DatasetProvenance, HunspellAffixKind,
    HunspellAffixRule, HunspellAnalysis, HunspellRoot, IndoSumArticle, IndoSumCorpus,
    IndoSumPartition, IndoSumRouge1, IndonesianVocabulary, LexicalAnalysis, LexicalRelation,
    LexicalRelationType, LexiconEntry, LexiconEntryInput, LexiconMetadata, LicenseRegistry,
    LinguisticMetadata, NusaXLexicon, NusaXLexiconEntry, NusaXParallelCorpus,
    NusaXParallelSentence, NusaXSentimentCorpus, NusaXSentimentExample, NusaXSentimentLabel,
    NusaXSentimentMetrics, NusaXSplit, NusaXWordPair, SourceProvenance, StopwordSet,
};
pub use network::{LanguageNetwork, LexicalConnection};
pub use normalization::normalize_text;
pub use tokenizer::tokenize;
pub use vocabulary::Vocabulary;

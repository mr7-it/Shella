//! Pipeline bahasa lokal dengan analisis semantik dan respons konservatif berbukti.
mod encoder;
mod indonesian;
mod network;
mod normalization;
mod semantic;
mod tokenizer;
mod vocabulary;

pub use encoder::{EncodedText, TextEncoder};
pub use indonesian::{
    DatasetEncoding, DatasetFileFormat, DatasetFileIntegrity, DatasetId, DatasetIntegrityManifest,
    DatasetIntegrityReport, DatasetLoadReport, DatasetLoader, DatasetProvenance, HunspellAffixKind,
    HunspellAffixRule, HunspellAnalysis, HunspellRoot, IndoSumArticle, IndoSumCorpus,
    IndoSumPartition, IndoSumRouge1, IndonesianVocabulary, LexicalAnalysis, LexicalRelation,
    LexicalRelationType, LexiconEntry, LexiconEntryInput, LexiconMetadata, LicenseRegistry,
    LinguisticMetadata, NusaXLexicon, NusaXLexiconEntry, NusaXParallelCorpus,
    NusaXParallelSentence, NusaXSentimentCorpus, NusaXSentimentExample, NusaXSentimentLabel,
    NusaXSentimentMetrics, NusaXSplit, NusaXWordPair, SourceProvenance, StopwordSet,
    WordNetBahasaDataset, WordNetBahasaDefinition, WordNetBahasaImportReport, WordNetBahasaMember,
};
pub use network::{LanguageNetwork, LexicalConnection};
pub use normalization::normalize_text;
pub use semantic::{
    AssertionStatus, ConceptRelation, ContextResolution, EvidenceLevel, GroupedSemanticExample,
    InferenceTrace, PhraseEntry, PhraseKind, PhraseMatch, SemanticAnalysis, SemanticAnswerEvidence,
    SemanticConfidenceBand, SemanticEncoding, SemanticEvaluationReport, SemanticEvidence,
    SemanticExample, SemanticGapReport, SemanticMemory, SemanticQuestionIntent, SemanticRecall,
    SemanticRelationKind, SemanticResonance, SemanticResponse, SemanticRole, SemanticRoleKind,
    SemanticSplitMetrics, SemanticToken, SentenceStructure, TokenAmbiguity, TokenSpan,
};
pub use tokenizer::tokenize;
pub use vocabulary::Vocabulary;

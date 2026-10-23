//! Shella: jaringan LIF dan SDK Rust modular untuk aplikasi neuromorfik.
//!
//! Crate menyediakan runtime/neuron LIF, jaringan sinkron, STDP reward-modulated,
//! evaluator dan benchmark, checkpoint tervalidasi, memori file-backed, decision
//! scoring deterministik, preprocessing teks, analisis semantik, respons berbukti,
//! serta SDK input/output. Analisis/intent menggunakan aturan eksplisit; confidence
//! semantic bersifat ordinal, bukan probabilistik atau pemahaman bahasa umum.

pub mod benchmark;
pub mod checkpoint;
pub mod config;
pub mod decision;
pub mod engine;
pub mod error;
pub mod evaluator;
pub mod identity;
pub mod knowledge;
pub mod language;
pub mod learning;
pub mod memory;
pub mod network;
pub mod neuron;
pub mod reasoning;
pub mod runtime;
pub mod sdk;

pub use benchmark::{
    ConditioningBenchmarkConfig, ConditioningBenchmarkResult, ContinualBenchmarkConfig,
    ContinualBenchmarkResult, ContinualReplayBenchmarkConfig, ContinualReplayBenchmarkResult,
    ContinualReplayNoiseResult, ContinualReplayRunResult, ContinualReplayTaskMetrics,
    ContinualReplayValidation, ContinualSeedResult, GeneralizationBenchmarkConfig,
    GeneralizationBenchmarkResult, GeneralizationNoiseResult, GeneralizationRunResult,
    InterferenceBenchmarkConfig, InterferenceBenchmarkResult, InterferenceSeedResult, IrisSample,
    LongTermLearningBenchmarkConfig, LongTermLearningBenchmarkResult, LongTermLearningNoiseResult,
    LongTermLearningRunResult, LongTermLearningTaskMetrics, SelfEvaluation, StratifiedFold,
    StratifiedSplit, TrainScaler, evaluate_predictions, noise, stratified_folds, stratified_split,
};
pub use checkpoint::{
    Checkpoint, CheckpointFingerprints, RestoredIntegratedState, RestoredKnowledgeState,
    RestoredState,
};
pub use config::LifConfig;
pub use decision::{
    AbstentionReason, CalibrationEvaluator, CalibrationMetrics, Candidate, Decision,
    DecisionEvaluator, DecisionMetrics, DecisionPolicy, FeatureContribution, IsotonicCalibrator,
    PlattCalibrator,
};
pub use engine::{
    LearningOutcome, MemoryAttention, ReasoningConfig, ReasoningEngine, ReasoningResult,
    ReasoningStep, SemanticLearningOutcome,
};
pub use error::SimError;
pub use evaluator::{RewardEvaluation, RewardEvaluator, RewardEvaluatorConfig, RewardMetrics};
pub use identity::{NeuronID, SubwordID, SynapseID, WordID};
pub use knowledge::{
    CalibrationParameters, ClassKnowledge, ClassKnowledgeInference, ClassKnowledgeModel,
    ClassPrediction, Exemplar, ExemplarPolicy, ExemplarReason, FeatureDefinition, FeatureMetadata,
    FeatureScaler, FeatureStatistics, FeatureType, IrisFeatures, LegacyPrototypeSnapshot,
    ModelApprovalStatus, Prototype, PrototypeCandidateScore, SCORING_VERSION, feature_definitions,
};
pub use language::generative::{
    CharacterTokenizer, ComputeDevice, CorpusManifest, CorpusRecord, CorpusSourceManifest,
    CorpusSplit, DatasetSplitConfig, EpochMetrics, EvaluationMetrics, FeedbackAuthority,
    FeedbackAuthorityCounts, GeneratedText, GenerationConfig, GenerationRepetitionMetrics,
    GenerationStrategy, GenerativeError, GenerativeLanguageModel, HybridAction,
    HybridActionPrediction, HybridLearningOutcome, HybridTrainingConfig, HybridTrainingReport,
    SplitManifest, TrainingConfig, TrainingPhase, TrainingProgress, TrainingReport,
    TransformerConfig, VerifiedActionExample,
};
pub use language::{
    AssertionStatus, ConceptRelation, ContextResolution, DatasetEncoding, DatasetFileFormat,
    DatasetFileIntegrity, DatasetId, DatasetIntegrityManifest, DatasetIntegrityReport,
    DatasetLoadReport, DatasetLoader, DatasetProvenance, EncodedText, EvidenceLevel,
    GroupedSemanticExample, HunspellAffixKind, HunspellAffixRule, HunspellAnalysis, HunspellRoot,
    IndoSumArticle, IndoSumCorpus, IndoSumPartition, IndoSumRouge1, IndonesianVocabulary,
    InferenceTrace, LanguageNetwork, LexicalAnalysis, LexicalConnection, LexicalRelation,
    LexicalRelationType, LexiconEntry, LexiconEntryInput, LexiconMetadata, LicenseRegistry,
    LinguisticMetadata, NusaXLexicon, NusaXLexiconEntry, NusaXParallelCorpus,
    NusaXParallelSentence, NusaXSentimentCorpus, NusaXSentimentExample, NusaXSentimentLabel,
    NusaXSentimentMetrics, NusaXSplit, NusaXWordPair, PhraseEntry, PhraseKind, PhraseMatch,
    SemanticAnalysis, SemanticAnswerEvidence, SemanticConfidenceBand, SemanticEncoding,
    SemanticEvaluationReport, SemanticEvidence, SemanticExample, SemanticGapReport, SemanticMemory,
    SemanticQuestionIntent, SemanticRecall, SemanticRelationKind, SemanticResonance,
    SemanticResponse, SemanticRole, SemanticRoleKind, SemanticSplitMetrics, SemanticToken,
    SentenceStructure, SourceProvenance, StopwordSet, TextEncoder, TokenAmbiguity, TokenSpan,
    Vocabulary, normalize_text, tokenize,
};
pub use learning::{
    ConsolidationReport, ForgettingReport, LearningMemory, MemoryConflict, MemoryFeedbackWrite,
    MemoryInference, MemoryRecall, ReplayReport, RewardModulatedStdpConfig, StdpConfig,
};
pub use memory::{MemoryRecord, MemoryStore, RecordId, SecureMemoryStore, TextEmbedding};
pub use network::{LifNetwork, NetworkStep, Synapse};
pub use neuron::lif::{LifNeuron, StepOutcome};
pub use reasoning::{
    CalibrationSample, CalibrationState, CausalAssociation, ContextFrame, EvidenceContradiction,
    ReasoningCheckpoint, UncertaintyDecomposition,
};
pub use runtime::{LifRuntime, SimulationClock};
pub use sdk::{
    AnswerAbstentionReason, AnswerActionMap, AnswerDraftReview, DraftVerification, Engine,
    HybridLearningError, Input as SdkInput, InputKind as SdkInputKind, LifAnswerTrace,
    MemoryConditionedGeneration, MemoryGenerationError, MemoryGenerationStatus,
    Output as SdkOutput, Session as SdkSession, VerifiedAnswer, VerifiedAnswerError,
    VerifiedAnswerStatus,
};

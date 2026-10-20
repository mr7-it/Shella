use super::{LanguageNetwork, Vocabulary, tokenize};
use crate::{LifNetwork, SimError, WordID};
use hashbrown::HashMap;
use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};
use std::collections::BTreeSet;

mod datasets;
mod hunspell;
mod indosum;
mod integrity;
mod licenses;
mod nusax;
mod relations;

pub use datasets::{DatasetLoadReport, DatasetLoader, StopwordSet};
pub use hunspell::{HunspellAffixKind, HunspellAffixRule, HunspellAnalysis, HunspellRoot};
pub use indosum::{IndoSumArticle, IndoSumCorpus, IndoSumPartition, IndoSumRouge1};
pub use integrity::{
    DatasetEncoding, DatasetFileFormat, DatasetFileIntegrity, DatasetIntegrityManifest,
    DatasetIntegrityReport,
};
pub use licenses::{DatasetId, DatasetProvenance, LicenseRegistry};
pub use nusax::{
    NusaXLexicon, NusaXLexiconEntry, NusaXParallelCorpus, NusaXParallelSentence,
    NusaXSentimentCorpus, NusaXSentimentExample, NusaXSentimentLabel, NusaXSentimentMetrics,
    NusaXSplit, NusaXWordPair,
};
pub use relations::{LexicalAnalysis, LexicalRelation, LexicalRelationType};

const LEXICON_SCHEMA_VERSION: u32 = 1;

/// Provenance for one entry. The library records the supplied license information;
/// it does not independently certify the source's licensing claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceProvenance {
    pub source_name: String,
    pub source_uri: String,
    pub license: String,
    pub license_uri: String,
    pub source_entry_id: String,
}

/// Dataset-level metadata; entry-level provenance remains attached to each word.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexiconMetadata {
    pub name: String,
    pub version: String,
    pub language_tag: String,
}

impl LexiconMetadata {
    #[must_use]
    pub fn indonesian(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            language_tag: "id".to_owned(),
        }
    }
}

/// Linguistic assertions remain optional and can be tied to a cited source entry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct LinguisticMetadata {
    pub part_of_speech: Option<String>,
    pub lemma: Option<String>,
    pub affixes: Vec<String>,
    pub spelling_variants: Vec<String>,
    pub definition: Option<String>,
    pub definition_source_entry_id: Option<String>,
}

/// Entry persisted with its stable `WordID` and recorded provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexiconEntry {
    pub word_id: WordID,
    pub headword: String,
    pub sources: Vec<SourceProvenance>,
    pub linguistic: LinguisticMetadata,
}

/// Input accepted by validated entry insertion and JSON batch import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexiconEntryInput {
    pub headword: String,
    pub sources: Vec<SourceProvenance>,
    #[serde(default)]
    pub linguistic: LinguisticMetadata,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LexiconImportDocument {
    schema_version: u32,
    metadata: LexiconMetadata,
    entries: Vec<LexiconEntryInput>,
}

/// Indonesian lexicon index. It ships without a word dataset: callers add entries
/// only with explicit per-entry source and license provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndonesianVocabulary {
    schema_version: u32,
    metadata: LexiconMetadata,
    vocabulary: Vocabulary,
    entries: Vec<LexiconEntry>,
    #[serde(default)]
    datasets: Vec<DatasetProvenance>,
    #[serde(default)]
    analyses: Vec<LexicalAnalysis>,
    #[serde(default)]
    relations: Vec<LexicalRelation>,
    #[serde(default)]
    hunspell_roots: Vec<HunspellRoot>,
    #[serde(default)]
    hunspell_rules: Vec<HunspellAffixRule>,
    #[serde(skip)]
    aliases: HashMap<String, WordID>,
    #[serde(skip)]
    word_index: HashMap<WordID, usize>,
    #[serde(skip)]
    lemma_index: HashMap<String, Vec<WordID>>,
}

impl<'de> Deserialize<'de> for IndonesianVocabulary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct StoredLexicon {
            schema_version: u32,
            metadata: LexiconMetadata,
            vocabulary: Vocabulary,
            entries: Vec<LexiconEntry>,
            #[serde(default)]
            datasets: Vec<DatasetProvenance>,
            #[serde(default)]
            analyses: Vec<LexicalAnalysis>,
            #[serde(default)]
            relations: Vec<LexicalRelation>,
            #[serde(default)]
            hunspell_roots: Vec<HunspellRoot>,
            #[serde(default)]
            hunspell_rules: Vec<HunspellAffixRule>,
        }

        let stored = StoredLexicon::deserialize(deserializer)?;
        let mut lexicon = Self {
            schema_version: stored.schema_version,
            metadata: stored.metadata,
            vocabulary: stored.vocabulary,
            entries: stored.entries,
            datasets: stored.datasets,
            analyses: stored.analyses,
            relations: stored.relations,
            hunspell_roots: stored.hunspell_roots,
            hunspell_rules: stored.hunspell_rules,
            aliases: HashMap::new(),
            word_index: HashMap::new(),
            lemma_index: HashMap::new(),
        };
        lexicon.rebuild_aliases().map_err(D::Error::custom)?;
        lexicon.validate().map_err(D::Error::custom)?;
        Ok(lexicon)
    }
}

impl IndonesianVocabulary {
    pub fn new(metadata: LexiconMetadata) -> Result<Self, SimError> {
        validate_metadata(&metadata)?;
        Ok(Self {
            schema_version: LEXICON_SCHEMA_VERSION,
            metadata,
            vocabulary: Vocabulary::fit(std::iter::empty::<&str>(), 1, 1)?,
            entries: Vec::new(),
            datasets: Vec::new(),
            analyses: Vec::new(),
            relations: Vec::new(),
            hunspell_roots: Vec::new(),
            hunspell_rules: Vec::new(),
            aliases: HashMap::new(),
            word_index: HashMap::new(),
            lemma_index: HashMap::new(),
        })
    }

    #[must_use]
    pub const fn metadata(&self) -> &LexiconMetadata {
        &self.metadata
    }

    #[must_use]
    pub fn entries(&self) -> &[LexiconEntry] {
        &self.entries
    }

    #[must_use]
    pub fn datasets(&self) -> &[DatasetProvenance] {
        &self.datasets
    }

    #[must_use]
    pub fn analyses(&self) -> &[LexicalAnalysis] {
        &self.analyses
    }

    #[must_use]
    pub fn relations(&self) -> &[LexicalRelation] {
        &self.relations
    }

    #[must_use]
    pub fn hunspell_roots(&self) -> &[HunspellRoot] {
        &self.hunspell_roots
    }

    #[must_use]
    pub fn hunspell_rules(&self) -> &[HunspellAffixRule] {
        &self.hunspell_rules
    }

    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[must_use]
    pub const fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Returns the registered entry for a normalized headword or spelling alias.
    pub fn lookup(&self, text: &str) -> Result<Option<&LexiconEntry>, SimError> {
        let term = normalize_lexeme(text)?;
        let Some(word_id) = self.aliases.get(&term) else {
            return Ok(None);
        };
        Ok(self
            .word_index
            .get(word_id)
            .and_then(|index| self.entries.get(*index)))
    }

    /// Look up inflected forms linked to a normalized lemma using the lemma index.
    pub fn lookup_lemma(&self, lemma: &str) -> Result<Vec<&LexiconEntry>, SimError> {
        let lemma = normalize_lexeme(lemma)?;
        Ok(self
            .lemma_index
            .get(&lemma)
            .into_iter()
            .flatten()
            .filter_map(|word_id| self.word_index.get(word_id))
            .filter_map(|index| self.entries.get(*index))
            .collect())
    }

    /// Resolve a stable word ID without scanning the lexicon.
    #[must_use]
    pub fn by_word_id(&self, id: WordID) -> Option<&LexiconEntry> {
        self.word_index
            .get(&id)
            .and_then(|index| self.entries.get(*index))
    }

    /// Attach a source-verified morphology analysis without overwriting other
    /// contextual analyses for the same word.
    pub fn add_analysis(
        &mut self,
        analysis: LexicalAnalysis,
        registry: &LicenseRegistry,
    ) -> Result<(), SimError> {
        registry.verify(&analysis.dataset)?;
        if self.by_word_id(analysis.word_id).is_none() {
            return Err(invalid_lexicon(
                "analisis merujuk WordID yang tidak tersedia",
            ));
        }
        normalize_lexeme(&analysis.lemma)?;
        if !analysis.features.keys().all(|key| !invalid_text_field(key))
            || !analysis
                .features
                .values()
                .all(|value| !invalid_text_field(value))
            || invalid_text_field(&analysis.upos)
        {
            return Err(SimError::InvalidInput {
                parameter: "morphology",
                reason: "UPOS dan fitur morfologi harus bernilai valid",
            });
        }
        if !self.datasets.contains(&analysis.dataset) {
            self.datasets.push(analysis.dataset.clone());
        }
        if !self.analyses.iter().any(|existing| existing == &analysis) {
            self.analyses.push(analysis.clone());
        }
        self.rebuild_indexes()?;
        self.validate()
    }

    /// Add a typed lexical relation with verified dataset provenance.
    pub fn add_relation(
        &mut self,
        relation: LexicalRelation,
        registry: &LicenseRegistry,
    ) -> Result<(), SimError> {
        registry.verify(&relation.dataset)?;
        let is_dependency = matches!(relation.relation_type, LexicalRelationType::Dependency(_));
        if !is_dependency && !registry.supports_semantic_relations(&relation.dataset)? {
            return Err(SimError::InvalidInput {
                parameter: "semantic_relation",
                reason: "sumber terverifikasi tidak menyediakan relasi semantik",
            });
        }
        if (!is_dependency
            && relation
                .source_word
                .is_none_or(|source| source == relation.target_word))
            || relation
                .source_word
                .is_some_and(|source| self.by_word_id(source).is_none())
            || self.by_word_id(relation.target_word).is_none()
            || invalid_text_field(&relation.source_entry_id)
        {
            return Err(SimError::InvalidInput {
                parameter: "lexical_relation",
                reason: "relasi memerlukan dua kata berbeda dan sumber yang valid",
            });
        }
        if !self.datasets.contains(&relation.dataset) {
            self.datasets.push(relation.dataset.clone());
        }
        if !self.relations.iter().any(|existing| existing == &relation) {
            self.relations.push(relation);
        }
        self.validate()
    }

    #[must_use]
    pub fn relations_from(&self, word_id: WordID) -> Vec<&LexicalRelation> {
        self.relations
            .iter()
            .filter(|relation| relation.source_word == Some(word_id))
            .collect()
    }

    #[must_use]
    pub fn relations_to(&self, word_id: WordID) -> Vec<&LexicalRelation> {
        self.relations
            .iter()
            .filter(|relation| relation.target_word == word_id)
            .collect()
    }

    /// Add or update an entry while retaining its existing `WordID`.
    ///
    /// An update replaces linguistic assertions and adds newly supplied source
    /// records without deleting previously recorded provenance.
    pub fn upsert(&mut self, input: LexiconEntryInput) -> Result<WordID, SimError> {
        let headword = normalize_lexeme(&input.headword)?;
        validate_input(&input, &headword)?;

        let mut staged = self.clone();
        let (word_id, headword) = if let Some(existing) = staged.aliases.get(&headword).copied() {
            let canonical = staged
                .vocabulary
                .word(existing)
                .ok_or_else(|| invalid_lexicon("alias merujuk WordID yang tidak dikenal"))?
                .to_owned();
            (existing, canonical)
        } else {
            (staged.vocabulary.intern_word(&headword)?, headword)
        };

        let sources =
            if let Some(existing) = staged.entries.iter().find(|entry| entry.word_id == word_id) {
                merge_sources(&existing.sources, input.sources)?
            } else {
                input.sources
            };

        let entry = LexiconEntry {
            word_id,
            headword,
            sources,
            linguistic: input.linguistic,
        };
        if let Some(existing) = staged
            .entries
            .iter_mut()
            .find(|existing| existing.word_id == word_id)
        {
            *existing = entry;
        } else {
            staged.entries.push(entry);
        }
        staged.rebuild_aliases()?;
        staged.validate()?;
        *self = staged;
        Ok(word_id)
    }

    pub(super) fn upsert_dataset_entry(
        &mut self,
        input: LexiconEntryInput,
    ) -> Result<WordID, SimError> {
        let headword = normalize_lexeme(&input.headword)?;
        validate_input(&input, &headword)?;
        if let Some(word_id) = self.aliases.get(&headword).copied() {
            let canonical = self
                .vocabulary
                .word(word_id)
                .ok_or_else(|| invalid_lexicon("alias merujuk WordID yang tidak dikenal"))?
                .to_owned();
            let index = *self
                .word_index
                .get(&word_id)
                .ok_or_else(|| invalid_lexicon("WordID dataset tidak memiliki indeks"))?;
            let existing = self
                .entries
                .get_mut(index)
                .ok_or_else(|| invalid_lexicon("indeks entry dataset berada di luar lexicon"))?;
            existing.sources = merge_sources(&existing.sources, input.sources)?;
            existing.linguistic = input.linguistic;
            self.aliases.insert(canonical, word_id);
            return Ok(word_id);
        }

        let word_id = self.vocabulary.intern_word(&headword)?;
        let index = self.entries.len();
        self.entries.push(LexiconEntry {
            word_id,
            headword: headword.clone(),
            sources: input.sources,
            linguistic: input.linguistic,
        });
        self.aliases.insert(headword, word_id);
        self.word_index.insert(word_id, index);
        Ok(word_id)
    }

    /// Import a versioned JSON batch atomically. Failed batches leave this library
    /// and all existing IDs unchanged.
    pub fn import_json(&mut self, json: &str) -> Result<(), SimError> {
        let document: LexiconImportDocument =
            serde_json::from_str(json).map_err(|error| SimError::CorruptData {
                subsystem: "Indonesian lexicon import",
                reason: error.to_string(),
            })?;
        if document.schema_version != LEXICON_SCHEMA_VERSION {
            return Err(SimError::UnsupportedDataVersion {
                subsystem: "Indonesian lexicon",
                version: document.schema_version,
            });
        }
        validate_metadata(&document.metadata)?;
        if document.metadata != self.metadata {
            return Err(invalid_lexicon(
                "metadata dataset tidak cocok dengan library",
            ));
        }

        let mut staged = self.clone();
        for entry in document.entries {
            staged.upsert(entry)?;
        }
        *self = staged;
        Ok(())
    }

    /// Serialize the validated persistent JSON representation.
    pub fn to_json(&self) -> Result<String, SimError> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| SimError::SerializationFailed {
            subsystem: "Indonesian lexicon",
            reason: error.to_string(),
        })
    }

    /// Restore a lexicon JSON representation, rebuilding and validating indexes.
    pub fn from_json(json: &str) -> Result<Self, SimError> {
        serde_json::from_str(json).map_err(|error| SimError::CorruptData {
            subsystem: "Indonesian lexicon",
            reason: error.to_string(),
        })
    }

    /// Segment an unregistered term into normalized Unicode characters. This is
    /// orthographic segmentation only and does not infer its meaning or morphology.
    pub fn segment_unlisted(&self, text: &str) -> Result<Vec<String>, SimError> {
        let term = normalize_lexeme(text)?;
        Ok(term
            .chars()
            .map(|character| character.to_string())
            .collect())
    }

    /// Create a neural language network using the lexicon's persisted word IDs.
    pub fn to_language_network(&self, network: LifNetwork) -> Result<LanguageNetwork, SimError> {
        LanguageNetwork::new(network, self.vocabulary.clone())
    }

    pub(crate) fn validate(&self) -> Result<(), SimError> {
        if self.schema_version != LEXICON_SCHEMA_VERSION {
            return Err(SimError::UnsupportedDataVersion {
                subsystem: "Indonesian lexicon",
                version: self.schema_version,
            });
        }
        validate_metadata(&self.metadata)?;
        self.vocabulary.validate()?;
        if self.entries.len() + 1 != self.vocabulary.len() {
            return Err(invalid_lexicon(
                "jumlah entri tidak cocok dengan ID vocabulary",
            ));
        }
        let mut ids = BTreeSet::new();
        for entry in &self.entries {
            let normalized = normalize_lexeme(&entry.headword)?;
            if normalized != entry.headword
                || entry.word_id.get() == 0
                || !ids.insert(entry.word_id)
                || self.vocabulary.word(entry.word_id) != Some(entry.headword.as_str())
            {
                return Err(invalid_lexicon(
                    "headword atau WordID tidak konsisten dengan vocabulary",
                ));
            }
            validate_sources(&entry.sources)?;
            validate_linguistic(&entry.linguistic, &entry.sources)?;
        }
        let rebuilt = build_aliases(&self.entries)?;
        if rebuilt != self.aliases {
            return Err(invalid_lexicon("indeks alias tidak konsisten"));
        }
        let unique_dataset_ids = self
            .datasets
            .iter()
            .map(|dataset| dataset.id)
            .collect::<BTreeSet<_>>();
        if unique_dataset_ids.len() != self.datasets.len() {
            return Err(invalid_lexicon("provenance dataset duplikat"));
        }
        let registry = LicenseRegistry::verified_sources();
        for dataset in &self.datasets {
            registry.verify(dataset)?;
        }
        for analysis in &self.analyses {
            if self.by_word_id(analysis.word_id).is_none()
                || !self.datasets.contains(&analysis.dataset)
                || normalize_lexeme(&analysis.lemma).is_err()
                || invalid_text_field(&analysis.upos)
            {
                return Err(invalid_lexicon("analisis morfologi tidak konsisten"));
            }
        }
        for relation in &self.relations {
            if relation
                .source_word
                .is_some_and(|source| self.by_word_id(source).is_none())
                || self.by_word_id(relation.target_word).is_none()
                || !self.datasets.contains(&relation.dataset)
                || validate_sources(std::slice::from_ref(&relation.source)).is_err()
            {
                return Err(invalid_lexicon("relasi leksikal tidak konsisten"));
            }
        }
        for root in &self.hunspell_roots {
            if self.by_word_id(root.word_id).is_none()
                || !self.datasets.contains(&root.dataset)
                || root.flags.iter().any(String::is_empty)
            {
                return Err(invalid_lexicon("Hunspell root tidak konsisten"));
            }
        }
        for rule in &self.hunspell_rules {
            if !self.datasets.contains(&rule.dataset)
                || rule.flag.is_empty()
                || rule.condition.is_empty()
            {
                return Err(invalid_lexicon("aturan afiks Hunspell tidak konsisten"));
            }
        }
        Ok(())
    }

    fn rebuild_aliases(&mut self) -> Result<(), SimError> {
        self.aliases = build_aliases(&self.entries)?;
        self.rebuild_indexes()?;
        Ok(())
    }

    fn rebuild_indexes(&mut self) -> Result<(), SimError> {
        self.word_index.clear();
        self.lemma_index.clear();
        for (index, entry) in self.entries.iter().enumerate() {
            self.word_index.insert(entry.word_id, index);
        }
        for analysis in &self.analyses {
            let lemma = normalize_lexeme(&analysis.lemma)?;
            let forms = self.lemma_index.entry(lemma).or_default();
            if !forms.contains(&analysis.word_id) {
                forms.push(analysis.word_id);
            }
        }
        for forms in self.lemma_index.values_mut() {
            forms.sort_unstable();
        }
        Ok(())
    }

    pub(super) fn add_dataset_provenance(
        &mut self,
        provenance: DatasetProvenance,
        registry: &LicenseRegistry,
    ) -> Result<(), SimError> {
        registry.verify(&provenance)?;
        if !self.datasets.contains(&provenance) {
            self.datasets.push(provenance);
        }
        Ok(())
    }
}

fn validate_metadata(metadata: &LexiconMetadata) -> Result<(), SimError> {
    if metadata.name.trim().is_empty()
        || metadata.version.trim().is_empty()
        || metadata.language_tag != "id"
        || metadata.name.chars().any(char::is_control)
        || metadata.version.chars().any(char::is_control)
    {
        return Err(SimError::InvalidInput {
            parameter: "lexicon_metadata",
            reason: "nama, versi, dan language_tag=id yang valid wajib diisi",
        });
    }
    Ok(())
}

fn validate_input(input: &LexiconEntryInput, normalized_headword: &str) -> Result<(), SimError> {
    if input.headword.chars().any(char::is_control) || normalized_headword.is_empty() {
        return Err(SimError::InvalidInput {
            parameter: "headword",
            reason: "harus memuat satu istilah tanpa karakter kontrol",
        });
    }
    validate_sources(&input.sources)?;
    validate_linguistic(&input.linguistic, &input.sources)
}

fn validate_sources(sources: &[SourceProvenance]) -> Result<(), SimError> {
    let mut source_ids = BTreeSet::new();
    if sources.is_empty()
        || sources.iter().any(|source| {
            source.source_name.trim().is_empty()
                || source.source_entry_id.trim().is_empty()
                || source.license.trim().is_empty()
                || !valid_http_uri(&source.source_uri)
                || !valid_http_uri(&source.license_uri)
                || source.source_name.chars().any(char::is_control)
                || source.source_entry_id.chars().any(char::is_control)
                || source.license.chars().any(char::is_control)
                || !source_ids.insert((source.source_uri.as_str(), source.source_entry_id.as_str()))
        })
    {
        return Err(SimError::InvalidInput {
            parameter: "source_provenance",
            reason: "setiap entri wajib memiliki sumber, ID asal, lisensi, dan URI HTTP(S) valid",
        });
    }
    Ok(())
}

fn merge_sources(
    existing: &[SourceProvenance],
    incoming: Vec<SourceProvenance>,
) -> Result<Vec<SourceProvenance>, SimError> {
    let mut merged = existing.to_vec();
    for source in incoming {
        if let Some(known) = merged.iter().find(|known| {
            known.source_uri == source.source_uri && known.source_entry_id == source.source_entry_id
        }) {
            if known != &source {
                return Err(SimError::InvalidInput {
                    parameter: "source_provenance",
                    reason: "metadata lisensi untuk sumber dan ID asal yang sama berkonflik",
                });
            }
        } else {
            merged.push(source);
        }
    }
    Ok(merged)
}

fn validate_linguistic(
    linguistic: &LinguisticMetadata,
    sources: &[SourceProvenance],
) -> Result<(), SimError> {
    if linguistic
        .part_of_speech
        .as_ref()
        .is_some_and(|value| invalid_text_field(value))
        || linguistic
            .lemma
            .as_ref()
            .is_some_and(|value| invalid_text_field(value))
        || linguistic
            .affixes
            .iter()
            .any(|value| invalid_text_field(value))
        || linguistic
            .spelling_variants
            .iter()
            .any(|value| invalid_text_field(value))
        || linguistic
            .definition
            .as_ref()
            .is_some_and(|value| invalid_text_field(value))
    {
        return Err(SimError::InvalidInput {
            parameter: "linguistic_metadata",
            reason: "metadata linguistik tidak boleh kosong atau memuat karakter kontrol",
        });
    }
    match (
        linguistic.definition.as_ref(),
        linguistic.definition_source_entry_id.as_ref(),
    ) {
        (Some(_), Some(source_id))
            if sources
                .iter()
                .any(|source| source.source_entry_id == *source_id) =>
        {
            Ok(())
        }
        (None, None) => Ok(()),
        _ => Err(SimError::InvalidInput {
            parameter: "definition_source_entry_id",
            reason: "definisi wajib dikaitkan dengan ID sumber yang tercatat pada entri",
        }),
    }
}

fn invalid_text_field(value: &str) -> bool {
    value.trim().is_empty() || value.chars().any(char::is_control)
}

fn valid_http_uri(uri: &str) -> bool {
    let Some((scheme, remainder)) = uri.split_once("://") else {
        return false;
    };
    if !matches!(scheme, "http" | "https")
        || remainder.is_empty()
        || uri.chars().any(char::is_whitespace)
    {
        return false;
    }
    let authority = remainder.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') {
        return false;
    }
    let host = if let Some(bracketed) = authority.strip_prefix('[') {
        let Some((address, suffix)) = bracketed.split_once(']') else {
            return false;
        };
        if address.is_empty()
            || !address.chars().all(|character| {
                character.is_ascii_hexdigit() || character == ':' || character == '.'
            })
            || (!suffix.is_empty() && !suffix.strip_prefix(':').is_some_and(valid_port))
        {
            return false;
        }
        return valid_percent_escapes(uri);
    } else if let Some((host, port)) = authority.rsplit_once(':') {
        if !valid_port(port) {
            return false;
        }
        host
    } else {
        authority
    };
    !host.is_empty()
        && !host.starts_with('.')
        && !host.ends_with('.')
        && host
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '-'))
        && valid_percent_escapes(uri)
}

fn valid_port(port: &str) -> bool {
    !port.is_empty() && port.parse::<u16>().is_ok()
}

fn valid_percent_escapes(uri: &str) -> bool {
    let bytes = uri.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len()
                || !bytes[index + 1].is_ascii_hexdigit()
                || !bytes[index + 2].is_ascii_hexdigit()
            {
                return false;
            }
            index += 3;
        } else {
            index += 1;
        }
    }
    true
}

fn normalize_lexeme(text: &str) -> Result<String, SimError> {
    let tokens = tokenize(text)?;
    if tokens.len() != 1 {
        return Err(SimError::InvalidInput {
            parameter: "headword",
            reason: "harus dinormalisasi menjadi tepat satu token",
        });
    }
    tokens.into_iter().next().ok_or(SimError::InvalidInput {
        parameter: "headword",
        reason: "tidak boleh kosong",
    })
}

fn build_aliases(entries: &[LexiconEntry]) -> Result<HashMap<String, WordID>, SimError> {
    let mut aliases = HashMap::new();
    for entry in entries {
        let headword = normalize_lexeme(&entry.headword)?;
        insert_alias(&mut aliases, headword, entry.word_id)?;
        for variant in &entry.linguistic.spelling_variants {
            insert_alias(&mut aliases, normalize_lexeme(variant)?, entry.word_id)?;
        }
    }
    Ok(aliases)
}

fn insert_alias(
    aliases: &mut HashMap<String, WordID>,
    alias: String,
    word_id: WordID,
) -> Result<(), SimError> {
    if aliases
        .insert(alias, word_id)
        .is_some_and(|previous| previous != word_id)
    {
        return Err(invalid_lexicon(
            "headword atau variasi ejaan bertabrakan dengan entri lain",
        ));
    }
    Ok(())
}

fn invalid_lexicon(reason: impl Into<String>) -> SimError {
    SimError::CorruptData {
        subsystem: "Indonesian lexicon",
        reason: reason.into(),
    }
}

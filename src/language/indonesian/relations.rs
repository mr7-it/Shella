use crate::WordID;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::{DatasetProvenance, SourceProvenance};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalAnalysis {
    pub word_id: WordID,
    pub lemma: String,
    pub upos: String,
    pub features: BTreeMap<String, String>,
    pub dataset: DatasetProvenance,
    pub source_entry_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum LexicalRelationType {
    Synonym,
    Antonym,
    Hypernym,
    Hyponym,
    Meronym,
    Holonym,
    Dependency(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalRelation {
    pub source_word: Option<WordID>,
    pub target_word: WordID,
    pub relation_type: LexicalRelationType,
    pub dataset: DatasetProvenance,
    pub source_entry_id: String,
    pub source: SourceProvenance,
}

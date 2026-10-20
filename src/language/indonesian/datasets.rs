use super::{
    DatasetId, DatasetProvenance, IndonesianVocabulary, LexicalAnalysis, LexicalRelation,
    LexicalRelationType, LexiconEntryInput, LicenseRegistry, LinguisticMetadata, normalize_lexeme,
};
use crate::SimError;
use crate::language::normalize_text;
use hashbrown::{HashMap, HashSet};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

const MAX_DATASET_BYTES: u64 = 256 * 1024 * 1024;
type StopwordPhrases = HashMap<String, Vec<Vec<String>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DatasetLoadReport {
    pub sentences: usize,
    pub tokens: usize,
    pub entries_added: usize,
    pub analyses_added: usize,
    pub dependencies_added: usize,
    pub skipped_forms: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopwordSet {
    provenance: DatasetProvenance,
    terms: HashSet<String>,
    phrases: HashMap<String, Vec<Vec<String>>>,
}

impl StopwordSet {
    #[must_use]
    pub const fn provenance(&self) -> &DatasetProvenance {
        &self.provenance
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn contains(&self, term: &str) -> Result<bool, SimError> {
        let normalized = normalize_text(term)?;
        Ok(self.terms.contains(&normalized))
    }

    pub fn filter_text(&self, text: &str) -> Result<Vec<String>, SimError> {
        let tokens = super::tokenize(text)?;
        let mut filtered = Vec::with_capacity(tokens.len());
        let mut index = 0;
        while index < tokens.len() {
            let phrase = self.phrases.get(&tokens[index]).and_then(|phrases| {
                phrases
                    .iter()
                    .find(|phrase| tokens[index..].starts_with(phrase))
            });
            if let Some(phrase) = phrase {
                index += phrase.len();
            } else {
                filtered.push(tokens[index].clone());
                index += 1;
            }
        }
        Ok(filtered)
    }
}

#[derive(Debug)]
struct ConlluToken {
    id: usize,
    form: String,
    lemma: String,
    upos: String,
    features: BTreeMap<String, String>,
    head: usize,
    dependency: String,
    line_number: usize,
}

struct UdImportContext<'a> {
    sentence_id: &'a str,
    file_name: &'a str,
    source_id: DatasetId,
    dataset: &'a DatasetProvenance,
    registry: &'a LicenseRegistry,
}

/// Parsers for local files at explicitly pinned, license-registered revisions.
pub struct DatasetLoader;

impl DatasetLoader {
    /// Load UD CoNLL-U token, lemma, UPOS, FEATS, and dependency annotations.
    ///
    /// Multiword-token and empty-node rows are skipped, as required for lexical
    /// word IDs. The entire import is staged and committed only after validation.
    pub fn load_ud_conllu(
        path: impl AsRef<Path>,
        source_id: DatasetId,
        lexicon: &mut IndonesianVocabulary,
        registry: &LicenseRegistry,
    ) -> Result<DatasetLoadReport, SimError> {
        if !matches!(
            source_id,
            DatasetId::UdIndonesianGsd | DatasetId::UdIndonesianCsui | DatasetId::UdIndonesianPud
        ) {
            return Err(SimError::InvalidInput {
                parameter: "dataset_id",
                reason: "UD CoNLL-U loader hanya menerima sumber UD yang terdaftar",
            });
        }
        let source = registry.provenance(source_id)?;
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| io_error("buka dataset CoNLL-U", error))?;
        let length = file
            .metadata()
            .map_err(|error| io_error("baca metadata dataset", error))?
            .len();
        if length > MAX_DATASET_BYTES {
            return Err(dataset_error("dataset melampaui batas 256 MiB"));
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| dataset_error("nama file dataset tidak valid"))?;
        let mut contents = Vec::with_capacity(length as usize);
        file.take(MAX_DATASET_BYTES + 1)
            .read_to_end(&mut contents)
            .map_err(|error| io_error("baca dataset CoNLL-U", error))?;
        if contents.len() as u64 > MAX_DATASET_BYTES {
            return Err(dataset_error("dataset melampaui batas 256 MiB"));
        }
        registry.verify_file(source_id, file_name, &contents)?;
        std::str::from_utf8(&contents)
            .map_err(|_| dataset_error("dataset CoNLL-U bukan UTF-8 valid"))?;
        let mut staged = lexicon.clone();
        staged.add_dataset_provenance(source.clone(), registry)?;
        let initial_entries = staged.entries().len();
        let mut report = DatasetLoadReport::default();
        let mut sentence = Vec::new();
        let mut sentence_id = String::new();
        let mut reader = BufReader::new(contents.as_slice());
        let mut line = String::new();
        let mut line_number = 0;
        loop {
            line.clear();
            let read = reader
                .read_line(&mut line)
                .map_err(|error| io_error("baca dataset CoNLL-U", error))?;
            if read == 0 {
                break;
            }
            line_number += 1;
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                if !sentence.is_empty() {
                    import_sentence(
                        &mut staged,
                        &sentence,
                        UdImportContext {
                            sentence_id: &sentence_id,
                            file_name,
                            source_id,
                            dataset: &source,
                            registry,
                        },
                        &mut report,
                    )?;
                    report.sentences += 1;
                    sentence.clear();
                    sentence_id.clear();
                }
                continue;
            }
            if let Some(comment) = trimmed.strip_prefix("# sent_id = ") {
                sentence_id = comment.trim().to_owned();
                continue;
            }
            if trimmed.starts_with('#') {
                continue;
            }
            let fields = trimmed.split('\t').collect::<Vec<_>>();
            if fields.len() != 10 {
                return Err(dataset_error(format!(
                    "baris {line_number} tidak memiliki 10 kolom CoNLL-U"
                )));
            }
            if fields[0].contains('-') || fields[0].contains('.') {
                continue;
            }
            let token_id = parse_positive(fields[0], line_number, "ID token")?;
            let head = fields[6]
                .parse::<usize>()
                .map_err(|_| dataset_error(format!("HEAD tidak valid pada baris {line_number}")))?;
            if fields[1].is_empty()
                || fields[2].is_empty()
                || fields[3].is_empty()
                || fields[7].is_empty()
            {
                return Err(dataset_error(format!(
                    "FORM, LEMMA, UPOS, atau DEPREL kosong pada baris {line_number}"
                )));
            }
            let features = parse_features(fields[5], line_number)?;
            sentence.push(ConlluToken {
                id: token_id,
                form: fields[1].to_owned(),
                lemma: fields[2].to_owned(),
                upos: fields[3].to_owned(),
                features,
                head,
                dependency: fields[7].to_owned(),
                line_number,
            });
        }
        if !sentence.is_empty() {
            import_sentence(
                &mut staged,
                &sentence,
                UdImportContext {
                    sentence_id: &sentence_id,
                    file_name,
                    source_id,
                    dataset: &source,
                    registry,
                },
                &mut report,
            )?;
            report.sentences += 1;
        }
        if report.tokens == 0 {
            return Err(dataset_error("dataset tidak memuat token leksikal"));
        }
        staged.rebuild_aliases()?;
        staged.validate()?;
        report.entries_added = staged.entries().len().saturating_sub(initial_entries);
        *lexicon = staged;
        Ok(report)
    }

    /// Load stopwords-id's pinned UTF-8 JSON array from the caller's local file.
    pub fn load_stopwords_json(
        path: impl AsRef<Path>,
        registry: &LicenseRegistry,
    ) -> Result<StopwordSet, SimError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| io_error("buka stopword JSON", error))?;
        let metadata = file
            .metadata()
            .map_err(|error| io_error("baca metadata stopword", error))?;
        if metadata.len() > 8 * 1024 * 1024 {
            return Err(dataset_error("stopword JSON melampaui batas 8 MiB"));
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| io_error("baca stopword JSON", error))?;
        if bytes.len() > 8 * 1024 * 1024 {
            return Err(dataset_error("stopword JSON melampaui batas 8 MiB"));
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| dataset_error("nama file stopword tidak valid"))?;
        registry.verify_file(DatasetId::StopwordsIsoId, file_name, &bytes)?;
        let json = std::str::from_utf8(&bytes)
            .map_err(|_| dataset_error("stopword JSON bukan UTF-8 valid"))?;
        let provenance = registry.provenance(DatasetId::StopwordsIsoId)?;
        let (terms, phrases) = parse_stopwords_terms(json)?;
        Ok(StopwordSet {
            provenance,
            terms,
            phrases,
        })
    }
}

fn parse_stopwords_terms(json: &str) -> Result<(HashSet<String>, StopwordPhrases), SimError> {
    let terms = serde_json::from_str::<Vec<String>>(json)
        .map_err(|error| dataset_error(format!("format stopword JSON invalid: {error}")))?;
    let mut normalized = HashSet::new();
    let mut phrases = HashMap::<String, Vec<Vec<String>>>::new();
    for term in terms {
        let term = normalize_text(&term)?;
        let components = super::tokenize(&term)?;
        if components.is_empty() {
            return Err(dataset_error(
                "stopword tidak boleh kosong atau hanya tanda baca",
            ));
        }
        if !normalized.insert(term) {
            return Err(dataset_error("stopword duplikat setelah normalisasi"));
        }
        phrases
            .entry(components[0].clone())
            .or_default()
            .push(components);
    }
    if normalized.is_empty() {
        return Err(dataset_error("stopword list kosong"));
    }
    for candidates in phrases.values_mut() {
        candidates.sort_by_key(|phrase| std::cmp::Reverse(phrase.len()));
    }
    Ok((normalized, phrases))
}

fn import_sentence(
    lexicon: &mut IndonesianVocabulary,
    rows: &[ConlluToken],
    context: UdImportContext<'_>,
    report: &mut DatasetLoadReport,
) -> Result<(), SimError> {
    let mut word_ids = BTreeMap::new();
    let mut row_ids = BTreeSet::new();
    for row in rows {
        if !row_ids.insert(row.id) {
            return Err(dataset_error(format!(
                "ID token duplikat dalam kalimat {}",
                if context.sentence_id.is_empty() {
                    "(tanpa sent_id)"
                } else {
                    context.sentence_id
                }
            )));
        }
        let Ok(form) = normalize_lexeme(&row.form) else {
            report.skipped_forms += 1;
            continue;
        };
        let source = context.registry.entry_provenance(
            context.source_id,
            context.file_name,
            row.line_number,
        )?;
        let word_id = lexicon.upsert_dataset_entry(LexiconEntryInput {
            headword: form,
            sources: vec![source.clone()],
            linguistic: LinguisticMetadata {
                part_of_speech: (row.upos != "_").then(|| row.upos.clone()),
                lemma: (row.lemma != "_").then(|| row.lemma.clone()),
                ..LinguisticMetadata::default()
            },
        })?;
        word_ids.insert(row.id, word_id);
        report.tokens += 1;

        if row.lemma != "_" && row.upos != "_" {
            let lemma_tokens = super::tokenize(&row.lemma)?;
            if lemma_tokens.is_empty() {
                return Err(dataset_error(format!(
                    "lemma kosong pada baris {}",
                    row.line_number
                )));
            }
            if lemma_tokens.len() == 1 {
                lexicon.analyses.push(LexicalAnalysis {
                    word_id,
                    lemma: lemma_tokens[0].clone(),
                    upos: row.upos.clone(),
                    features: row.features.clone(),
                    dataset: context.dataset.clone(),
                    source_entry_id: source.source_entry_id.clone(),
                });
                report.analyses_added += 1;
            }
        }
    }
    let row_by_id = rows
        .iter()
        .map(|row| (row.id, row))
        .collect::<BTreeMap<_, _>>();
    for row in rows {
        let Some(target_word) = word_ids.get(&row.id).copied() else {
            continue;
        };
        if row.head == row.id || (row.head != 0 && !row_by_id.contains_key(&row.head)) {
            return Err(dataset_error(format!(
                "HEAD tidak merujuk token valid pada baris {}",
                row.line_number
            )));
        }
        let head_word = word_ids.get(&row.head).copied();
        if row.head != 0 && head_word.is_none() {
            continue;
        }
        let source = context.registry.entry_provenance(
            context.source_id,
            context.file_name,
            row.line_number,
        )?;
        let relation = LexicalRelation {
            source_word: if row.head == 0 { None } else { head_word },
            target_word,
            relation_type: LexicalRelationType::Dependency(row.dependency.clone()),
            dataset: context.dataset.clone(),
            source_entry_id: format!(
                "{}:{}:{}",
                if context.sentence_id.is_empty() {
                    "sentence"
                } else {
                    context.sentence_id
                },
                row.id,
                row.line_number
            ),
            source,
        };
        lexicon.relations.push(relation);
        report.dependencies_added += 1;
    }
    Ok(())
}

fn parse_features(value: &str, line_number: usize) -> Result<BTreeMap<String, String>, SimError> {
    if value == "_" {
        return Ok(BTreeMap::new());
    }
    let mut features = BTreeMap::new();
    for feature in value.split('|') {
        let Some((key, value)) = feature.split_once('=') else {
            return Err(dataset_error(format!(
                "FEATS tidak valid pada baris {line_number}"
            )));
        };
        if key.is_empty()
            || value.is_empty()
            || features.insert(key.to_owned(), value.to_owned()).is_some()
        {
            return Err(dataset_error(format!(
                "FEATS kosong atau duplikat pada baris {line_number}"
            )));
        }
    }
    Ok(features)
}

fn parse_positive(value: &str, line_number: usize, field: &str) -> Result<usize, SimError> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| dataset_error(format!("{field} tidak valid pada baris {line_number}")))?;
    if parsed == 0 {
        return Err(dataset_error(format!(
            "{field} harus positif pada baris {line_number}"
        )));
    }
    Ok(parsed)
}

pub(super) fn io_error(operation: &'static str, error: std::io::Error) -> SimError {
    SimError::Io {
        operation,
        kind: error.kind(),
    }
}

pub(super) fn dataset_error(reason: impl Into<String>) -> SimError {
    SimError::CorruptData {
        subsystem: "Indonesian dataset",
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        LifConfig, LifNetwork,
        language::{LanguageNetwork, LexiconMetadata, SourceProvenance},
    };

    fn test_provenance() -> DatasetProvenance {
        DatasetProvenance {
            id: DatasetId::StopwordsIsoId,
            repository_url: "https://example.invalid/test-fixture".to_owned(),
            revision: "test-only".to_owned(),
            license_id: "test-only".to_owned(),
            license_url: "https://example.invalid/test-license".to_owned(),
            attribution: "test fixture, not upstream dataset".to_owned(),
            data_scope: "unit-test-only".to_owned(),
            file_git_blob_sha1: BTreeMap::new(),
        }
    }

    #[test]
    fn stopword_parser_normalizes_terms_and_rejects_duplicates_and_empty_lists() {
        let (terms, phrases) =
            parse_stopwords_terms(r#"[" DAN ","berkali-kali"]"#).expect("test JSON valid");
        assert!(terms.contains("dan"));
        assert!(phrases.contains_key("berkali"));

        assert!(parse_stopwords_terms(r#"["Dan","dan"]"#).is_err());
        assert!(parse_stopwords_terms("[]").is_err());
        assert!(parse_stopwords_terms(r#"["..."]"#).is_err());
    }

    #[test]
    fn stopwords_filter_real_language_network_input() {
        let (terms, phrases) =
            parse_stopwords_terms(r#"["dan","yang","berkali-kali"]"#).expect("test terms");
        let stopwords = StopwordSet {
            provenance: test_provenance(),
            terms,
            phrases,
        };
        let mut lexicon = IndonesianVocabulary::new(LexiconMetadata::indonesian("unit-test", "1"))
            .expect("metadata valid");
        let word_id = lexicon
            .upsert(LexiconEntryInput {
                headword: "nasi".to_owned(),
                sources: vec![SourceProvenance {
                    source_name: "test fixture, not upstream data".to_owned(),
                    source_uri: "https://example.invalid/test-fixture".to_owned(),
                    license: "test-only".to_owned(),
                    license_uri: "https://example.invalid/test-license".to_owned(),
                    source_entry_id: "fixture:1".to_owned(),
                }],
                linguistic: LinguisticMetadata::default(),
            })
            .expect("test entry");
        let network = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0), 1)
            .expect("network config valid");
        let mut language: LanguageNetwork = lexicon
            .to_language_network(network)
            .expect("lexicon integrates with network");
        let neuron = language.network().neuron_id(0).expect("neuron exists");
        language
            .bind_word(word_id, &[neuron])
            .expect("word-neuron binding");
        assert_eq!(
            language
                .encode_currents_without_stopwords("yang nasi", 40.0, &stopwords)
                .expect("filtered currents"),
            [40.0]
        );
        assert!(
            language
                .encode_currents_without_stopwords("dan yang", 40.0, &stopwords)
                .is_err()
        );
    }
}

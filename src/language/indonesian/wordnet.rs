use super::{DatasetId, DatasetProvenance, LicenseRegistry, datasets::dataset_error};
use crate::SimError;
use crate::language::{normalize_text, tokenize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

const MAX_WORDNET_FILE_BYTES: u64 = 256 * 1024 * 1024;
pub(crate) const MEMBERSHIP_PATH: &str = "data/msa-wordnet-r24/wn-msa-all.tab";
pub(crate) const DEFINITION_PATH: &str = "data/msa-wordnet-r24/wn-ind-def.tab";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordNetBahasaMember {
    pub synset_id: String,
    pub lemma: String,
    pub source_line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordNetBahasaDefinition {
    pub text: String,
    pub source_line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordNetBahasaDataset {
    source: DatasetProvenance,
    members: Vec<WordNetBahasaMember>,
    lemma_synsets: BTreeMap<String, BTreeSet<String>>,
    definitions: BTreeMap<String, WordNetBahasaDefinition>,
    definition_conflicts: BTreeSet<String>,
    pub membership_rows: usize,
    pub indonesian_rows: usize,
    pub indonesian_quality_y_rows: usize,
    pub multiword_quality_y_rows: usize,
    pub duplicate_quality_y_rows: usize,
    pub definition_rows: usize,
}

impl WordNetBahasaDataset {
    pub(crate) const fn source(&self) -> &DatasetProvenance {
        &self.source
    }

    pub(crate) fn members(&self) -> &[WordNetBahasaMember] {
        &self.members
    }

    pub(crate) fn synsets_for_lemma(&self, lemma: &str) -> Option<&BTreeSet<String>> {
        self.lemma_synsets.get(lemma)
    }

    pub(crate) fn definition(&self, synset_id: &str) -> Option<&WordNetBahasaDefinition> {
        self.definitions.get(synset_id)
    }

    pub(crate) fn has_definition_conflict(&self, synset_id: &str) -> bool {
        self.definition_conflicts.contains(synset_id)
    }

    #[must_use]
    pub fn unique_synsets(&self) -> usize {
        self.members
            .iter()
            .map(|member| member.synset_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    }
}

impl super::DatasetLoader {
    /// Read and verify the pinned WordNet Bahasa r24 membership and Indonesian gloss files.
    /// Only I/Y single-token membership rows are returned for import; other I-quality
    /// rows are retained in the ambiguity index but never promoted to knowledge.
    pub fn load_wordnet_bahasa(
        membership_path: impl AsRef<Path>,
        definition_path: impl AsRef<Path>,
        registry: &LicenseRegistry,
    ) -> Result<WordNetBahasaDataset, SimError> {
        let source = registry.provenance(DatasetId::WordNetBahasa)?;
        let membership_bytes = read_verified_file(
            membership_path.as_ref(),
            MEMBERSHIP_PATH,
            registry,
        )?;
        let definition_bytes = read_verified_file(
            definition_path.as_ref(),
            DEFINITION_PATH,
            registry,
        )?;
        let membership_text = std::str::from_utf8(&membership_bytes)
            .map_err(|_| dataset_error("WordNet Bahasa membership bukan UTF-8"))?;
        let definition_text = std::str::from_utf8(&definition_bytes)
            .map_err(|_| dataset_error("WordNet Bahasa definitions bukan UTF-8"))?;

        let mut members = Vec::new();
        let mut lemma_synsets = BTreeMap::<String, BTreeSet<String>>::new();
        let mut selected_members = BTreeSet::new();
        let mut membership_rows = 0;
        let mut indonesian_rows = 0;
        let mut indonesian_quality_y_rows = 0;
        let mut multiword_quality_y_rows = 0;
        let mut duplicate_quality_y_rows = 0;
        for (line_index, line) in membership_text.lines().enumerate() {
            let source_line = line_index + 1;
            if line.is_empty() {
                return Err(dataset_error("baris kosong pada WordNet Bahasa membership"));
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 4
                || !valid_synset_id(fields[0])
                || !matches!(fields[1], "B" | "I" | "M")
                || !matches!(fields[2], "Y" | "O" | "M" | "L" | "X")
                || fields[3].trim().is_empty()
                || fields[3].chars().any(char::is_control)
            {
                return Err(dataset_error("format baris WordNet Bahasa tidak valid"));
            }
            membership_rows += 1;
            if fields[1] != "I" {
                continue;
            }
            indonesian_rows += 1;
            let normalized = normalize_text(fields[3])?;
            lemma_synsets
                .entry(normalized.clone())
                .or_default()
                .insert(fields[0].to_owned());
            if fields[2] != "Y" {
                continue;
            }
            indonesian_quality_y_rows += 1;
            let tokens = tokenize(&normalized)?;
            if tokens.len() != 1 || tokens[0] != normalized {
                multiword_quality_y_rows += 1;
                continue;
            }
            if !selected_members.insert((fields[0].to_owned(), normalized.clone())) {
                duplicate_quality_y_rows += 1;
                continue;
            }
            members.push(WordNetBahasaMember {
                synset_id: fields[0].to_owned(),
                lemma: normalized,
                source_line,
            });
        }
        if membership_rows == 0 || indonesian_rows == 0 {
            return Err(dataset_error("WordNet Bahasa membership kosong"));
        }

        let mut definitions_by_synset = BTreeMap::<String, BTreeMap<String, usize>>::new();
        let mut definition_rows = 0;
        for (line_index, line) in definition_text.lines().enumerate() {
            let fields = line.splitn(3, '\t').collect::<Vec<_>>();
            if fields.len() != 3
                || !valid_synset_id(fields[0])
                || fields[1] != "ind:def"
                || fields[2].trim().is_empty()
                || fields[2].chars().any(char::is_control)
            {
                return Err(dataset_error("format baris WordNet Bahasa definition tidak valid"));
            }
            definition_rows += 1;
            definitions_by_synset
                .entry(fields[0].to_owned())
                .or_default()
                .entry(fields[2].to_owned())
                .or_insert(line_index + 1);
        }
        if definition_rows == 0 {
            return Err(dataset_error("WordNet Bahasa definition kosong"));
        }

        let selected_synsets = members
            .iter()
            .map(|member| member.synset_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut definitions = BTreeMap::new();
        let mut definition_conflicts = BTreeSet::new();
        for synset_id in selected_synsets {
            let Some(candidates) = definitions_by_synset.get(synset_id) else {
                continue;
            };
            if candidates.len() > 1 {
                definition_conflicts.insert(synset_id.to_owned());
                continue;
            }
            if let Some((text, source_line)) = candidates.first_key_value() {
                definitions.insert(
                    synset_id.to_owned(),
                    WordNetBahasaDefinition {
                        text: text.clone(),
                        source_line: *source_line,
                    },
                );
            }
        }

        Ok(WordNetBahasaDataset {
            source,
            members,
            lemma_synsets,
            definitions,
            definition_conflicts,
            membership_rows,
            indonesian_rows,
            indonesian_quality_y_rows,
            multiword_quality_y_rows,
            duplicate_quality_y_rows,
            definition_rows,
        })
    }
}

fn read_verified_file(
    path: &Path,
    source_path: &str,
    registry: &LicenseRegistry,
) -> Result<Vec<u8>, SimError> {
    let file = File::open(path).map_err(|error| super::datasets::io_error("buka WordNet Bahasa", error))?;
    let length = file
        .metadata()
        .map_err(|error| super::datasets::io_error("baca metadata WordNet Bahasa", error))?
        .len();
    if length > MAX_WORDNET_FILE_BYTES {
        return Err(dataset_error("file WordNet Bahasa melebihi batas 256 MiB"));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(MAX_WORDNET_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| super::datasets::io_error("baca WordNet Bahasa", error))?;
    if bytes.len() as u64 != length {
        return Err(dataset_error("ukuran file WordNet Bahasa berubah saat dibaca"));
    }
    registry.verify_file(DatasetId::WordNetBahasa, source_path, &bytes)?;
    Ok(bytes)
}

fn valid_synset_id(value: &str) -> bool {
    let Some((offset, part_of_speech)) = value.split_once('-') else {
        return false;
    };
    offset.len() == 8
        && offset.bytes().all(|byte| byte.is_ascii_digit())
        && matches!(part_of_speech, "n" | "v" | "a" | "s" | "r")
}
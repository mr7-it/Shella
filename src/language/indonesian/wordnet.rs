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
    pub quality: char,
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
    pub indonesian_quality_o_rows: usize,
    pub multiword_quality_rows: usize,
    pub duplicate_quality_rows: usize,
    pub definition_rows: usize,
}

impl WordNetBahasaDataset {
    #[must_use]
    pub const fn source(&self) -> &DatasetProvenance {
        &self.source
    }

    #[must_use]
    pub fn members(&self) -> &[WordNetBahasaMember] {
        &self.members
    }

    pub(crate) fn synsets_for_lemma(&self, lemma: &str) -> Option<&BTreeSet<String>> {
        self.lemma_synsets.get(lemma)
    }

    #[must_use]
    pub fn definition(&self, synset_id: &str) -> Option<&WordNetBahasaDefinition> {
        self.definitions.get(synset_id)
    }

    #[must_use]
    pub fn has_definition_conflict(&self, synset_id: &str) -> bool {
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WordNetBahasaImportReport {
    pub source_memberships: usize,
    pub indonesian_rows: usize,
    pub imported_entries: usize,
    pub reused_entries: usize,
    pub imported_definitions: usize,
    pub validated_synset_relations: usize,
    pub skipped_ambiguous_relations: usize,
    pub skipped_conflicting_relations: usize,
}

impl super::DatasetLoader {
    /// Import quality Y/O Indonesian lemmas and same-synset relations atomically into
    /// the existing lexicon. No translation edges or cross-synset relations are inferred.
    pub fn import_wordnet_bahasa(
        dataset: &WordNetBahasaDataset,
        lexicon: &mut super::IndonesianVocabulary,
        registry: &LicenseRegistry,
    ) -> Result<WordNetBahasaImportReport, SimError> {
        registry.verify(&dataset.source)?;
        let mut staged = lexicon.clone();
        staged.add_dataset_provenance(dataset.source.clone(), registry)?;
        let mut report = WordNetBahasaImportReport {
            source_memberships: dataset.members.len(),
            indonesian_rows: dataset.indonesian_rows,
            ..WordNetBahasaImportReport::default()
        };
        let mut ids = BTreeMap::new();
        let mut rows = BTreeMap::new();
        for member in &dataset.members {
            let existed = staged.lookup(&member.lemma)?.is_some();
            let mut source = registry.entry_provenance(
                DatasetId::WordNetBahasa,
                MEMBERSHIP_PATH,
                member.source_line,
            )?;
            source.source_entry_id = format!(
                "synset:{}:quality:{}:line:{}",
                member.synset_id, member.quality, member.source_line
            );
            let current_entry = staged.lookup(&member.lemma)?;
            let current_linguistic = current_entry.map(|entry| entry.linguistic.clone());
            let mut linguistic = current_linguistic.unwrap_or_default();
            let has_existing_pos_analysis = current_entry.is_some_and(|entry| {
                staged
                    .analyses()
                    .iter()
                    .any(|analysis| analysis.word_id == entry.word_id && !analysis.upos.is_empty())
            });
            let mut sources = current_entry
                .and_then(|entry| {
                    entry
                        .linguistic
                        .definition_source_entry_id
                        .as_deref()
                        .and_then(|source_id| {
                            entry
                                .sources
                                .iter()
                                .find(|known| known.source_entry_id == source_id)
                        })
                        .cloned()
                })
                .into_iter()
                .collect::<Vec<_>>();
            if !sources.iter().any(|known| {
                known.source_uri == source.source_uri
                    && known.source_entry_id == source.source_entry_id
            }) {
                sources.push(source);
            }
            let mut source_for_definition = Vec::new();
            if linguistic.lemma.is_none() {
                linguistic.lemma = Some(member.lemma.clone());
            }
            if linguistic.part_of_speech.is_none()
                && !has_existing_pos_analysis
                && dataset
                    .synsets_for_lemma(&member.lemma)
                    .is_some_and(|synsets| synsets.len() == 1)
            {
                linguistic.part_of_speech =
                    wordnet_part_of_speech(&member.synset_id).map(str::to_owned);
            }
            if let Some(definition) = dataset.definition(&member.synset_id) {
                if linguistic.definition.is_none()
                    && dataset
                        .synsets_for_lemma(&member.lemma)
                        .is_some_and(|synsets| synsets.len() == 1)
                    && !dataset.has_definition_conflict(&member.synset_id)
                {
                    let definition_source = registry.entry_provenance(
                        DatasetId::WordNetBahasa,
                        DEFINITION_PATH,
                        definition.source_line,
                    )?;
                    linguistic.definition = Some(definition.text.clone());
                    linguistic.definition_source_entry_id =
                        Some(definition_source.source_entry_id.clone());
                    source_for_definition.push(definition_source);
                    report.imported_definitions += 1;
                }
            }
            sources.extend(source_for_definition);
            let id = staged.upsert_dataset_entry(super::LexiconEntryInput {
                headword: member.lemma.clone(),
                sources,
                linguistic,
            })?;
            if existed {
                report.reused_entries += 1;
            } else {
                report.imported_entries += 1;
            }
            ids.insert((member.synset_id.clone(), member.lemma.clone()), id);
            rows.insert(
                (member.synset_id.clone(), member.lemma.clone()),
                member.source_line,
            );
        }

        let mut by_synset = BTreeMap::<String, BTreeSet<String>>::new();
        for member in &dataset.members {
            by_synset
                .entry(member.synset_id.clone())
                .or_default()
                .insert(member.lemma.clone());
        }
        let mut relations = Vec::new();
        for (synset, lemmas) in by_synset {
            let all_members = lemmas.len();
            let eligible = lemmas
                .into_iter()
                .filter(|lemma| {
                    dataset
                        .synsets_for_lemma(lemma)
                        .is_some_and(|synsets| synsets.len() == 1)
                })
                .collect::<Vec<_>>();
            report.skipped_ambiguous_relations += all_members.saturating_sub(eligible.len());
            for left in 0..eligible.len() {
                for right in (left + 1)..eligible.len() {
                    let left_lemma = &eligible[left];
                    let right_lemma = &eligible[right];
                    let source_line = *rows
                        .get(&(synset.clone(), left_lemma.clone()))
                        .ok_or_else(|| dataset_error("source line synset tidak ditemukan"))?;
                    let target_line = *rows
                        .get(&(synset.clone(), right_lemma.clone()))
                        .ok_or_else(|| dataset_error("source line synset tidak ditemukan"))?;
                    let source_word = *ids
                        .get(&(synset.clone(), left_lemma.clone()))
                        .ok_or_else(|| dataset_error("WordID synset tidak ditemukan"))?;
                    let target_word = *ids
                        .get(&(synset.clone(), right_lemma.clone()))
                        .ok_or_else(|| dataset_error("WordID synset tidak ditemukan"))?;
                    let source = registry.entry_provenance(
                        DatasetId::WordNetBahasa,
                        MEMBERSHIP_PATH,
                        source_line,
                    )?;
                    let relation = super::LexicalRelation {
                        source_word: Some(source_word),
                        target_word,
                        relation_type: super::LexicalRelationType::Synonym,
                        dataset: dataset.source.clone(),
                        source_entry_id: format!(
                            "synset:{synset}:lines:{source_line}:{target_line}"
                        ),
                        source,
                    };
                    relations.push(relation);
                }
            }
        }
        let (added, conflicts) = staged.add_relations_batch(relations, registry)?;
        report.validated_synset_relations = added;
        report.skipped_conflicting_relations = conflicts;
        *lexicon = staged;
        Ok(report)
    }

    /// Read and verify the pinned WordNet Bahasa r24 membership and Indonesian gloss files.
    /// Only Indonesian I/Y/O single-token rows are returned for import. Other I-quality
    /// rows remain in the ambiguity index but are not promoted as lexicon entries.
    pub fn load_wordnet_bahasa(
        membership_path: impl AsRef<Path>,
        definition_path: impl AsRef<Path>,
        registry: &LicenseRegistry,
    ) -> Result<WordNetBahasaDataset, SimError> {
        let source = registry.provenance(DatasetId::WordNetBahasa)?;
        let membership_bytes =
            read_verified_file(membership_path.as_ref(), MEMBERSHIP_PATH, registry)?;
        let definition_bytes =
            read_verified_file(definition_path.as_ref(), DEFINITION_PATH, registry)?;
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
        let mut indonesian_quality_o_rows = 0;
        let mut multiword_quality_rows = 0;
        let mut duplicate_quality_rows = 0;
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
            if !matches!(fields[2], "Y" | "O") {
                continue;
            }
            if fields[2] == "Y" {
                indonesian_quality_y_rows += 1;
            } else {
                indonesian_quality_o_rows += 1;
            }
            let tokens = tokenize(&normalized)?;
            if tokens.len() != 1 || tokens[0] != normalized {
                multiword_quality_rows += 1;
                continue;
            }
            if !selected_members.insert((fields[0].to_owned(), normalized.clone())) {
                duplicate_quality_rows += 1;
                continue;
            }
            members.push(WordNetBahasaMember {
                synset_id: fields[0].to_owned(),
                lemma: normalized,
                quality: fields[2]
                    .chars()
                    .next()
                    .ok_or_else(|| dataset_error("label mutu kosong"))?,
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
                return Err(dataset_error(
                    "format baris WordNet Bahasa definition tidak valid",
                ));
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
            indonesian_quality_o_rows,
            multiword_quality_rows,
            duplicate_quality_rows,
            definition_rows,
        })
    }
}

fn read_verified_file(
    path: &Path,
    source_path: &str,
    registry: &LicenseRegistry,
) -> Result<Vec<u8>, SimError> {
    let file = File::open(path)
        .map_err(|error| super::datasets::io_error("buka WordNet Bahasa", error))?;
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
        return Err(dataset_error(
            "ukuran file WordNet Bahasa berubah saat dibaca",
        ));
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

fn wordnet_part_of_speech(synset_id: &str) -> Option<&'static str> {
    match synset_id.rsplit_once('-')?.1 {
        "n" => Some("NOUN"),
        "v" => Some("VERB"),
        "a" | "s" => Some("ADJ"),
        "r" => Some("ADV"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DatasetLoader, IndonesianVocabulary};
    use super::*;
    use crate::language::{
        LexicalRelation, LexicalRelationType, LexiconEntryInput, LexiconMetadata,
        LinguisticMetadata, SourceProvenance,
    };
    use std::collections::BTreeMap;

    #[test]
    fn impor_synset_mengabaikan_edge_yang_bertentangan_dan_atomik() {
        let registry = LicenseRegistry::verified_sources();
        let provenance = registry
            .provenance(DatasetId::WordNetBahasa)
            .expect("WordNet terdaftar");
        let mut lexicon = IndonesianVocabulary::new(LexiconMetadata::indonesian("fixture", "1"))
            .expect("lexicon valid");
        let fixture_source = |word: &str| SourceProvenance {
            source_name: "unit test fixture".to_owned(),
            source_uri: "https://example.invalid/fixture".to_owned(),
            license: "CC0-1.0".to_owned(),
            license_uri: "https://creativecommons.org/publicdomain/zero/1.0/".to_owned(),
            source_entry_id: format!("fixture:{word}"),
        };
        let animal = lexicon
            .upsert(LexiconEntryInput {
                headword: "hewan".to_owned(),
                sources: vec![fixture_source("hewan")],
                linguistic: LinguisticMetadata::default(),
            })
            .expect("WordID hewan");
        let fauna = lexicon
            .upsert(LexiconEntryInput {
                headword: "satwa".to_owned(),
                sources: vec![fixture_source("satwa")],
                linguistic: LinguisticMetadata::default(),
            })
            .expect("WordID satwa");
        let source = registry
            .entry_provenance(DatasetId::WordNetBahasa, MEMBERSHIP_PATH, 1)
            .expect("provenance relasi fixture");
        lexicon
            .add_relation(
                LexicalRelation {
                    source_word: Some(animal),
                    target_word: fauna,
                    relation_type: LexicalRelationType::Antonym,
                    dataset: provenance.clone(),
                    source_entry_id: "fixture:existing-antonym".to_owned(),
                    source,
                },
                &registry,
            )
            .expect("edge existing");

        let dataset = WordNetBahasaDataset {
            source: provenance,
            members: vec![
                WordNetBahasaMember {
                    synset_id: "00015388-n".to_owned(),
                    lemma: "hewan".to_owned(),
                    quality: 'O',
                    source_line: 1,
                },
                WordNetBahasaMember {
                    synset_id: "00015388-n".to_owned(),
                    lemma: "satwa".to_owned(),
                    quality: 'Y',
                    source_line: 2,
                },
            ],
            lemma_synsets: BTreeMap::from([
                (
                    "hewan".to_owned(),
                    BTreeSet::from(["00015388-n".to_owned()]),
                ),
                (
                    "satwa".to_owned(),
                    BTreeSet::from(["00015388-n".to_owned()]),
                ),
            ]),
            definitions: BTreeMap::new(),
            definition_conflicts: BTreeSet::new(),
            membership_rows: 2,
            indonesian_rows: 2,
            indonesian_quality_y_rows: 1,
            indonesian_quality_o_rows: 1,
            multiword_quality_rows: 0,
            duplicate_quality_rows: 0,
            definition_rows: 0,
        };
        let report = DatasetLoader::import_wordnet_bahasa(&dataset, &mut lexicon, &registry)
            .expect("batch conflict is handled without aborting");
        assert_eq!(report.validated_synset_relations, 0);
        assert_eq!(report.skipped_conflicting_relations, 1);
        assert_eq!(lexicon.relations().len(), 1);
        assert_eq!(
            lexicon.relations()[0].relation_type,
            LexicalRelationType::Antonym
        );
    }
}

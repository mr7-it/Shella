use super::{
    DatasetId, DatasetLoadReport, DatasetProvenance, IndonesianVocabulary, LexicalAnalysis,
    LexiconEntry, LicenseRegistry, LinguisticMetadata, normalize_lexeme,
};
use crate::{SimError, WordID};
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::Path,
};

const MAX_DICTIONARY_BYTES: u64 = 64 * 1024 * 1024;
const MAX_AFFIX_BYTES: u64 = 2 * 1024 * 1024;
type AffixGroupCounts = BTreeMap<(String, String), (usize, usize, bool)>;

struct AffixRuleSpec<'a> {
    kind: &'a str,
    flag: &'a str,
    strip: &'a str,
    add: &'a str,
    condition: &'a str,
    line_number: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HunspellAffixKind {
    Prefix,
    Suffix,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HunspellRoot {
    pub word_id: WordID,
    pub flags: Vec<String>,
    pub dataset: DatasetProvenance,
    pub source_entry_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HunspellAffixRule {
    pub kind: HunspellAffixKind,
    pub flag: String,
    pub strip: String,
    pub add: String,
    pub condition: String,
    pub cross_product: bool,
    pub dataset: DatasetProvenance,
    pub source_entry_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HunspellAnalysis {
    pub surface: String,
    pub lemma_word_id: WordID,
    pub affix_rule: Option<String>,
    pub source_entry_id: String,
}

impl super::DatasetLoader {
    /// Load pinned Indonesia.dic and Indonesia.aff without bundling their contents.
    ///
    /// Only single-token roots are interned; compounds unsupported by Shella's
    /// token identity model are counted as skipped. Derivation analysis applies
    /// one affix rule at a time and does not claim full Hunspell parity.
    pub fn load_hunspell_files(
        dictionary_path: impl AsRef<Path>,
        affix_path: impl AsRef<Path>,
        lexicon: &mut IndonesianVocabulary,
        registry: &LicenseRegistry,
    ) -> Result<DatasetLoadReport, SimError> {
        let dataset = registry.provenance(DatasetId::HunspellDictionaries)?;
        let dictionary_bytes = read_bounded(
            dictionary_path.as_ref(),
            MAX_DICTIONARY_BYTES,
            "buka Hunspell dictionary",
        )?;
        let affix_bytes =
            read_bounded(affix_path.as_ref(), MAX_AFFIX_BYTES, "buka Hunspell affix")?;
        registry.verify_file(
            DatasetId::HunspellDictionaries,
            "Indonesia.dic",
            &dictionary_bytes,
        )?;
        registry.verify_file(
            DatasetId::HunspellDictionaries,
            "Indonesia.aff",
            &affix_bytes,
        )?;
        let dictionary = std::str::from_utf8(&dictionary_bytes)
            .map_err(|_| dataset_error("Indonesia.dic bukan UTF-8 valid"))?;
        let affix = std::str::from_utf8(&affix_bytes)
            .map_err(|_| dataset_error("Indonesia.aff bukan UTF-8 valid"))?;
        let rules = parse_affix(affix, &dataset)?;
        let roots = parse_dictionary(dictionary)?;

        let mut staged = lexicon.clone();
        staged.add_dataset_provenance(dataset.clone(), registry)?;
        let initial_entries = staged.entries.len();
        let mut report = DatasetLoadReport::default();
        let mut root_records = Vec::new();
        let mut root_keys = staged
            .hunspell_roots
            .iter()
            .filter(|root| root.dataset.id == DatasetId::HunspellDictionaries)
            .map(|root| (root.word_id, root.flags.clone()))
            .collect::<HashSet<_>>();
        let mut word_indices = staged.aliases.clone();
        for (line_number, spelling, flags) in roots {
            if spelling.contains('-') {
                report.skipped_forms += 1;
                continue;
            }
            let Ok(headword) = normalize_lexeme(&spelling) else {
                report.skipped_forms += 1;
                continue;
            };
            report.tokens += 1;
            let provenance = registry.entry_provenance(
                DatasetId::HunspellDictionaries,
                "Indonesia.dic",
                line_number,
            )?;
            let word_id = if let Some(existing) = word_indices.get(&headword).copied() {
                let index = *staged.word_index.get(&existing).ok_or_else(|| {
                    dataset_error("WordID root Hunspell tidak memiliki indeks vocabulary")
                })?;
                let entry = staged.entries.get_mut(index).ok_or_else(|| {
                    dataset_error("indeks entry Hunspell berada di luar vocabulary")
                })?;
                if !entry.sources.iter().any(|source| source == &provenance) {
                    entry.sources.push(provenance.clone());
                }
                existing
            } else {
                let word_id = staged.vocabulary.intern_word(&headword)?;
                let index = staged.entries.len();
                word_indices.insert(headword.clone(), word_id);
                staged.entries.push(LexiconEntry {
                    word_id,
                    headword,
                    sources: vec![provenance],
                    linguistic: LinguisticMetadata::default(),
                });
                staged.word_index.insert(word_id, index);
                word_id
            };
            if root_keys.insert((word_id, flags.clone())) {
                root_records.push(HunspellRoot {
                    word_id,
                    flags,
                    dataset: dataset.clone(),
                    source_entry_id: format!("Indonesia.dic:{line_number}"),
                });
            }
        }
        report.entries_added = staged.entries.len().saturating_sub(initial_entries);
        report.analyses_added = rules.len();
        report.sentences = 0;
        staged.hunspell_roots.extend(root_records);
        staged
            .hunspell_rules
            .retain(|rule| rule.dataset.id != DatasetId::HunspellDictionaries);
        staged.hunspell_rules.extend(rules);
        staged.rebuild_aliases()?;
        staged.validate()?;
        *lexicon = staged;
        Ok(report)
    }
}

impl IndonesianVocabulary {
    /// Resolve a registered surface to its Hunspell root through one direct rule.
    pub fn analyze_hunspell(&self, surface: &str) -> Result<Vec<HunspellAnalysis>, SimError> {
        let surface = normalize_lexeme(surface)?;
        let mut found = Vec::new();
        for root in &self.hunspell_roots {
            let Some(lemma) = self.vocabulary.word(root.word_id) else {
                continue;
            };
            if lemma == surface {
                found.push(HunspellAnalysis {
                    surface: surface.clone(),
                    lemma_word_id: root.word_id,
                    affix_rule: None,
                    source_entry_id: root.source_entry_id.clone(),
                });
                continue;
            }
            for rule in self.hunspell_rules.iter().filter(|rule| {
                rule.dataset == root.dataset && root.flags.iter().any(|flag| flag == &rule.flag)
            }) {
                if apply_rule(lemma, rule).as_deref() == Some(surface.as_str()) {
                    found.push(HunspellAnalysis {
                        surface: surface.clone(),
                        lemma_word_id: root.word_id,
                        affix_rule: Some(rule.source_entry_id.clone()),
                        source_entry_id: root.source_entry_id.clone(),
                    });
                }
            }
        }
        found.sort_by(|left, right| {
            left.lemma_word_id
                .cmp(&right.lemma_word_id)
                .then_with(|| left.affix_rule.cmp(&right.affix_rule))
        });
        found.dedup();
        Ok(found)
    }

    /// Attach a morphology record from another verified source such as UD.
    pub fn add_verified_analysis(
        &mut self,
        analysis: LexicalAnalysis,
        registry: &LicenseRegistry,
    ) -> Result<(), SimError> {
        self.add_analysis(analysis, registry)
    }
}

fn parse_dictionary(contents: &str) -> Result<Vec<(usize, String, Vec<String>)>, SimError> {
    let mut lines = contents.lines().enumerate();
    let (header_line, header) = lines
        .next()
        .ok_or_else(|| dataset_error("Indonesia.dic kosong"))?;
    let expected = header.trim().parse::<usize>().map_err(|_| {
        dataset_error(format!(
            "jumlah entri Indonesia.dic invalid pada baris {}",
            header_line + 1
        ))
    })?;
    let mut roots = Vec::with_capacity(expected.min(contents.lines().count()));
    for (index, raw) in lines {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (word, flags) = match line.split_once('/') {
            Some((word, flags)) => (word, parse_long_flags(flags)?),
            None => (line, Vec::new()),
        };
        if word.is_empty() {
            return Err(dataset_error(format!(
                "kata kosong pada baris dictionary {}",
                index + 1
            )));
        }
        roots.push((index + 1, word.to_owned(), flags));
    }
    if roots.len() != expected {
        return Err(dataset_error(format!(
            "jumlah baris dictionary {} tidak sesuai header {expected}",
            roots.len()
        )));
    }
    Ok(roots)
}

fn parse_long_flags(flags: &str) -> Result<Vec<String>, SimError> {
    if flags.is_empty() {
        return Ok(Vec::new());
    }
    if flags.len() & 1 != 0 {
        return Err(dataset_error(
            "flag Hunspell long harus terdiri dari pasangan byte",
        ));
    }
    if !flags.is_ascii() {
        return Err(dataset_error("flag Hunspell harus ASCII"));
    }
    Ok(flags
        .as_bytes()
        .chunks_exact(2)
        .map(|flag| String::from_utf8_lossy(flag).into_owned())
        .collect())
}

fn parse_affix(
    contents: &str,
    dataset: &DatasetProvenance,
) -> Result<Vec<HunspellAffixRule>, SimError> {
    let mut rules = Vec::new();
    let mut group_counts = AffixGroupCounts::new();
    let mut encoding_is_utf8 = false;
    let mut flag_mode_long = false;
    for (line_number, raw) in contents.lines().enumerate() {
        let body = raw.split('#').next().unwrap_or_default().trim();
        if body.is_empty() {
            continue;
        }
        let columns = body.split_whitespace().collect::<Vec<_>>();
        match columns.as_slice() {
            ["SET", "UTF-8"] => encoding_is_utf8 = true,
            ["FLAG", "long"] => flag_mode_long = true,
            [kind @ ("PFX" | "SFX"), flag, cross, count] if matches!(*cross, "Y" | "N") => {
                let count = count.parse::<usize>().map_err(|_| {
                    dataset_error(format!(
                        "jumlah aturan afiks invalid pada baris {}",
                        line_number + 1
                    ))
                })?;
                if flag.is_empty() {
                    return Err(dataset_error(format!(
                        "header aturan afiks invalid pada baris {}",
                        line_number + 1
                    )));
                }
                let key = ((*kind).to_owned(), (*flag).to_owned());
                if group_counts
                    .insert(key, (count, 0, *cross == "Y"))
                    .is_some()
                {
                    return Err(dataset_error("header grup afiks duplikat"));
                }
            }
            [kind @ ("PFX" | "SFX"), flag, strip, add] => {
                append_affix_rule(
                    &mut rules,
                    &mut group_counts,
                    dataset,
                    AffixRuleSpec {
                        kind,
                        flag,
                        strip,
                        add,
                        condition: ".",
                        line_number: line_number + 1,
                    },
                )?;
            }
            [kind @ ("PFX" | "SFX"), flag, strip, add, condition] => {
                append_affix_rule(
                    &mut rules,
                    &mut group_counts,
                    dataset,
                    AffixRuleSpec {
                        kind,
                        flag,
                        strip,
                        add,
                        condition,
                        line_number: line_number + 1,
                    },
                )?;
            }
            [kind, ..] if *kind == "PFX" || *kind == "SFX" => {
                return Err(dataset_error(format!(
                    "baris aturan afiks invalid pada {}",
                    line_number + 1
                )));
            }
            _ => {}
        }
    }
    if !encoding_is_utf8 || !flag_mode_long {
        return Err(dataset_error(
            "Indonesia.aff wajib menyatakan SET UTF-8 dan FLAG long",
        ));
    }
    if group_counts
        .values()
        .any(|(expected, actual, _)| expected != actual)
    {
        return Err(dataset_error("jumlah aturan berbeda dari header Hunspell"));
    }
    Ok(rules)
}

fn append_affix_rule(
    rules: &mut Vec<HunspellAffixRule>,
    group_counts: &mut AffixGroupCounts,
    dataset: &DatasetProvenance,
    spec: AffixRuleSpec<'_>,
) -> Result<(), SimError> {
    let key = (spec.kind.to_owned(), spec.flag.to_owned());
    let group = group_counts.get_mut(&key).ok_or_else(|| {
        dataset_error(format!(
            "aturan afiks tanpa header pada baris {}",
            spec.line_number
        ))
    })?;
    if parse_condition(spec.condition).is_none() {
        return Err(dataset_error(format!(
            "kondisi aturan afiks invalid pada baris {}",
            spec.line_number
        )));
    }
    group.1 += 1;
    let (add, _) = spec.add.split_once('/').unwrap_or((spec.add, ""));
    rules.push(HunspellAffixRule {
        kind: if spec.kind == "PFX" {
            HunspellAffixKind::Prefix
        } else {
            HunspellAffixKind::Suffix
        },
        flag: spec.flag.to_owned(),
        strip: if spec.strip == "0" {
            String::new()
        } else {
            spec.strip.to_owned()
        },
        add: if add == "0" {
            String::new()
        } else {
            add.to_owned()
        },
        condition: spec.condition.to_owned(),
        cross_product: group_counts.get(&key).is_some_and(|(_, _, cross)| *cross),
        dataset: dataset.clone(),
        source_entry_id: format!("Indonesia.aff:{}", spec.line_number),
    });
    Ok(())
}

fn apply_rule(word: &str, rule: &HunspellAffixRule) -> Option<String> {
    if !condition_matches(word, &rule.condition, rule.kind) {
        return None;
    }
    match rule.kind {
        HunspellAffixKind::Prefix => {
            let remainder = word.strip_prefix(&rule.strip)?;
            Some(format!("{}{remainder}", rule.add))
        }
        HunspellAffixKind::Suffix => {
            let base = word.strip_suffix(&rule.strip)?;
            Some(format!("{base}{}", rule.add))
        }
    }
}

fn condition_matches(word: &str, condition: &str, kind: HunspellAffixKind) -> bool {
    let Some(atoms) = parse_condition(condition) else {
        return false;
    };
    let characters = word.chars().collect::<Vec<_>>();
    if characters.len() < atoms.len() {
        return false;
    }
    match kind {
        HunspellAffixKind::Prefix => atoms
            .iter()
            .zip(&characters)
            .all(|(atom, character)| atom.matches(*character)),
        HunspellAffixKind::Suffix => atoms
            .iter()
            .rev()
            .zip(characters.iter().rev())
            .all(|(atom, character)| atom.matches(*character)),
    }
}

#[derive(Debug)]
enum ConditionAtom {
    Any,
    Literal(char),
    Set { chars: BTreeSet<char>, negate: bool },
}

impl ConditionAtom {
    fn matches(&self, character: char) -> bool {
        match self {
            Self::Any => true,
            Self::Literal(expected) => *expected == character,
            Self::Set { chars, negate } => chars.contains(&character) != *negate,
        }
    }
}

fn parse_condition(condition: &str) -> Option<Vec<ConditionAtom>> {
    let mut characters = condition.chars().peekable();
    let mut atoms = Vec::new();
    while let Some(character) = characters.next() {
        match character {
            '^' if atoms.is_empty() => {}
            '.' => atoms.push(ConditionAtom::Any),
            '[' => {
                let mut members = Vec::new();
                let mut closed = false;
                for member in characters.by_ref() {
                    if member == ']' {
                        closed = true;
                        break;
                    }
                    members.push(member);
                }
                if !closed || members.is_empty() {
                    return None;
                }
                let negate = members.first() == Some(&'^');
                let chars = members
                    .into_iter()
                    .filter(|member| *member != '^')
                    .collect();
                atoms.push(ConditionAtom::Set { chars, negate });
            }
            ']' => return None,
            literal => atoms.push(ConditionAtom::Literal(literal)),
        }
    }
    (!atoms.is_empty()).then_some(atoms)
}

fn read_bounded(path: &Path, maximum: u64, operation: &'static str) -> Result<Vec<u8>, SimError> {
    let file = fs::File::open(path).map_err(|error| io_error(operation, error))?;
    let metadata = file
        .metadata()
        .map_err(|error| io_error(operation, error))?;
    if metadata.len() > maximum {
        return Err(dataset_error(
            "file Hunspell melampaui batas ukuran yang diizinkan",
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(operation, error))?;
    if bytes.len() as u64 > maximum {
        return Err(dataset_error(
            "file Hunspell melampaui batas ukuran yang diizinkan",
        ));
    }
    Ok(bytes)
}

fn io_error(operation: &'static str, error: std::io::Error) -> SimError {
    SimError::Io {
        operation,
        kind: error.kind(),
    }
}

fn dataset_error(reason: impl Into<String>) -> SimError {
    SimError::CorruptData {
        subsystem: "Hunspell Indonesian",
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_provenance() -> DatasetProvenance {
        DatasetProvenance {
            id: DatasetId::HunspellDictionaries,
            repository_url: "https://example.invalid/test-fixture".to_owned(),
            revision: "test-only".to_owned(),
            license_id: "test-only".to_owned(),
            license_url: "https://example.invalid/test-license".to_owned(),
            attribution: "test fixture, not upstream dictionary".to_owned(),
            data_scope: "unit-test-only".to_owned(),
            file_git_blob_sha1: BTreeMap::new(),
        }
    }

    #[test]
    fn dictionary_parser_respects_long_flags_and_declared_root_count() {
        let roots = parse_dictionary("2\nmakan/ABCD\nminum\n").expect("dictionary valid");
        assert_eq!(
            roots[0],
            (
                2,
                "makan".to_owned(),
                vec!["AB".to_owned(), "CD".to_owned()]
            )
        );
        assert_eq!(roots[1], (3, "minum".to_owned(), Vec::new()));
        assert!(parse_dictionary("2\nmakan/ABC\n").is_err());
        assert!(parse_dictionary("2\nmakan\n").is_err());
    }

    #[test]
    fn affix_parser_applies_supported_direct_prefix_and_suffix_rules() {
        let input = "SET UTF-8\n\
                     FLAG long\n\
                     PFX AB Y 1\n\
                     PFX AB 0 ber .\n\
                     SFX CD N 1\n\
                     SFX CD 0 kan\n";
        let rules = parse_affix(input, &test_provenance()).expect("affix data valid");
        assert_eq!(rules.len(), 2);
        assert!(rules[0].cross_product);
        assert_eq!(apply_rule("makan", &rules[0]).as_deref(), Some("bermakan"));
        assert_eq!(apply_rule("minum", &rules[1]).as_deref(), Some("minumkan"));
        assert_eq!(rules[1].condition, ".");
        assert!(apply_rule("makan", &rules[1]).is_some());
    }

    #[test]
    fn affix_parser_rejects_inconsistent_rule_counts_and_conditions() {
        let malformed = "SET UTF-8\nFLAG long\nPFX AB N 2\nPFX AB 0 ber .\n";
        assert!(parse_affix(malformed, &test_provenance()).is_err());

        let invalid_condition = "SET UTF-8\nFLAG long\nPFX AB N 1\nPFX AB 0 ber [x\n";
        assert!(parse_affix(invalid_condition, &test_provenance()).is_err());
    }
}

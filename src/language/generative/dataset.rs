use super::{GenerativeError, Result};
use crate::language::{DatasetLoader, DatasetProvenance, LicenseRegistry, NusaXSplit};
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashSet},
    fs,
    path::Path,
};
use unicode_normalization::UnicodeNormalization;

const UPSTREAM_TRAIN_PATH: &str = "datasets/sentiment/indonesian/train.csv";
const NEAR_DUPLICATE_JACCARD: f64 = 0.90;
const NEAR_DUPLICATE_LENGTH_RATIO: f64 = 0.85;
const NEAR_DUPLICATE_MIN_CHARS: usize = 50;
const MAX_CORPUS_RECORDS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusRecord {
    pub id: usize,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSplitConfig {
    pub seed: u64,
    pub training_fraction: f64,
    pub validation_fraction: f64,
    pub test_fraction: f64,
}

impl Default for DatasetSplitConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            training_fraction: 0.8,
            validation_fraction: 0.1,
            test_fraction: 0.1,
        }
    }
}

impl DatasetSplitConfig {
    pub fn validate(&self) -> Result<()> {
        let fractions = [
            self.training_fraction,
            self.validation_fraction,
            self.test_fraction,
        ];
        if fractions
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || (fractions.iter().sum::<f64>() - 1.0).abs() > 1e-9
        {
            return Err(GenerativeError::InvalidConfig {
                parameter: "dataset_split",
                reason: "proporsi train/validation/test harus positif dan berjumlah 1",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitManifest {
    pub seed: u64,
    pub training_fraction_ppm: u32,
    pub validation_fraction_ppm: u32,
    pub test_fraction_ppm: u32,
    pub exact_duplicates_removed: usize,
    pub near_duplicates_removed: usize,
    pub source_records: usize,
    pub training_ids: Vec<usize>,
    pub validation_ids: Vec<usize>,
    pub test_ids: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusSourceManifest {
    pub dataset_id: String,
    pub repository_url: String,
    pub revision: String,
    pub license_id: String,
    pub license_url: String,
    pub attribution: String,
    pub data_scope: String,
    pub source_split: String,
    pub source_file_path: String,
    pub source_file_git_blob_sha1: String,
    pub source_file_sha256: String,
    pub source_file_size_bytes: u64,
    pub text_field: String,
    pub excluded_sources: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusSplit {
    pub training: Vec<CorpusRecord>,
    pub validation: Vec<CorpusRecord>,
    pub test: Vec<CorpusRecord>,
    pub manifest: CorpusManifest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifest {
    pub format_version: u32,
    pub source: CorpusSourceManifest,
    pub split: SplitManifest,
    pub training_characters: usize,
    pub validation_characters: usize,
    pub test_characters: usize,
    pub normalization: String,
    pub deduplication: String,
}

impl CorpusSplit {
    /// Memuat hanya NusaX Bahasa Indonesia split train melalui loader dan registry Shella.
    /// Split validasi/uji upstream sengaja tidak diberikan ke jalur gradient.
    pub fn load_nusax_training(
        csv_path: impl AsRef<Path>,
        registry: &LicenseRegistry,
        split_config: DatasetSplitConfig,
    ) -> Result<Self> {
        split_config.validate()?;
        let csv_path = csv_path.as_ref();
        let corpus = DatasetLoader::load_nusax_sentiment_csv(
            csv_path,
            "indonesian",
            NusaXSplit::Train,
            registry,
        )
        .map_err(GenerativeError::Dataset)?;
        let source_provenance = corpus.source.clone();
        if corpus.source.license_id != "CC BY-SA 4.0" {
            return Err(GenerativeError::InvalidInput {
                parameter: "dataset_license",
                reason: "lisensi NusaX tidak cocok dengan lisensi dataset terverifikasi",
            });
        }
        if corpus.examples.len() > MAX_CORPUS_RECORDS {
            return Err(GenerativeError::InvalidInput {
                parameter: "dataset",
                reason: "jumlah record melebihi batas 100000",
            });
        }

        let mut records = corpus
            .examples
            .into_iter()
            .map(|example| {
                Ok(CorpusRecord {
                    id: example.id,
                    text: normalize_training_text(&example.text)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        records.retain(|record| !record.text.is_empty());
        records.sort_by_key(|record| record.id);

        let source_records = records.len();
        let (unique, exact_duplicates_removed) = remove_exact_duplicates(records);
        let (mut unique, near_duplicates_removed) = remove_near_duplicates(unique);
        if unique.len() < 3 {
            return Err(GenerativeError::InvalidInput {
                parameter: "dataset",
                reason: "setelah deduplikasi dibutuhkan sedikitnya 3 dokumen",
            });
        }
        let ids = unique
            .iter()
            .map(|record| record.id)
            .collect::<HashSet<_>>();
        if ids.len() != unique.len() {
            return Err(GenerativeError::InvalidInput {
                parameter: "dataset",
                reason: "ID sumber duplikat setelah validasi",
            });
        }
        let mut split = split_records(&mut unique, &split_config)?;

        let source_file_git_blob_sha1 = source_provenance
            .file_git_blob_sha1
            .get(UPSTREAM_TRAIN_PATH)
            .cloned()
            .ok_or(GenerativeError::InvalidInput {
                parameter: "dataset_provenance",
                reason: "hash Git upstream untuk file NusaX train tidak tersedia",
            })?;
        let bytes = fs::read(csv_path).map_err(|error| GenerativeError::Io {
            operation: "baca NusaX train untuk fingerprint",
            kind: error.kind(),
        })?;
        let source_file_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let source = CorpusSourceManifest {
            dataset_id: dataset_id_name(&source_provenance),
            repository_url: source_provenance.repository_url,
            revision: source_provenance.revision,
            license_id: source_provenance.license_id,
            license_url: source_provenance.license_url,
            attribution: source_provenance.attribution,
            data_scope: source_provenance.data_scope,
            source_split: "train (NusaX resmi; hanya split train dibaca untuk model)".to_owned(),
            source_file_path: UPSTREAM_TRAIN_PATH.to_owned(),
            source_file_git_blob_sha1,
            source_file_sha256,
            source_file_size_bytes: bytes.len() as u64,
            text_field: "text".to_owned(),
            excluded_sources: vec![
                "NusaX-MT kolom indonesian tidak dipakai karena isinya identik dengan NusaX sentiment per split".to_owned(),
                "NusaX valid/test upstream tidak dipakai; split model dibentuk dari NusaX train saja".to_owned(),
                "IndoSum tidak dipakai untuk training neural karena record menautkan artikel berita sumber dan hak teks artikel tidak diverifikasi terpisah".to_owned(),
                "WordNet Bahasa tidak dipakai sebagai prosa pretraining".to_owned(),
            ],
        };
        let manifest = make_manifest(
            source,
            &mut split,
            source_records,
            exact_duplicates_removed,
            near_duplicates_removed,
        );
        let result = Self {
            training: split.training,
            validation: split.validation,
            test: split.test,
            manifest,
        };
        result.validate_disjoint_ids()?;
        Ok(result)
    }

    /// Memuat split korpus berlisensi yang sudah disiapkan, tanpa membentuk ulang split.
    /// Manifest, checksum provenance, jumlah karakter, dan keunikan ID divalidasi sebelum training.
    pub fn load_prepared_json(json_path: impl AsRef<Path>) -> Result<Self> {
        let json_path = json_path.as_ref();
        let metadata = fs::metadata(json_path)?;
        if metadata.len() > 512 * 1024 * 1024 {
            return Err(GenerativeError::InvalidInput {
                parameter: "prepared_corpus",
                reason: "berkas korpus JSON melebihi batas 512 MiB",
            });
        }
        let bytes = fs::read(json_path)?;
        let corpus: Self = serde_json::from_slice(&bytes)?;
        if corpus.manifest.format_version != 1 {
            return Err(GenerativeError::InvalidInput {
                parameter: "prepared_corpus",
                reason: "format_version korpus harus 1",
            });
        }
        let source = &corpus.manifest.source;
        if source.dataset_id.trim().is_empty()
            || source.repository_url.trim().is_empty()
            || source.license_id.trim().is_empty()
            || !source.license_url.starts_with("https://")
            || source.source_file_sha256.len() != 64
            || source.source_file_git_blob_sha1.len() != 40
            || source.source_file_size_bytes == 0
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "prepared_corpus_provenance",
                reason: "metadata sumber, lisensi, atau checksum tidak lengkap",
            });
        }
        let provenance_relative_path = Path::new(&source.source_file_path);
        if provenance_relative_path.is_absolute()
            || provenance_relative_path
                .components()
                .any(|component| component == std::path::Component::ParentDir)
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "prepared_corpus_provenance",
                reason: "path file provenance harus relatif terhadap folder korpus",
            });
        }
        let local_provenance_path = json_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(provenance_relative_path);
        let provenance_path = if local_provenance_path.is_file() {
            local_provenance_path
        } else {
            provenance_relative_path.to_path_buf()
        };
        let provenance_bytes = fs::read(provenance_path)?;
        let mut git_blob_hasher = Sha1::new();
        git_blob_hasher.update(format!("blob {}\0", provenance_bytes.len()).as_bytes());
        git_blob_hasher.update(&provenance_bytes);
        let actual_git_blob_sha1 = format!("{:x}", git_blob_hasher.finalize());
        let actual_sha256 = format!("{:x}", Sha256::digest(&provenance_bytes));
        if provenance_bytes.len() as u64 != source.source_file_size_bytes
            || actual_git_blob_sha1 != source.source_file_git_blob_sha1
            || actual_sha256 != source.source_file_sha256
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "prepared_corpus_provenance",
                reason: "ukuran, SHA-256, atau Git blob file provenance tidak cocok",
            });
        }
        corpus.validate_disjoint_ids()?;

        let manifest_split = &corpus.manifest.split;
        if manifest_split.training_fraction_ppm == 0
            || manifest_split.validation_fraction_ppm == 0
            || manifest_split.test_fraction_ppm == 0
            || manifest_split.training_fraction_ppm
                + manifest_split.validation_fraction_ppm
                + manifest_split.test_fraction_ppm
                != 1_000_000
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "prepared_corpus_split",
                reason: "proporsi split manifest harus positif dan berjumlah satu juta ppm",
            });
        }
        let ids_match = |records: &[CorpusRecord], listed_ids: &[usize]| {
            let actual = records
                .iter()
                .map(|record| record.id)
                .collect::<HashSet<_>>();
            let listed = listed_ids.iter().copied().collect::<HashSet<_>>();
            actual.len() == records.len() && listed.len() == listed_ids.len() && actual == listed
        };
        if !ids_match(&corpus.training, &manifest_split.training_ids)
            || !ids_match(&corpus.validation, &manifest_split.validation_ids)
            || !ids_match(&corpus.test, &manifest_split.test_ids)
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "prepared_corpus_split",
                reason: "ID record tidak cocok dengan ID yang dicatat pada manifest split",
            });
        }
        let all_records = corpus
            .training
            .iter()
            .chain(&corpus.validation)
            .chain(&corpus.test)
            .collect::<Vec<_>>();
        let expected_source_records = all_records
            .len()
            .saturating_add(manifest_split.exact_duplicates_removed)
            .saturating_add(manifest_split.near_duplicates_removed);
        if expected_source_records != manifest_split.source_records {
            return Err(GenerativeError::InvalidInput {
                parameter: "prepared_corpus_split",
                reason: "jumlah sumber tidak konsisten dengan record dan deduplikasi manifest",
            });
        }
        for record in &all_records {
            if record.text.is_empty()
                || record.text.len() > 1_000_000
                || normalize_training_text(&record.text)? != record.text
            {
                return Err(GenerativeError::InvalidInput {
                    parameter: "prepared_corpus_text",
                    reason: "teks harus nonkosong, maksimal 1 MiB, dan sudah dinormalisasi NFC/lowercase",
                });
            }
        }
        let character_count = |records: &[CorpusRecord]| {
            records
                .iter()
                .map(|record| record.text.chars().count())
                .sum::<usize>()
        };
        if character_count(&corpus.training) != corpus.manifest.training_characters
            || character_count(&corpus.validation) != corpus.manifest.validation_characters
            || character_count(&corpus.test) != corpus.manifest.test_characters
        {
            return Err(GenerativeError::InvalidInput {
                parameter: "prepared_corpus_token_count",
                reason: "jumlah karakter/token teks tidak cocok dengan manifest",
            });
        }
        Ok(corpus)
    }

    pub fn validate_disjoint_ids(&self) -> Result<()> {
        let mut seen = HashSet::new();
        for record in self
            .training
            .iter()
            .chain(&self.validation)
            .chain(&self.test)
        {
            if !seen.insert(record.id) {
                return Err(GenerativeError::InvalidInput {
                    parameter: "dataset_split",
                    reason: "ID dokumen muncul pada lebih dari satu split",
                });
            }
        }
        if self.training.is_empty() || self.validation.is_empty() || self.test.is_empty() {
            return Err(GenerativeError::InvalidInput {
                parameter: "dataset_split",
                reason: "setiap split harus memuat dokumen",
            });
        }
        Ok(())
    }
}

fn dataset_id_name(source: &DatasetProvenance) -> String {
    format!("{:?}", source.id)
}

fn normalize_training_text(text: &str) -> Result<String> {
    if text.len() > 1_000_000 {
        return Err(GenerativeError::InvalidInput {
            parameter: "dataset_text",
            reason: "dokumen melebihi batas 1 MiB",
        });
    }
    if text
        .chars()
        .any(|character| character.is_control() && !character.is_whitespace())
    {
        return Err(GenerativeError::InvalidInput {
            parameter: "dataset_text",
            reason: "dokumen memuat karakter kontrol non-whitespace",
        });
    }
    Ok(text
        .nfc()
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" "))
}

fn remove_exact_duplicates(records: Vec<CorpusRecord>) -> (Vec<CorpusRecord>, usize) {
    let mut seen = HashSet::new();
    let mut unique = Vec::with_capacity(records.len());
    let mut removed = 0;
    for record in records {
        if seen.insert(record.text.clone()) {
            unique.push(record);
        } else {
            removed += 1;
        }
    }
    (unique, removed)
}

fn trigrams(text: &str) -> BTreeSet<String> {
    let characters = text.chars().collect::<Vec<_>>();
    characters
        .windows(3)
        .map(|window| window.iter().collect())
        .collect()
}

fn is_near_duplicate(left: &str, left_grams: &BTreeSet<String>, right: &str) -> bool {
    let left_length = left.chars().count();
    let right_length = right.chars().count();
    if left_length < NEAR_DUPLICATE_MIN_CHARS || right_length < NEAR_DUPLICATE_MIN_CHARS {
        return false;
    }
    let length_ratio = left_length.min(right_length) as f64 / left_length.max(right_length) as f64;
    if length_ratio < NEAR_DUPLICATE_LENGTH_RATIO {
        return false;
    }
    let right_grams = trigrams(right);
    let intersection = left_grams.intersection(&right_grams).count();
    let union = left_grams.union(&right_grams).count();
    union > 0 && intersection as f64 / union as f64 >= NEAR_DUPLICATE_JACCARD
}

fn remove_near_duplicates(records: Vec<CorpusRecord>) -> (Vec<CorpusRecord>, usize) {
    let mut retained: Vec<CorpusRecord> = Vec::with_capacity(records.len());
    let mut retained_grams = Vec::with_capacity(records.len());
    let mut removed = 0;
    for record in records {
        let grams = trigrams(&record.text);
        if retained
            .iter()
            .zip(&retained_grams)
            .any(|(prior, prior_grams)| is_near_duplicate(&prior.text, prior_grams, &record.text))
        {
            removed += 1;
        } else {
            retained.push(record);
            retained_grams.push(grams);
        }
    }
    (retained, removed)
}

fn split_records(
    records: &mut Vec<CorpusRecord>,
    config: &DatasetSplitConfig,
) -> Result<RecordSplit> {
    deterministic_shuffle(records, config.seed);
    let total = records.len();
    let training_count = ((total as f64) * config.training_fraction).floor() as usize;
    let validation_count = ((total as f64) * config.validation_fraction).floor() as usize;
    let test_count = total.saturating_sub(training_count + validation_count);
    if training_count == 0
        || validation_count == 0
        || test_count == 0
        || training_count + validation_count + test_count != total
    {
        return Err(GenerativeError::InvalidInput {
            parameter: "dataset_split",
            reason: "rasio yang dipilih menghasilkan split kosong",
        });
    }
    let mut records = std::mem::take(records).into_iter();
    let training = records.by_ref().take(training_count).collect();
    let validation = records.by_ref().take(validation_count).collect();
    let test = records.collect();
    Ok(RecordSplit {
        training,
        validation,
        test,
        manifest: SplitManifest {
            seed: config.seed,
            training_fraction_ppm: (config.training_fraction * 1_000_000.0).round() as u32,
            validation_fraction_ppm: (config.validation_fraction * 1_000_000.0).round() as u32,
            test_fraction_ppm: (config.test_fraction * 1_000_000.0).round() as u32,
            exact_duplicates_removed: 0,
            near_duplicates_removed: 0,
            source_records: total,
            training_ids: Vec::new(),
            validation_ids: Vec::new(),
            test_ids: Vec::new(),
        },
    })
}

struct RecordSplit {
    training: Vec<CorpusRecord>,
    validation: Vec<CorpusRecord>,
    test: Vec<CorpusRecord>,
    manifest: SplitManifest,
}

fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    for index in (1..items.len()).rev() {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        items.swap(index, (value as usize) % (index + 1));
    }
}

fn make_manifest(
    source: CorpusSourceManifest,
    split: &mut RecordSplit,
    source_records: usize,
    exact_duplicates_removed: usize,
    near_duplicates_removed: usize,
) -> CorpusManifest {
    let training_characters = split
        .training
        .iter()
        .map(|record| record.text.chars().count())
        .sum();
    let validation_characters = split
        .validation
        .iter()
        .map(|record| record.text.chars().count())
        .sum();
    let test_characters = split
        .test
        .iter()
        .map(|record| record.text.chars().count())
        .sum();
    split.manifest.source_records = source_records;
    split.manifest.exact_duplicates_removed = exact_duplicates_removed;
    split.manifest.near_duplicates_removed = near_duplicates_removed;
    split.manifest.training_ids = split.training.iter().map(|record| record.id).collect();
    split.manifest.validation_ids = split.validation.iter().map(|record| record.id).collect();
    split.manifest.test_ids = split.test.iter().map(|record| record.id).collect();
    CorpusManifest {
        format_version: 1,
        source,
        split: split.manifest.clone(),
        training_characters,
        validation_characters,
        test_characters,
        normalization: "UTF-8 verified by DatasetLoader; NFC; lowercase; collapse whitespace; preserve punctuation and digits".to_owned(),
        deduplication: "Exact after normalization; near duplicate removed when text has >=50 Unicode scalars, length ratio >=0.85 and character-trigram Jaccard >=0.90".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_split_has_disjoint_ids_and_expected_sizes() {
        let mut records = (0..500)
            .map(|id| CorpusRecord {
                id,
                text: format!("kalimat uji unik nomor {id}"),
            })
            .collect::<Vec<_>>();
        let first = split_records(&mut records, &DatasetSplitConfig::default()).unwrap();
        let mut records = (0..500)
            .map(|id| CorpusRecord {
                id,
                text: format!("kalimat uji unik nomor {id}"),
            })
            .collect::<Vec<_>>();
        let second = split_records(&mut records, &DatasetSplitConfig::default()).unwrap();
        assert_eq!(first.training, second.training);
        assert_eq!(first.validation, second.validation);
        assert_eq!(first.test, second.test);
        assert_eq!(first.training.len(), 400);
        assert_eq!(first.validation.len(), 50);
        assert_eq!(first.test.len(), 50);
    }

    #[test]
    fn near_duplicate_filter_uses_documented_conservative_threshold() {
        let exact = "contoh ulasan ini memiliki cukup banyak karakter untuk diperiksa";
        let near = "contoh ulasan ini memiliki cukup banyak karakter untuk diperiksa!";
        assert!(is_near_duplicate(exact, &trigrams(exact), near));
        assert!(!is_near_duplicate("pendek", &trigrams("pendek"), "pendek"));
    }
}

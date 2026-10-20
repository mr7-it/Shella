use super::{DatasetId, DatasetProvenance, LicenseRegistry, datasets::dataset_error};
use crate::{SimError, language::normalize_text};
use flate2::read::GzDecoder;
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::Path,
};
use tar::Archive;

const MAX_INDOSUM_ARCHIVE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_INDOSUM_SPLIT_BYTES: u64 = 128 * 1024 * 1024;
const MAX_INDOSUM_RECORD_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndoSumPartition {
    Train,
    Development,
    Test,
}

impl IndoSumPartition {
    const fn prefix(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Development => "dev",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndoSumArticle {
    pub id: String,
    pub paragraphs: Vec<Vec<Vec<String>>>,
    pub summary: Vec<Vec<String>>,
    pub gold_labels: Vec<Vec<bool>>,
    pub category: String,
    pub source: String,
    pub source_url: String,
}

impl IndoSumArticle {
    /// Returns the dataset's human-annotated extractive oracle, not a model output.
    pub fn gold_extractive_summary(&self) -> Vec<Vec<String>> {
        self.paragraphs
            .iter()
            .zip(&self.gold_labels)
            .flat_map(|(paragraph, labels)| {
                paragraph
                    .iter()
                    .zip(labels)
                    .filter(|(_, selected)| **selected)
                    .map(|(sentence, _)| sentence.clone())
            })
            .collect()
    }

    /// Computes token-level ROUGE-1 against the supplied reference summary.
    /// Tokens are normalized with Shella's Unicode/case normalization only.
    pub fn evaluate_summary(&self, candidate: &[Vec<String>]) -> Result<IndoSumRouge1, SimError> {
        let reference_counts = token_counts(self.summary.iter().flatten())?;
        let candidate_counts = token_counts(candidate.iter().flatten())?;
        let reference_tokens = reference_counts.values().sum::<usize>();
        let candidate_tokens = candidate_counts.values().sum::<usize>();
        if reference_tokens == 0 || candidate_tokens == 0 {
            return Err(SimError::InvalidInput {
                parameter: "summary",
                reason: "reference dan kandidat harus memiliki token",
            });
        }
        let overlap = candidate_counts
            .iter()
            .map(|(token, count)| count.min(reference_counts.get(token).unwrap_or(&0)))
            .sum::<usize>();
        let precision = overlap as f64 / candidate_tokens as f64;
        let recall = overlap as f64 / reference_tokens as f64;
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Ok(IndoSumRouge1 {
            overlap,
            candidate_tokens,
            reference_tokens,
            precision,
            recall,
            f1,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndoSumRouge1 {
    pub overlap: usize,
    pub candidate_tokens: usize,
    pub reference_tokens: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndoSumCorpus {
    pub source: DatasetProvenance,
    pub fold: u8,
    pub partition: IndoSumPartition,
    pub articles: Vec<IndoSumArticle>,
}

impl super::DatasetLoader {
    /// Load one verified five-fold IndoSum JSONL split from the original tar.gz.
    ///
    /// The source archive stays compressed on disk; only the selected JSONL
    /// member is parsed into memory.
    pub fn load_indosum_archive(
        path: impl AsRef<Path>,
        fold: u8,
        partition: IndoSumPartition,
        registry: &LicenseRegistry,
    ) -> Result<IndoSumCorpus, SimError> {
        if !(1..=5).contains(&fold) {
            return Err(SimError::InvalidInput {
                parameter: "fold",
                reason: "IndoSum hanya memiliki fold 1 sampai 5",
            });
        }
        let path = path.as_ref();
        let mut file = File::open(path)
            .map_err(|error| super::datasets::io_error("buka arsip IndoSum", error))?;
        let length = file
            .metadata()
            .map_err(|error| super::datasets::io_error("baca metadata arsip IndoSum", error))?
            .len();
        if length > MAX_INDOSUM_ARCHIVE_BYTES {
            return Err(dataset_error("arsip IndoSum melampaui batas 128 MiB"));
        }
        registry.verify_file_reader(DatasetId::IndoSum, "indosum.tar.gz", length, &mut file)?;
        let source = registry.provenance(DatasetId::IndoSum)?;
        let member_name = format!("indosum/{}.{fold:02}.jsonl", partition.prefix());
        file.seek(SeekFrom::Start(0))
            .map_err(|error| super::datasets::io_error("kembali ke awal arsip IndoSum", error))?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        let entries = archive
            .entries()
            .map_err(|error| dataset_error(format!("arsip IndoSum rusak: {error}")))?;
        for entry in entries {
            let mut entry = entry
                .map_err(|error| dataset_error(format!("header tar IndoSum invalid: {error}")))?;
            let path = entry
                .path()
                .map_err(|error| dataset_error(format!("path tar IndoSum invalid: {error}")))?;
            if path.to_str() != Some(member_name.as_str()) {
                continue;
            }
            if entry.size() > MAX_INDOSUM_SPLIT_BYTES {
                return Err(dataset_error("split IndoSum melampaui batas 128 MiB"));
            }
            let articles = parse_jsonl(&mut entry)?;
            return Ok(IndoSumCorpus {
                source,
                fold,
                partition,
                articles,
            });
        }
        Err(dataset_error(
            "split JSONL tidak ditemukan dalam arsip IndoSum",
        ))
    }
}

fn parse_jsonl(reader: impl Read) -> Result<Vec<IndoSumArticle>, SimError> {
    let mut reader = BufReader::new(reader);
    let mut articles = Vec::new();
    let mut ids = HashSet::new();
    let mut line = String::new();
    let mut line_number = 0;
    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|error| super::datasets::io_error("baca JSONL IndoSum", error))?;
        if bytes_read == 0 {
            break;
        }
        line_number += 1;
        if line.len() > MAX_INDOSUM_RECORD_BYTES {
            return Err(dataset_error(format!(
                "record IndoSum pada baris {line_number} melampaui 4 MiB"
            )));
        }
        if line.trim().is_empty() {
            return Err(dataset_error(format!(
                "baris kosong ditemukan pada JSONL IndoSum ({line_number})"
            )));
        }
        let article: IndoSumArticle = serde_json::from_str(&line).map_err(|error| {
            dataset_error(format!(
                "record JSONL IndoSum invalid pada baris {line_number}: {error}"
            ))
        })?;
        validate_article(&article, line_number)?;
        if !ids.insert(article.id.clone()) {
            return Err(dataset_error(format!(
                "ID artikel IndoSum duplikat pada baris {line_number}"
            )));
        }
        articles.push(article);
    }
    if articles.is_empty() {
        return Err(dataset_error("split JSONL IndoSum kosong"));
    }
    Ok(articles)
}

fn validate_article(article: &IndoSumArticle, line_number: usize) -> Result<(), SimError> {
    if article.id.trim().is_empty()
        || article.category.trim().is_empty()
        || article.source.trim().is_empty()
        || article.source_url.trim().is_empty()
        || article.paragraphs.is_empty()
        || article.paragraphs.len() != article.gold_labels.len()
        || article.summary.is_empty()
    {
        return Err(dataset_error(format!(
            "metadata, paragraphs, summary, atau gold_labels tidak valid pada baris {line_number}"
        )));
    }
    for (paragraph, labels) in article.paragraphs.iter().zip(&article.gold_labels) {
        if paragraph.is_empty() || paragraph.len() != labels.len() {
            return Err(dataset_error(format!(
                "jumlah kalimat dan label IndoSum tidak cocok pada baris {line_number}"
            )));
        }
        if paragraph.iter().any(|sentence| sentence.is_empty()) {
            return Err(dataset_error(format!(
                "kalimat kosong ditemukan pada baris {line_number}"
            )));
        }
    }
    if article
        .summary
        .iter()
        .any(|sentence| sentence.is_empty() || sentence.iter().any(String::is_empty))
    {
        return Err(dataset_error(format!(
            "kalimat summary IndoSum kosong pada baris {line_number}"
        )));
    }
    Ok(())
}

fn token_counts<'a>(
    tokens: impl Iterator<Item = &'a String>,
) -> Result<HashMap<String, usize>, SimError> {
    let mut counts = HashMap::new();
    for token in tokens {
        let normalized = normalize_text(token)?;
        if normalized.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "summary_token",
                reason: "token summary tidak boleh kosong",
            });
        }
        let count = counts.entry(normalized).or_insert(0_usize);
        *count = count.checked_add(1).ok_or(SimError::CounterOverflow {
            counter: "token summary",
        })?;
    }
    Ok(counts)
}

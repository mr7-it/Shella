use super::DatasetId;
use crate::{SimError, checkpoint::atomic_write};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{self, Read},
    path::{Component, Path, PathBuf},
    sync::OnceLock,
};

const FORMAT_VERSION: u32 = 1;
const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const BUNDLED_MANIFEST: &str = include_str!("../../../data/language-integrity.json");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetIntegrityManifest {
    pub format_version: u32,
    pub files: BTreeMap<String, DatasetFileIntegrity>,
    /// Pinned dataset inputs validated on demand but not bundled in the directory.
    #[serde(default)]
    pub external_files: BTreeMap<String, DatasetFileIntegrity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetFileIntegrity {
    pub size_bytes: u64,
    pub sha256: String,
    pub format: DatasetFileFormat,
    pub encoding: DatasetEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetFileFormat {
    Csv,
    Json,
    Conllu,
    Text,
    TarGzip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetEncoding {
    Utf8,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetIntegrityReport {
    pub files_verified: usize,
    pub total_bytes: u64,
}

impl DatasetIntegrityManifest {
    /// Read the version-controlled manifest bundled with Shella.
    pub fn bundled() -> Result<Self, SimError> {
        let manifest: Self =
            serde_json::from_str(BUNDLED_MANIFEST).map_err(|error| corrupt(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Create a manifest from the files currently present under `root`.
    ///
    /// This computes checksums; it does not establish provenance or approve a
    /// dataset revision. Review generated digests against the pinned source.
    pub fn create(root: impl AsRef<Path>) -> Result<Self, SimError> {
        let root = root.as_ref();
        validate_root(root)?;
        let mut paths = Vec::new();
        collect_files(root, Path::new(""), &mut paths)?;
        let mut files = BTreeMap::new();
        for relative_path in paths {
            let path = root.join(&relative_path);
            let bytes = read_bounded(&path)?;
            validate_contents(&relative_path, &bytes)?;
            files.insert(
                relative_key(&relative_path)?,
                DatasetFileIntegrity {
                    size_bytes: bytes.len() as u64,
                    sha256: sha256(&bytes),
                    format: classify_format(&relative_path)?,
                    encoding: classify_encoding(&relative_path)?,
                },
            );
        }
        let manifest = Self {
            format_version: FORMAT_VERSION,
            files,
            external_files: BTreeMap::new(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Verify every file and reject both missing and untracked files.
    pub fn verify_directory(
        &self,
        root: impl AsRef<Path>,
    ) -> Result<DatasetIntegrityReport, SimError> {
        self.validate()?;
        let root = root.as_ref();
        validate_root(root)?;
        let mut actual_paths = Vec::new();
        collect_files(root, Path::new(""), &mut actual_paths)?;
        let actual = actual_paths
            .iter()
            .map(|path| relative_key(path))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let expected = self.files.keys().cloned().collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(corrupt(
                "daftar file dataset berbeda dari manifest (file hilang atau tak terduga)",
            ));
        }

        let mut total_bytes = 0_u64;
        for (relative, expected_file) in &self.files {
            let path = root.join(relative);
            let bytes = read_bounded(&path)?;
            verify_contents(Path::new(relative), expected_file, &bytes)?;
            total_bytes =
                total_bytes
                    .checked_add(bytes.len() as u64)
                    .ok_or(SimError::CounterOverflow {
                        counter: "ukuran dataset",
                    })?;
        }
        Ok(DatasetIntegrityReport {
            files_verified: self.files.len(),
            total_bytes,
        })
    }

    /// Verify exact bytes against a bundled or externally pinned source entry.
    pub fn verify_source_bytes(
        &self,
        source: DatasetId,
        source_path: &str,
        bytes: &[u8],
    ) -> Result<(), SimError> {
        let (expected, path) = self.source_entry(source, source_path).ok_or_else(|| {
            corrupt(format!(
                "SHA-256 tidak tercatat untuk {source:?}:{source_path}"
            ))
        })?;
        verify_contents(Path::new(&path), expected, bytes)
    }

    /// Save the manifest via an atomic replace.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SimError> {
        self.validate()?;
        let bytes =
            serde_json::to_vec_pretty(self).map_err(|error| SimError::SerializationFailed {
                subsystem: "dataset integrity manifest",
                reason: error.to_string(),
            })?;
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(corrupt("manifest melebihi batas ukuran 16 MiB"));
        }
        atomic_write(
            path.as_ref(),
            &bytes,
            "dataset integrity manifest",
            "simpan manifest integritas atomik",
        )
    }

    /// Load and validate an on-disk manifest.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SimError> {
        let path = path.as_ref();
        let bytes = read_bounded_limit(path, MAX_MANIFEST_BYTES, "baca manifest dataset")?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .map_err(|error| corrupt(format!("format manifest invalid: {error}")))?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<(), SimError> {
        if self.format_version != FORMAT_VERSION {
            return Err(SimError::UnsupportedDataVersion {
                subsystem: "dataset integrity manifest",
                version: self.format_version,
            });
        }
        if self.files.is_empty() {
            return Err(corrupt("manifest integritas kosong"));
        }
        for (path, entry) in &self.files {
            validate_entry(path, entry)?;
        }
        for (path, entry) in &self.external_files {
            validate_entry(path, entry)?;
        }
        Ok(())
    }

    fn source_entry(
        &self,
        source: DatasetId,
        source_path: &str,
    ) -> Option<(&DatasetFileIntegrity, String)> {
        if let Some(asset_path) = bundled_asset_path(source, source_path) {
            return self.files.get(&asset_path).map(|entry| (entry, asset_path));
        }
        let external_path = external_asset_path(source, source_path)?;
        self.external_files
            .get(&external_path)
            .map(|entry| (entry, external_path))
    }
}

fn validate_entry(path: &str, entry: &DatasetFileIntegrity) -> Result<(), SimError> {
    validate_relative_path(path)?;
    let path = Path::new(path);
    if entry.size_bytes > MAX_FILE_BYTES
        || entry.sha256.len() != 64
        || !entry
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        || classify_format(path)? != entry.format
        || classify_encoding(path)? != entry.encoding
    {
        return Err(corrupt("metadata file manifest tidak valid"));
    }
    Ok(())
}

/// Re-verify bytes during normal dataset loading in addition to upstream Git hashes.
pub(super) fn verify_source_sha256(
    source: DatasetId,
    source_path: &str,
    size_bytes: u64,
    digest: &str,
) -> Result<(), SimError> {
    let manifest = bundled_manifest()?;
    let (expected, asset_path) = manifest.source_entry(source, source_path).ok_or_else(|| {
        corrupt(format!(
            "SHA-256 tidak tercatat untuk {source:?}:{source_path}"
        ))
    })?;
    if expected.size_bytes != size_bytes || expected.sha256 != digest {
        return Err(corrupt(format!(
            "ukuran atau SHA-256 dataset tidak cocok untuk {asset_path}"
        )));
    }
    Ok(())
}

fn bundled_manifest() -> Result<&'static DatasetIntegrityManifest, SimError> {
    static MANIFEST: OnceLock<Result<DatasetIntegrityManifest, String>> = OnceLock::new();
    match MANIFEST.get_or_init(|| {
        let manifest: DatasetIntegrityManifest =
            serde_json::from_str(BUNDLED_MANIFEST).map_err(|error| error.to_string())?;
        manifest.validate().map_err(|error| error.to_string())?;
        Ok(manifest)
    }) {
        Ok(manifest) => Ok(manifest),
        Err(reason) => Err(corrupt(format!(
            "manifest integritas bawaan invalid: {reason}"
        ))),
    }
}

fn bundled_asset_path(source: DatasetId, source_path: &str) -> Option<String> {
    match source {
        DatasetId::StopwordsIsoId if source_path == "stopwords-id.json" => {
            Some("stopwords-id.json".to_owned())
        }
        DatasetId::UdIndonesianPud if source_path == "id_pud-ud-test.conllu" => {
            Some("ud-indonesian-pud/id_pud-ud-test.conllu".to_owned())
        }
        DatasetId::NusaX if source_path == "datasets/LICENSE" => Some("nusax/LICENSE".to_owned()),
        DatasetId::NusaX => {
            let path = source_path.strip_prefix("datasets/")?;
            let valid = if path.starts_with("mt/") {
                matches!(
                    path,
                    "mt/README.md" | "mt/train.csv" | "mt/valid.csv" | "mt/test.csv"
                )
            } else if path.starts_with("sentiment/") {
                if path == "sentiment/README.md" {
                    true
                } else {
                    let mut parts = path.split('/');
                    let _ = parts.next();
                    let language = parts.next()?;
                    let file = parts.next()?;
                    parts.next().is_none()
                        && [
                            "acehnese",
                            "balinese",
                            "banjarese",
                            "buginese",
                            "english",
                            "indonesian",
                            "javanese",
                            "madurese",
                            "minangkabau",
                            "ngaju",
                            "sundanese",
                            "toba_batak",
                        ]
                        .contains(&language)
                        && ["train.csv", "valid.csv", "test.csv"].contains(&file)
                }
            } else if path.starts_with("lexicon/") {
                let language = path.strip_prefix("lexicon/")?;
                [
                    "acehnese.csv",
                    "balinese.csv",
                    "banjarese.csv",
                    "buginese.csv",
                    "english.csv",
                    "javanese.csv",
                    "madurese.csv",
                    "minangkabau.csv",
                    "ngaju.csv",
                    "sundanese.csv",
                    "toba_batak.csv",
                ]
                .contains(&language)
            } else {
                false
            };
            valid.then(|| format!("nusax/{path}"))
        }
        DatasetId::IndoSum => match source_path {
            "indosum.tar.gz" => Some("indosum/indosum.tar.gz".to_owned()),
            "README.txt" => Some("indosum/README.txt".to_owned()),
            _ => None,
        },
        DatasetId::WordNetBahasa => match source_path {
            "data/msa-wordnet-r24/wn-msa-all.tab" => {
                Some("wordnet-bahasa/wn-msa-all.tab".to_owned())
            }
            "data/msa-wordnet-r24/wn-ind-def.tab" => {
                Some("wordnet-bahasa/wn-ind-def.tab".to_owned())
            }
            _ => None,
        },
        _ => None,
    }
}

fn external_asset_path(source: DatasetId, source_path: &str) -> Option<String> {
    match (source, source_path) {
        (DatasetId::HunspellDictionaries, "Indonesia.dic") => {
            Some("hunspell/Indonesia.dic".to_owned())
        }
        (DatasetId::HunspellDictionaries, "Indonesia.aff") => {
            Some("hunspell/Indonesia.aff".to_owned())
        }
        _ => None,
    }
}

fn collect_files(root: &Path, relative: &Path, output: &mut Vec<PathBuf>) -> Result<(), SimError> {
    let directory = root.join(relative);
    let entries =
        fs::read_dir(&directory).map_err(|error| io_error("daftar file dataset", error))?;
    let mut entries = entries
        .map(|entry| entry.map_err(|error| io_error("baca entri direktori dataset", error)))
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let file_type = entry
            .file_type()
            .map_err(|error| io_error("baca tipe file dataset", error))?;
        let child = relative.join(entry.file_name());
        if file_type.is_symlink() {
            return Err(corrupt("symlink tidak diizinkan dalam direktori dataset"));
        }
        if file_type.is_dir() {
            collect_files(root, &child, output)?;
        } else if file_type.is_file() {
            output.push(child);
        } else {
            return Err(corrupt(
                "entri bukan file/direktori ditemukan dalam dataset",
            ));
        }
    }
    Ok(())
}

fn validate_root(root: &Path) -> Result<(), SimError> {
    let metadata =
        fs::symlink_metadata(root).map_err(|error| io_error("baca direktori dataset", error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(corrupt("root dataset harus direktori nyata, bukan symlink"));
    }
    Ok(())
}

fn relative_key(path: &Path) -> Result<String, SimError> {
    let path = path
        .to_str()
        .ok_or_else(|| corrupt("path dataset harus valid UTF-8"))?;
    Ok(path.replace('\\', "/"))
}

fn validate_relative_path(path: &str) -> Result<(), SimError> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.to_string_lossy().contains('\\')
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(corrupt(
            "path manifest harus relatif dan tidak boleh traversal",
        ));
    }
    Ok(())
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, SimError> {
    read_bounded_limit(path, MAX_FILE_BYTES, "baca file dataset")
}

fn read_bounded_limit(
    path: &Path,
    maximum: u64,
    operation: &'static str,
) -> Result<Vec<u8>, SimError> {
    let file = File::open(path).map_err(|error| io_error("buka file dataset", error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error("baca metadata file dataset", error))?
        .len();
    if length > maximum {
        return Err(corrupt("file melampaui batas ukuran yang diizinkan"));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(operation, error))?;
    if bytes.len() as u64 != length {
        return Err(corrupt("ukuran file berubah saat dibaca"));
    }
    Ok(bytes)
}

fn verify_contents(
    path: &Path,
    expected: &DatasetFileIntegrity,
    bytes: &[u8],
) -> Result<(), SimError> {
    if bytes.len() as u64 != expected.size_bytes || sha256(bytes) != expected.sha256 {
        return Err(corrupt(format!(
            "ukuran atau SHA-256 tidak cocok untuk {}",
            path.display()
        )));
    }
    validate_contents(path, bytes)
}

fn validate_contents(path: &Path, bytes: &[u8]) -> Result<(), SimError> {
    match classify_encoding(path)? {
        DatasetEncoding::Utf8 => {
            std::str::from_utf8(bytes).map_err(|_| {
                corrupt(format!("file dataset {} bukan UTF-8 valid", path.display()))
            })?;
        }
        DatasetEncoding::Binary => {
            if !bytes.starts_with(&[0x1f, 0x8b]) {
                return Err(corrupt(format!(
                    "magic gzip file dataset {} tidak valid",
                    path.display()
                )));
            }
        }
    }
    match classify_format(path)? {
        DatasetFileFormat::Csv => {
            let mut reader = csv::ReaderBuilder::new().flexible(false).from_reader(bytes);
            for record in reader.records() {
                record.map_err(|error| {
                    corrupt(format!("CSV dataset {} rusak: {error}", path.display()))
                })?;
            }
        }
        DatasetFileFormat::Json => {
            serde_json::from_slice::<serde_json::Value>(bytes)
                .map_err(|error| corrupt(format!("JSON dataset rusak: {error}")))?;
        }
        DatasetFileFormat::Conllu => {
            let text = std::str::from_utf8(bytes).map_err(|_| {
                corrupt(format!(
                    "CoNLL-U dataset {} bukan UTF-8 valid",
                    path.display()
                ))
            })?;
            if !text.lines().any(|line| {
                !line.is_empty() && !line.starts_with('#') && line.split('\t').count() == 10
            }) {
                return Err(corrupt("dataset CoNLL-U tidak memiliki baris token valid"));
            }
        }
        DatasetFileFormat::TarGzip => {
            let decoder = GzDecoder::new(bytes);
            let mut archive = tar::Archive::new(decoder);
            let entries = archive
                .entries()
                .map_err(|error| corrupt(format!("tar dataset rusak: {error}")))?;
            let mut names = BTreeSet::new();
            let mut expanded = 0_u64;
            for item in entries {
                let mut entry =
                    item.map_err(|error| corrupt(format!("entry tar rusak: {error}")))?;
                let path = entry
                    .path()
                    .map_err(|error| corrupt(format!("path tar rusak: {error}")))?;
                if path.is_absolute()
                    || path
                        .components()
                        .any(|component| !matches!(component, Component::Normal(_)))
                {
                    return Err(corrupt("path tar harus relatif dan bebas traversal"));
                }
                let name = path
                    .to_str()
                    .ok_or_else(|| corrupt("path tar bukan UTF-8 valid"))?
                    .to_owned();
                if !names.insert(name) {
                    return Err(corrupt("path duplikat ditemukan dalam arsip tar"));
                }
                let entry_type = entry.header().entry_type();
                if !entry_type.is_file() && !entry_type.is_dir() {
                    return Err(corrupt("jenis file tar bukan file/direktori biasa"));
                }
                expanded = expanded
                    .checked_add(entry.size())
                    .ok_or(SimError::CounterOverflow {
                        counter: "ukuran arsip dataset",
                    })?;
                if expanded > MAX_FILE_BYTES || entry.size() > MAX_FILE_BYTES {
                    return Err(corrupt("arsip dataset terurai melebihi 512 MiB"));
                }
                let declared_size = entry.size();
                let copied = io::copy(&mut entry, &mut io::sink())
                    .map_err(|error| corrupt(format!("isi tar rusak: {error}")))?;
                if copied != declared_size {
                    return Err(corrupt("ukuran entry tar tidak cocok dengan konten"));
                }
            }
            let mut decoder = archive.into_inner();
            let remaining = MAX_FILE_BYTES
                .checked_sub(expanded)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| corrupt("ukuran arsip terurai melampaui batas"))?;
            let extra = io::copy(&mut decoder.by_ref().take(remaining), &mut io::sink())
                .map_err(|error| corrupt(format!("checksum gzip invalid: {error}")))?;
            if extra >= remaining {
                return Err(corrupt("arsip dataset terurai melebihi 512 MiB"));
            }
        }
        DatasetFileFormat::Text => {}
    }
    Ok(())
}

fn classify_format(path: &Path) -> Result<DatasetFileFormat, SimError> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("csv") => Ok(DatasetFileFormat::Csv),
        Some("json") => Ok(DatasetFileFormat::Json),
        Some("conllu") => Ok(DatasetFileFormat::Conllu),
        Some("tab") => Ok(DatasetFileFormat::Text),
        Some("gz") => Ok(DatasetFileFormat::TarGzip),
        Some("txt" | "md" | "dic" | "aff" | "LICENSE") | None => Ok(DatasetFileFormat::Text),
        _ if path.file_name().is_some_and(|name| name == "LICENSE") => Ok(DatasetFileFormat::Text),
        _ => Err(corrupt(format!(
            "ekstensi dataset tidak didukung: {}",
            path.display()
        ))),
    }
}

fn classify_encoding(path: &Path) -> Result<DatasetEncoding, SimError> {
    match classify_format(path)? {
        DatasetFileFormat::TarGzip => Ok(DatasetEncoding::Binary),
        _ => Ok(DatasetEncoding::Utf8),
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn corrupt(reason: impl Into<String>) -> SimError {
    SimError::CorruptData {
        subsystem: "dataset integrity",
        reason: reason.into(),
    }
}

fn io_error(operation: &'static str, error: io::Error) -> SimError {
    SimError::Io {
        operation,
        kind: error.kind(),
    }
}

use super::{
    CorpusManifest, GenerativeError, GenerativeLanguageModel, HybridTrainingReport, Result,
    TrainingReport,
    model::{CausalLanguageModel, TransformerConfig},
    tokenizer::CharacterTokenizer,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAGIC: &[u8; 8] = b"SHLMLM01";
const FORMAT_VERSION: u32 = 2;
const MAX_CHECKPOINT_BYTES: u64 = 128 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointMetadata {
    format_version: u32,
    config: TransformerConfig,
    tokenizer: CharacterTokenizer,
    tokenizer_sha256: String,
    corpus_manifest: Option<CorpusManifest>,
    training_report: Option<TrainingReport>,
    #[serde(default)]
    hybrid_training_report: Option<HybridTrainingReport>,
    weights_sha256: String,
    weights_bytes: u64,
}

pub(super) fn save(model: &GenerativeLanguageModel, path: &Path) -> Result<String> {
    let temporary_directory = create_private_temp_directory()?;
    let weights_path = temporary_directory.join("weights.safetensors");
    let weights_result = model.backend.variables.save(&weights_path);
    if let Err(error) = weights_result {
        let _ = fs::remove_dir_all(&temporary_directory);
        return Err(error.into());
    }
    let weights = match fs::read(&weights_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary_directory);
            return Err(GenerativeError::Io {
                operation: "baca bobot sementara",
                kind: error.kind(),
            });
        }
    };
    let _ = fs::remove_dir_all(&temporary_directory);
    let weights_sha256 = sha256_hex(&weights);
    let tokenizer_bytes = serde_json::to_vec(&model.tokenizer)?;
    let metadata = CheckpointMetadata {
        format_version: FORMAT_VERSION,
        config: model.backend.config.clone(),
        tokenizer: model.tokenizer.clone(),
        tokenizer_sha256: sha256_hex(&tokenizer_bytes),
        corpus_manifest: model.corpus_manifest.clone(),
        training_report: model.training_report.clone(),
        hybrid_training_report: model.hybrid_training_report.clone(),
        weights_sha256,
        weights_bytes: weights.len() as u64,
    };
    let metadata_bytes = serde_json::to_vec(&metadata)?;
    let total_length = 24_u64
        .checked_add(metadata_bytes.len() as u64)
        .and_then(|length| length.checked_add(weights.len() as u64))
        .ok_or_else(|| GenerativeError::Checkpoint("ukuran checkpoint overflow".to_owned()))?;
    if total_length > MAX_CHECKPOINT_BYTES {
        return Err(GenerativeError::Checkpoint(
            "checkpoint melampaui batas 128 MiB".to_owned(),
        ));
    }
    let mut bytes = Vec::with_capacity(total_length as usize);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(weights.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&metadata_bytes);
    bytes.extend_from_slice(&weights);
    let checkpoint_sha256 = sha256_hex(&bytes);
    crate::checkpoint::atomic_write(
        path,
        &bytes,
        "generative model checkpoint",
        "tulis checkpoint model generatif atomik",
    )
    .map_err(|error| GenerativeError::Checkpoint(error.to_string()))?;
    Ok(checkpoint_sha256)
}

pub(super) fn load(path: &Path) -> Result<GenerativeLanguageModel> {
    let file_metadata = fs::metadata(path).map_err(|error| GenerativeError::Io {
        operation: "baca metadata checkpoint model",
        kind: error.kind(),
    })?;
    if file_metadata.len() > MAX_CHECKPOINT_BYTES {
        return Err(GenerativeError::Checkpoint(
            "checkpoint melampaui batas 128 MiB".to_owned(),
        ));
    }
    let bytes = fs::read(path).map_err(|error| GenerativeError::Io {
        operation: "baca checkpoint model",
        kind: error.kind(),
    })?;
    if bytes.len() < 24 || &bytes[..8] != MAGIC {
        return Err(GenerativeError::Checkpoint(
            "magic atau header checkpoint tidak cocok".to_owned(),
        ));
    }
    let metadata_length = usize::try_from(read_u64(&bytes[8..16])?)
        .map_err(|_| GenerativeError::Checkpoint("panjang metadata overflow".to_owned()))?;
    let weights_length = usize::try_from(read_u64(&bytes[16..24])?)
        .map_err(|_| GenerativeError::Checkpoint("panjang bobot overflow".to_owned()))?;
    let metadata_end = 24_usize
        .checked_add(metadata_length)
        .ok_or_else(|| GenerativeError::Checkpoint("offset metadata overflow".to_owned()))?;
    let weights_end = metadata_end
        .checked_add(weights_length)
        .ok_or_else(|| GenerativeError::Checkpoint("offset bobot overflow".to_owned()))?;
    if weights_end != bytes.len() {
        return Err(GenerativeError::Checkpoint(
            "panjang metadata/bobot tidak cocok dengan file".to_owned(),
        ));
    }
    let metadata: CheckpointMetadata = serde_json::from_slice(&bytes[24..metadata_end])?;
    if !(1..=FORMAT_VERSION).contains(&metadata.format_version) {
        return Err(GenerativeError::Checkpoint(format!(
            "versi format {} tidak didukung",
            metadata.format_version
        )));
    }
    if metadata.weights_bytes != weights_length as u64 {
        return Err(GenerativeError::Checkpoint(
            "panjang bobot metadata tidak cocok".to_owned(),
        ));
    }
    metadata.tokenizer.validate()?;
    let tokenizer_bytes = serde_json::to_vec(&metadata.tokenizer)?;
    if sha256_hex(&tokenizer_bytes) != metadata.tokenizer_sha256 {
        return Err(GenerativeError::Checkpoint(
            "checksum tokenizer tidak cocok".to_owned(),
        ));
    }
    metadata
        .config
        .validate(metadata.tokenizer.vocabulary_size())?;
    let weights = &bytes[metadata_end..weights_end];
    if sha256_hex(weights) != metadata.weights_sha256 {
        return Err(GenerativeError::Checkpoint(
            "checksum bobot tidak cocok".to_owned(),
        ));
    }

    let mut backend = CausalLanguageModel::new(
        metadata.config.clone(),
        metadata.tokenizer.vocabulary_size(),
    )?;
    load_weights(&mut backend, weights, metadata.format_version == 1)?;
    let checkpoint_sha256 = sha256_hex(&bytes);
    Ok(GenerativeLanguageModel {
        tokenizer: metadata.tokenizer,
        backend,
        corpus_manifest: metadata.corpus_manifest,
        training_report: metadata.training_report,
        hybrid_training_report: metadata.hybrid_training_report,
        checkpoint_sha256: Some(checkpoint_sha256),
    })
}

fn load_weights(
    backend: &mut CausalLanguageModel,
    weights: &[u8],
    allow_missing_lif_head: bool,
) -> Result<()> {
    let temporary_directory = create_private_temp_directory()?;
    let weights_path = temporary_directory.join("weights.safetensors");
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&weights_path)
            .map_err(|error| GenerativeError::Io {
                operation: "buat file bobot sementara",
                kind: error.kind(),
            })?;
        file.write_all(weights)
            .map_err(|error| GenerativeError::Io {
                operation: "tulis bobot sementara",
                kind: error.kind(),
            })?;
        file.sync_all().map_err(|error| GenerativeError::Io {
            operation: "sync bobot sementara",
            kind: error.kind(),
        })?;
        drop(file);
        if allow_missing_lif_head {
            let tensors = candle_core::safetensors::SliceSafetensors::new(weights)?;
            let tensor_names = tensors
                .tensors()
                .into_iter()
                .map(|(name, _)| name)
                .collect::<std::collections::BTreeSet<_>>();
            let variables = backend.variables.data().lock().map_err(|_| {
                GenerativeError::Checkpoint("lock VarMap rusak saat migrasi checkpoint".to_owned())
            })?;
            for (name, variable) in variables.iter() {
                if tensor_names.contains(name) {
                    let value = tensors.load(name, variable.device())?;
                    variable.set(&value)?;
                } else if !name.starts_with("lif_head.") {
                    return Err(GenerativeError::Checkpoint(format!(
                        "checkpoint lama kehilangan tensor Transformer {name}"
                    )));
                }
            }
        } else {
            backend.variables.load(&weights_path)?;
        }
        Ok(())
    })();
    let _ = fs::remove_dir_all(&temporary_directory);
    result
}

fn create_private_temp_directory() -> Result<PathBuf> {
    let parent = std::env::temp_dir();
    for _ in 0..32 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!("shella-lm-{}-{sequence}", std::process::id()));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(GenerativeError::Io {
                    operation: "buat direktori bobot sementara",
                    kind: error.kind(),
                });
            }
        }
    }
    Err(GenerativeError::Checkpoint(
        "gagal memperoleh nama unik untuk berkas bobot sementara".to_owned(),
    ))
}

fn read_u64(bytes: &[u8]) -> Result<u64> {
    let array: [u8; 8] = bytes.try_into().map_err(|_| {
        GenerativeError::Checkpoint("field panjang header tidak lengkap".to_owned())
    })?;
    Ok(u64::from_le_bytes(array))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::generative::{
        CharacterTokenizer, ComputeDevice, FeedbackAuthority, GenerationConfig,
        GenerativeLanguageModel, HybridAction, HybridTrainingConfig, TransformerConfig,
        VerifiedActionExample,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("shella-test-model-{}-{nanos}", std::process::id()))
    }

    fn model() -> GenerativeLanguageModel {
        let tokenizer = CharacterTokenizer::fit(["halo dunia."]).unwrap();
        GenerativeLanguageModel::new(
            TransformerConfig {
                hidden_size: 8,
                layer_count: 1,
                head_count: 2,
                max_context: 8,
                seed: 31,
                device: ComputeDevice::Cpu,
            },
            tokenizer,
        )
        .unwrap()
    }

    #[test]
    fn checkpoint_round_trip_restores_generation_and_tensor_weights() {
        let path = unique_path();
        let mut original = model();
        let before = original.backend.parameter_snapshot().unwrap();
        let fingerprint = original.save_checkpoint(&path).unwrap();
        assert_eq!(fingerprint.len(), 64);
        let restored = GenerativeLanguageModel::load_checkpoint(&path).unwrap();
        let after = restored.backend.parameter_snapshot().unwrap();
        assert_eq!(before, after);
        assert_eq!(restored.checkpoint_sha256(), Some(fingerprint.as_str()));
        assert_eq!(
            original
                .generate(
                    "halo",
                    GenerationConfig {
                        max_new_tokens: 2,
                        ..GenerationConfig::default()
                    }
                )
                .unwrap()
                .continuation,
            restored
                .generate(
                    "halo",
                    GenerationConfig {
                        max_new_tokens: 2,
                        ..GenerationConfig::default()
                    }
                )
                .unwrap()
                .continuation,
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_detects_corruption_before_tensor_loading() {
        let path = unique_path();
        let mut original = model();
        original.save_checkpoint(&path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x5a;
        fs::write(&path, bytes).unwrap();
        assert!(GenerativeLanguageModel::load_checkpoint(&path).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_round_trip_restores_joint_lif_training_and_report() {
        let path = unique_path();
        let mut original = model();
        original
            .train_on_verified_feedback(
                &[VerifiedActionExample {
                    source_id: "review-001".to_owned(),
                    prompt: "halo dunia.".to_owned(),
                    candidate_response: Some("halo dunia.".to_owned()),
                    target_response: Some("halo dunia.".to_owned()),
                    expected_action: HybridAction::Answer,
                    authority: FeedbackAuthority::HumanConfirmed,
                }],
                HybridTrainingConfig::default(),
            )
            .unwrap();
        let prediction = original.predict_hybrid_action("halo dunia.").unwrap();
        let fingerprint = original.save_checkpoint(&path).unwrap();
        let restored = GenerativeLanguageModel::load_checkpoint(&path).unwrap();
        assert!(restored.has_trained_hybrid_head());
        assert_eq!(
            original.hybrid_training_report(),
            restored.hybrid_training_report()
        );
        assert_eq!(
            prediction,
            restored.predict_hybrid_action("halo dunia.").unwrap()
        );
        assert_eq!(restored.checkpoint_sha256(), Some(fingerprint.as_str()));
        fs::remove_file(path).unwrap();
    }
}

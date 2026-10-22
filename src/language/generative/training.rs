use super::{
    CorpusRecord, CorpusSplit, GenerativeError, GenerativeLanguageModel, Result,
    model::masked_next_token_loss,
    tokenizer::{CharacterTokenizer, PAD_ID},
};
use candle_core::{Device, Tensor, backprop::GradStore};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use serde::{Deserialize, Serialize};

const MAX_EPOCHS: usize = 100;
const MAX_BATCH_SIZE: usize = 64;
const MAX_ATTENTION_ACTIVATIONS: usize = 16_777_216;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub gradient_clip_norm: f64,
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 6,
            batch_size: 8,
            learning_rate: 0.002,
            weight_decay: 0.01,
            gradient_clip_norm: 1.0,
            seed: 42,
        }
    }
}

impl TrainingConfig {
    pub fn validate(&self, model: &GenerativeLanguageModel) -> Result<()> {
        if self.epochs == 0 || self.epochs > MAX_EPOCHS {
            return Err(GenerativeError::InvalidConfig {
                parameter: "epochs",
                reason: "harus berada pada rentang 1..=100",
            });
        }
        if self.batch_size == 0 || self.batch_size > MAX_BATCH_SIZE {
            return Err(GenerativeError::InvalidConfig {
                parameter: "batch_size",
                reason: "harus berada pada rentang 1..=64",
            });
        }
        if !self.learning_rate.is_finite() || !(1e-6..=0.1).contains(&self.learning_rate) {
            return Err(GenerativeError::InvalidConfig {
                parameter: "learning_rate",
                reason: "harus finite dan berada pada rentang 1e-6..=0.1",
            });
        }
        if !self.weight_decay.is_finite() || !(0.0..=1.0).contains(&self.weight_decay) {
            return Err(GenerativeError::InvalidConfig {
                parameter: "weight_decay",
                reason: "harus finite dan berada pada rentang 0..=1",
            });
        }
        if !self.gradient_clip_norm.is_finite()
            || !(1e-6..=100.0).contains(&self.gradient_clip_norm)
        {
            return Err(GenerativeError::InvalidConfig {
                parameter: "gradient_clip_norm",
                reason: "harus finite dan berada pada rentang 1e-6..=100",
            });
        }
        let config = model.config();
        let attention_elements = self
            .batch_size
            .saturating_mul(config.head_count)
            .saturating_mul(config.max_context)
            .saturating_mul(config.max_context);
        if attention_elements > MAX_ATTENTION_ACTIVATIONS {
            return Err(GenerativeError::InvalidConfig {
                parameter: "batch_size",
                reason: "perkiraan attention activation batch/context melampaui batas memori",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub training_loss: f64,
    pub training_perplexity: f64,
    pub training_tokens: usize,
    pub validation_loss: f64,
    pub validation_perplexity: f64,
    pub validation_tokens: usize,
    pub optimizer_steps: usize,
    pub mean_gradient_norm_before_clip: f64,
    pub clipped_batches: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingReport {
    pub config: TrainingConfig,
    pub epochs: Vec<EpochMetrics>,
    pub training_documents: usize,
    pub validation_documents: usize,
    pub parameter_count: usize,
    pub optimizer: String,
    pub optimizer_state_saved: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationMetrics {
    pub documents: usize,
    pub windows: usize,
    pub tokens: usize,
    pub mean_loss: f64,
    pub perplexity: f64,
}

#[derive(Debug)]
struct Window {
    input: Vec<u32>,
    target: Vec<u32>,
}

pub(crate) fn train_model(
    model: &mut GenerativeLanguageModel,
    corpus: &CorpusSplit,
    config: TrainingConfig,
) -> Result<TrainingReport> {
    config.validate(model)?;
    corpus.validate_disjoint_ids()?;
    if model.corpus_manifest.as_ref() != Some(&corpus.manifest) {
        return Err(GenerativeError::InvalidInput {
            parameter: "training_corpus",
            reason: "model dan split manifest harus berasal dari corpus yang sama",
        });
    }
    if corpus.training.is_empty() || corpus.validation.is_empty() {
        return Err(GenerativeError::InvalidInput {
            parameter: "training_corpus",
            reason: "training membutuhkan train dan validation nonkosong",
        });
    }

    let windows = make_windows(
        &corpus.training,
        &model.tokenizer,
        model.config().max_context,
    )?;
    if windows.is_empty() {
        return Err(GenerativeError::InvalidInput {
            parameter: "training_corpus",
            reason: "tidak ada pasangan input-target token berikutnya",
        });
    }
    let variables = model.backend.variables();
    let mut optimizer = AdamW::new(
        variables.clone(),
        ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        },
    )?;
    let mut epoch_metrics = Vec::with_capacity(config.epochs);
    let mut shuffle_state = config.seed;
    let mut total_optimizer_steps = 0usize;

    for epoch in 1..=config.epochs {
        let mut order = (0..windows.len()).collect::<Vec<_>>();
        deterministic_shuffle(&mut order, &mut shuffle_state);
        let mut training_loss_sum = 0.0_f64;
        let mut training_tokens = 0usize;
        let mut gradient_norm_sum = 0.0_f64;
        let mut clipped_batches = 0usize;
        let mut epoch_steps = 0usize;

        for batch_indices in order.chunks(config.batch_size) {
            let batch = make_batch(
                batch_indices.iter().map(|index| &windows[*index]),
                &Device::Cpu,
            )?;
            let logits = model.backend.forward(&batch.inputs)?;
            let (loss, valid_tokens) = masked_next_token_loss(&logits, &batch.targets)?;
            let loss_value = f64::from(loss.to_scalar::<f32>()?);
            if !loss_value.is_finite() || loss_value < 0.0 {
                return Err(GenerativeError::Numerical {
                    operation: "training cross-entropy",
                });
            }
            let mut gradients = loss.backward()?;
            let (gradient_norm, clipped) =
                clip_gradients(&mut gradients, &variables, config.gradient_clip_norm)?;
            optimizer.step(&gradients)?;
            training_loss_sum += loss_value * valid_tokens as f64;
            training_tokens += valid_tokens;
            gradient_norm_sum += gradient_norm;
            clipped_batches += usize::from(clipped);
            epoch_steps = epoch_steps
                .checked_add(1)
                .ok_or(GenerativeError::Numerical {
                    operation: "optimizer step counter",
                })?;
            total_optimizer_steps =
                total_optimizer_steps
                    .checked_add(1)
                    .ok_or(GenerativeError::Numerical {
                        operation: "optimizer step counter",
                    })?;
        }
        if training_tokens == 0 || epoch_steps == 0 {
            return Err(GenerativeError::InvalidInput {
                parameter: "training_corpus",
                reason: "epoch tidak menghasilkan token training atau langkah optimizer",
            });
        }
        let validation = evaluate_model(model, &corpus.validation, config.batch_size)?;
        let training_loss = training_loss_sum / training_tokens as f64;
        epoch_metrics.push(EpochMetrics {
            epoch,
            training_loss,
            training_perplexity: finite_perplexity(training_loss)?,
            training_tokens,
            validation_loss: validation.mean_loss,
            validation_perplexity: validation.perplexity,
            validation_tokens: validation.tokens,
            optimizer_steps: epoch_steps,
            mean_gradient_norm_before_clip: gradient_norm_sum / epoch_steps as f64,
            clipped_batches,
        });
    }

    Ok(TrainingReport {
        config,
        epochs: epoch_metrics,
        training_documents: corpus.training.len(),
        validation_documents: corpus.validation.len(),
        parameter_count: model.parameter_count(),
        optimizer: "Candle AdamW".to_owned(),
        optimizer_state_saved: false,
    })
}

pub(crate) fn evaluate_model(
    model: &GenerativeLanguageModel,
    records: &[CorpusRecord],
    batch_size: usize,
) -> Result<EvaluationMetrics> {
    if records.is_empty() {
        return Err(GenerativeError::InvalidInput {
            parameter: "evaluation_records",
            reason: "evaluasi membutuhkan dokumen nonkosong",
        });
    }
    if batch_size == 0 || batch_size > MAX_BATCH_SIZE {
        return Err(GenerativeError::InvalidConfig {
            parameter: "batch_size",
            reason: "harus berada pada rentang 1..=64",
        });
    }
    let windows = make_windows(records, &model.tokenizer, model.config().max_context)?;
    if windows.is_empty() {
        return Err(GenerativeError::InvalidInput {
            parameter: "evaluation_records",
            reason: "tidak ada token target untuk dihitung",
        });
    }
    let mut weighted_loss = 0.0_f64;
    let mut token_count = 0usize;
    let mut window_count = 0usize;
    for window_batch in windows.chunks(batch_size) {
        let batch = make_batch(window_batch.iter(), &Device::Cpu)?;
        let logits = model.backend.forward(&batch.inputs)?;
        let (loss, valid_tokens) = masked_next_token_loss(&logits, &batch.targets)?;
        let loss_value = f64::from(loss.to_scalar::<f32>()?);
        if !loss_value.is_finite() || loss_value < 0.0 {
            return Err(GenerativeError::Numerical {
                operation: "evaluation cross-entropy",
            });
        }
        weighted_loss += loss_value * valid_tokens as f64;
        token_count += valid_tokens;
        window_count += window_batch.len();
    }
    if token_count == 0 {
        return Err(GenerativeError::InvalidInput {
            parameter: "evaluation_records",
            reason: "evaluasi tidak memiliki label non-padding",
        });
    }
    let mean_loss = weighted_loss / token_count as f64;
    Ok(EvaluationMetrics {
        documents: records.len(),
        windows: window_count,
        tokens: token_count,
        mean_loss,
        perplexity: finite_perplexity(mean_loss)?,
    })
}

fn make_windows(
    records: &[CorpusRecord],
    tokenizer: &CharacterTokenizer,
    max_context: usize,
) -> Result<Vec<Window>> {
    let mut windows = Vec::new();
    for record in records {
        let sequence = tokenizer.encode_document(&record.text)?;
        if sequence.len() < 2 {
            continue;
        }
        let mut start = 0usize;
        while start + 1 < sequence.len() {
            let end = (start + max_context + 1).min(sequence.len());
            if end <= start + 1 {
                break;
            }
            windows.push(Window {
                input: sequence[start..end - 1].to_vec(),
                target: sequence[start + 1..end].to_vec(),
            });
            start += max_context;
        }
    }
    Ok(windows)
}

struct Batch {
    inputs: Tensor,
    targets: Tensor,
}

fn make_batch<'a>(windows: impl IntoIterator<Item = &'a Window>, device: &Device) -> Result<Batch> {
    let windows = windows.into_iter().collect::<Vec<_>>();
    if windows.is_empty() {
        return Err(GenerativeError::InvalidInput {
            parameter: "batch",
            reason: "batch tidak boleh kosong",
        });
    }
    let sequence = windows
        .iter()
        .map(|window| window.input.len())
        .max()
        .unwrap_or(0);
    if sequence == 0
        || windows
            .iter()
            .any(|window| window.input.len() != window.target.len())
    {
        return Err(GenerativeError::InvalidInput {
            parameter: "batch",
            reason: "panjang input-target batch tidak sah",
        });
    }
    let batch_size = windows.len();
    let mut inputs = Vec::with_capacity(batch_size * sequence);
    let mut targets = Vec::with_capacity(batch_size * sequence);
    for window in windows {
        inputs.extend_from_slice(&window.input);
        inputs.resize(inputs.len() + sequence - window.input.len(), PAD_ID);
        targets.extend_from_slice(&window.target);
        targets.resize(targets.len() + sequence - window.target.len(), PAD_ID);
    }
    Ok(Batch {
        inputs: Tensor::from_vec(inputs, (batch_size, sequence), device)?,
        targets: Tensor::from_vec(targets, (batch_size, sequence), device)?,
    })
}

fn clip_gradients(
    gradients: &mut GradStore,
    variables: &[candle_core::Var],
    maximum_norm: f64,
) -> Result<(f64, bool)> {
    let mut squared_norm = 0.0_f64;
    for variable in variables {
        if let Some(gradient) = gradients.get(variable.as_tensor()) {
            for value in gradient.flatten_all()?.to_vec1::<f32>()? {
                if !value.is_finite() {
                    return Err(GenerativeError::Numerical {
                        operation: "gradient clipping input",
                    });
                }
                squared_norm += f64::from(value) * f64::from(value);
            }
        }
    }
    let norm = squared_norm.sqrt();
    if !norm.is_finite() {
        return Err(GenerativeError::Numerical {
            operation: "gradient norm",
        });
    }
    if norm <= maximum_norm || norm == 0.0 {
        return Ok((norm, false));
    }
    let scale = maximum_norm / norm;
    for variable in variables {
        let tensor = variable.as_tensor();
        if let Some(gradient) = gradients.get(tensor).cloned() {
            gradients.insert(tensor, gradient.affine(scale, 0.0)?);
        }
    }
    Ok((norm, true))
}

fn finite_perplexity(mean_loss: f64) -> Result<f64> {
    let perplexity = mean_loss.exp();
    if !perplexity.is_finite() {
        return Err(GenerativeError::Numerical {
            operation: "perplexity exponentiation",
        });
    }
    Ok(perplexity)
}

fn deterministic_shuffle(values: &mut [usize], state: &mut u64) {
    for index in (1..values.len()).rev() {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut random = *state;
        random = (random ^ (random >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        random = (random ^ (random >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        random ^= random >> 31;
        values.swap(index, (random as usize) % (index + 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::generative::{
        CorpusManifest, CorpusSourceManifest, DatasetSplitConfig, GenerativeLanguageModel,
        SplitManifest, TransformerConfig,
    };

    fn fixture_corpus() -> CorpusSplit {
        let training = vec![
            CorpusRecord {
                id: 1,
                text: "saya makan nasi".to_owned(),
            },
            CorpusRecord {
                id: 2,
                text: "dia minum air".to_owned(),
            },
            CorpusRecord {
                id: 3,
                text: "kami membaca buku".to_owned(),
            },
            CorpusRecord {
                id: 4,
                text: "mereka pergi pagi".to_owned(),
            },
        ];
        let validation = vec![CorpusRecord {
            id: 5,
            text: "saya minum air".to_owned(),
        }];
        let test = vec![CorpusRecord {
            id: 6,
            text: "dia membaca buku".to_owned(),
        }];
        let manifest = CorpusManifest {
            format_version: 1,
            source: CorpusSourceManifest {
                dataset_id: "test-only-unit-fixture".to_owned(),
                repository_url: "test".to_owned(),
                revision: "test".to_owned(),
                license_id: "test".to_owned(),
                license_url: "test".to_owned(),
                attribution: "test-only fixture".to_owned(),
                data_scope: "unit test only; not corpus evidence".to_owned(),
                source_split: "test-only".to_owned(),
                source_file_path: "test".to_owned(),
                source_file_git_blob_sha1: "test".to_owned(),
                source_file_sha256: "test".to_owned(),
                source_file_size_bytes: 0,
                text_field: "text".to_owned(),
                excluded_sources: Vec::new(),
            },
            split: SplitManifest {
                seed: 1,
                training_fraction_ppm: 666_667,
                validation_fraction_ppm: 166_667,
                test_fraction_ppm: 166_666,
                exact_duplicates_removed: 0,
                near_duplicates_removed: 0,
                source_records: 6,
                training_ids: vec![1, 2, 3, 4],
                validation_ids: vec![5],
                test_ids: vec![6],
            },
            training_characters: 0,
            validation_characters: 0,
            test_characters: 0,
            normalization: "test".to_owned(),
            deduplication: "test".to_owned(),
        };
        CorpusSplit {
            training,
            validation,
            test,
            manifest,
        }
    }

    #[test]
    fn training_performs_real_gradient_updates_and_validation_is_finite() {
        let corpus = fixture_corpus();
        corpus.validate_disjoint_ids().unwrap();
        let mut model = GenerativeLanguageModel::from_corpus(
            TransformerConfig {
                hidden_size: 8,
                layer_count: 1,
                head_count: 2,
                max_context: 16,
                seed: 91,
                device: super::super::ComputeDevice::Cpu,
            },
            &corpus,
        )
        .unwrap();
        let before = model.backend.parameter_snapshot().unwrap();
        let report = model
            .train(
                &corpus,
                TrainingConfig {
                    epochs: 1,
                    batch_size: 2,
                    learning_rate: 0.01,
                    ..Default::default()
                },
            )
            .unwrap();
        let after = model.backend.parameter_snapshot().unwrap();
        assert!(before.iter().zip(&after).any(|(left, right)| left != right));
        assert_eq!(report.epochs.len(), 1);
        assert!(report.epochs[0].training_loss.is_finite());
        assert!(report.epochs[0].validation_loss.is_finite());
        assert!(report.epochs[0].optimizer_steps > 0);
        assert!(report.epochs[0].mean_gradient_norm_before_clip.is_finite());
    }

    #[test]
    fn training_config_rejects_unbounded_or_nonfinite_values() {
        let corpus = fixture_corpus();
        let model =
            GenerativeLanguageModel::from_corpus(TransformerConfig::default(), &corpus).unwrap();
        let invalid = TrainingConfig {
            learning_rate: f64::NAN,
            ..TrainingConfig::default()
        };
        assert!(invalid.validate(&model).is_err());
        let invalid = TrainingConfig {
            batch_size: 0,
            ..TrainingConfig::default()
        };
        assert!(invalid.validate(&model).is_err());
        assert!(DatasetSplitConfig::default().validate().is_ok());
    }
}

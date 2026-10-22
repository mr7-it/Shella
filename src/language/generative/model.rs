use super::{GenerativeError, Result, tokenizer::PAD_ID};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

const MAX_HIDDEN_SIZE: usize = 256;
const MAX_LAYER_COUNT: usize = 4;
const MAX_HEAD_COUNT: usize = 8;
const MAX_CONTEXT_LENGTH: usize = 256;
const MAX_VOCABULARY_SIZE: usize = 4100;
const MAX_PARAMETER_COUNT: usize = 4_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeDevice {
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub layer_count: usize,
    pub head_count: usize,
    pub max_context: usize,
    pub seed: u64,
    pub device: ComputeDevice,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 32,
            layer_count: 1,
            head_count: 2,
            max_context: 64,
            seed: 42,
            device: ComputeDevice::Cpu,
        }
    }
}

impl TransformerConfig {
    pub fn validate(&self, vocabulary_size: usize) -> Result<()> {
        if !(8..=MAX_HIDDEN_SIZE).contains(&self.hidden_size) {
            return Err(GenerativeError::InvalidConfig {
                parameter: "hidden_size",
                reason: "harus berada pada rentang 8..=256",
            });
        }
        if self.layer_count == 0 || self.layer_count > MAX_LAYER_COUNT {
            return Err(GenerativeError::InvalidConfig {
                parameter: "layer_count",
                reason: "harus berada pada rentang 1..=4",
            });
        }
        if self.head_count == 0
            || self.head_count > MAX_HEAD_COUNT
            || self.hidden_size % self.head_count != 0
        {
            return Err(GenerativeError::InvalidConfig {
                parameter: "head_count",
                reason: "harus 1..=8 dan membagi hidden_size secara tepat",
            });
        }
        if !(2..=MAX_CONTEXT_LENGTH).contains(&self.max_context) {
            return Err(GenerativeError::InvalidConfig {
                parameter: "max_context",
                reason: "harus berada pada rentang 2..=256 token",
            });
        }
        if !(4..=MAX_VOCABULARY_SIZE).contains(&vocabulary_size) {
            return Err(GenerativeError::InvalidConfig {
                parameter: "vocabulary_size",
                reason: "harus berada pada rentang 4..=4100 token",
            });
        }
        let estimated_parameters = vocabulary_size
            .saturating_mul(self.hidden_size)
            .saturating_add(self.max_context.saturating_mul(self.hidden_size))
            .saturating_add(
                self.layer_count
                    .saturating_mul(12)
                    .saturating_mul(self.hidden_size)
                    .saturating_mul(self.hidden_size),
            )
            .saturating_add(vocabulary_size.saturating_mul(self.hidden_size));
        if estimated_parameters > MAX_PARAMETER_COUNT {
            return Err(GenerativeError::InvalidConfig {
                parameter: "model_size",
                reason: "perkiraan parameter melampaui batas 4000000",
            });
        }
        Ok(())
    }
}

pub(crate) struct CausalLanguageModel {
    pub(crate) config: TransformerConfig,
    pub(crate) variables: VarMap,
    decoder: Decoder,
}

impl CausalLanguageModel {
    pub(crate) fn new(config: TransformerConfig, vocabulary_size: usize) -> Result<Self> {
        config.validate(vocabulary_size)?;
        let device = Device::Cpu;
        let mut variables = VarMap::new();
        let builder = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let decoder = Decoder::new(&config, vocabulary_size, builder)?;
        initialize_variables(&mut variables, &config, vocabulary_size)?;
        Ok(Self {
            config,
            variables,
            decoder,
        })
    }

    pub(crate) fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, sequence) = input_ids.dims2()?;
        if batch == 0 || sequence == 0 || sequence > self.config.max_context {
            return Err(GenerativeError::InvalidInput {
                parameter: "input_ids",
                reason: "batch dan sequence harus tidak kosong; panjang tidak boleh melebihi konteks",
            });
        }
        self.decoder.forward(input_ids, batch, sequence)
    }

    pub(crate) fn variables(&self) -> Vec<candle_core::Var> {
        self.variables.all_vars()
    }

    #[cfg(test)]
    pub(crate) fn parameter_snapshot(&self) -> Result<Vec<Vec<f32>>> {
        let mut snapshot = self
            .variables
            .all_vars()
            .into_iter()
            .map(|variable| Ok(variable.as_tensor().flatten_all()?.to_vec1::<f32>()?))
            .collect::<Result<Vec<_>>>()?;
        snapshot.sort_by(|left, right| {
            for (left_value, right_value) in left.iter().zip(right) {
                let ordering = left_value.total_cmp(right_value);
                if ordering != std::cmp::Ordering::Equal {
                    return ordering;
                }
            }
            left.len().cmp(&right.len())
        });
        Ok(snapshot)
    }
}

#[derive(Debug)]
struct Decoder {
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    output: Linear,
    vocabulary_size: usize,
}

impl Decoder {
    fn new(
        config: &TransformerConfig,
        vocabulary_size: usize,
        builder: VarBuilder,
    ) -> Result<Self> {
        let token_embedding = candle_nn::embedding(
            vocabulary_size,
            config.hidden_size,
            builder.pp("token_embedding"),
        )?;
        let position_embedding = candle_nn::embedding(
            config.max_context,
            config.hidden_size,
            builder.pp("position_embedding"),
        )?;
        let blocks = (0..config.layer_count)
            .map(|index| {
                TransformerBlock::new(
                    config.hidden_size,
                    config.head_count,
                    builder.pp(format!("blocks.{index}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let final_norm = candle_nn::layer_norm(config.hidden_size, 1e-5, builder.pp("final_norm"))?;
        let output = candle_nn::linear(config.hidden_size, vocabulary_size, builder.pp("output"))?;
        Ok(Self {
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            output,
            vocabulary_size,
        })
    }

    fn forward(&self, input_ids: &Tensor, batch: usize, sequence: usize) -> Result<Tensor> {
        let positions = Tensor::from_vec(
            (0..sequence).map(|value| value as u32).collect::<Vec<_>>(),
            (1, sequence),
            &Device::Cpu,
        )?;
        let token_values = self.token_embedding.forward(input_ids)?;
        let position_values = self.position_embedding.forward(&positions)?;
        let mut hidden = token_values.broadcast_add(&position_values)?;
        let causal_mask = create_causal_mask(sequence)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &causal_mask, batch, sequence)?;
        }
        let hidden = self.final_norm.forward(&hidden)?;
        let logits = self.output.forward(&hidden)?;
        if logits.dim(D::Minus1)? != self.vocabulary_size {
            return Err(GenerativeError::Backend {
                operation: "vocabulary projection",
                reason: "output head menghasilkan dimensi yang tidak cocok".to_owned(),
            });
        }
        Ok(logits)
    }
}

#[derive(Debug)]
struct TransformerBlock {
    attention_norm: LayerNorm,
    qkv: Linear,
    attention_projection: Linear,
    feed_forward_norm: LayerNorm,
    feed_forward_in: Linear,
    feed_forward_out: Linear,
    hidden_size: usize,
    head_count: usize,
}

impl TransformerBlock {
    fn new(hidden_size: usize, head_count: usize, builder: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention_norm: candle_nn::layer_norm(hidden_size, 1e-5, builder.pp("attention_norm"))?,
            qkv: candle_nn::linear(hidden_size, hidden_size * 3, builder.pp("qkv"))?,
            attention_projection: candle_nn::linear(
                hidden_size,
                hidden_size,
                builder.pp("attention_projection"),
            )?,
            feed_forward_norm: candle_nn::layer_norm(
                hidden_size,
                1e-5,
                builder.pp("feed_forward_norm"),
            )?,
            feed_forward_in: candle_nn::linear(
                hidden_size,
                hidden_size * 4,
                builder.pp("feed_forward_in"),
            )?,
            feed_forward_out: candle_nn::linear(
                hidden_size * 4,
                hidden_size,
                builder.pp("feed_forward_out"),
            )?,
            hidden_size,
            head_count,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        causal_mask: &Tensor,
        batch: usize,
        sequence: usize,
    ) -> Result<Tensor> {
        let normalized = self.attention_norm.forward(input)?;
        let qkv = self.qkv.forward(&normalized)?;
        let head_size = self.hidden_size / self.head_count;
        let query = qkv
            .narrow(2, 0, self.hidden_size)?
            .reshape((batch, sequence, self.head_count, head_size))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = qkv
            .narrow(2, self.hidden_size, self.hidden_size)?
            .reshape((batch, sequence, self.head_count, head_size))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = qkv
            .narrow(2, self.hidden_size * 2, self.hidden_size)?
            .reshape((batch, sequence, self.head_count, head_size))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_transposed = key.transpose(2, 3)?.contiguous()?;
        let scores = query
            .matmul(&key_transposed)?
            .affine(1.0 / (head_size as f64).sqrt(), 0.0)?
            .broadcast_add(causal_mask)?;
        let probabilities = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attended = probabilities
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, sequence, self.hidden_size))?;
        let projected = self.attention_projection.forward(&attended)?;
        let residual = input.broadcast_add(&projected)?;
        let normalized = self.feed_forward_norm.forward(&residual)?;
        let expanded = self.feed_forward_in.forward(&normalized)?.relu()?;
        let compressed = self.feed_forward_out.forward(&expanded)?;
        Ok(residual.broadcast_add(&compressed)?)
    }
}

fn create_causal_mask(sequence: usize) -> Result<Tensor> {
    let mut values = Vec::with_capacity(sequence * sequence);
    for row in 0..sequence {
        for column in 0..sequence {
            values.push(if column <= row { 0.0_f32 } else { -1.0e9_f32 });
        }
    }
    Ok(Tensor::from_vec(
        values,
        (sequence, sequence),
        &Device::Cpu,
    )?)
}

fn initialize_variables(
    variables: &mut VarMap,
    config: &TransformerConfig,
    vocabulary_size: usize,
) -> Result<()> {
    let hidden = config.hidden_size;
    let mut state = config.seed ^ 0x6a09_e667_f3bc_c909;
    let embedding_bound = (3.0 / hidden as f64).sqrt() as f32;
    // Embedding rows use a fan-in-scaled interval rather than the linear-layer Xavier bound.
    let token_values = deterministic_values(vocabulary_size * hidden, embedding_bound, &mut state);
    variables.set_one(
        "token_embedding.weight",
        Tensor::from_vec(token_values, (vocabulary_size, hidden), &Device::Cpu)?,
    )?;
    let position_values =
        deterministic_values(config.max_context * hidden, embedding_bound, &mut state);
    variables.set_one(
        "position_embedding.weight",
        Tensor::from_vec(position_values, (config.max_context, hidden), &Device::Cpu)?,
    )?;

    for index in 0..config.layer_count {
        let prefix = format!("blocks.{index}");
        set_layer_norm(variables, &format!("{prefix}.attention_norm"), hidden)?;
        set_linear(
            variables,
            &format!("{prefix}.qkv"),
            hidden,
            hidden * 3,
            &mut state,
        )?;
        set_linear(
            variables,
            &format!("{prefix}.attention_projection"),
            hidden,
            hidden,
            &mut state,
        )?;
        set_layer_norm(variables, &format!("{prefix}.feed_forward_norm"), hidden)?;
        set_linear(
            variables,
            &format!("{prefix}.feed_forward_in"),
            hidden,
            hidden * 4,
            &mut state,
        )?;
        set_linear(
            variables,
            &format!("{prefix}.feed_forward_out"),
            hidden * 4,
            hidden,
            &mut state,
        )?;
    }
    set_layer_norm(variables, "final_norm", hidden)?;
    set_linear(variables, "output", hidden, vocabulary_size, &mut state)?;
    Ok(())
}

fn set_layer_norm(variables: &mut VarMap, prefix: &str, hidden: usize) -> Result<()> {
    variables.set_one(
        format!("{prefix}.weight"),
        Tensor::ones(hidden, DType::F32, &Device::Cpu)?,
    )?;
    variables.set_one(
        format!("{prefix}.bias"),
        Tensor::zeros(hidden, DType::F32, &Device::Cpu)?,
    )?;
    Ok(())
}

fn set_linear(
    variables: &mut VarMap,
    prefix: &str,
    fan_in: usize,
    fan_out: usize,
    state: &mut u64,
) -> Result<()> {
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt() as f32;
    let values = deterministic_values(fan_in * fan_out, bound, state);
    variables.set_one(
        format!("{prefix}.weight"),
        Tensor::from_vec(values, (fan_out, fan_in), &Device::Cpu)?,
    )?;
    variables.set_one(
        format!("{prefix}.bias"),
        Tensor::zeros(fan_out, DType::F32, &Device::Cpu)?,
    )?;
    Ok(())
}

fn deterministic_values(length: usize, bound: f32, state: &mut u64) -> Vec<f32> {
    (0..length)
        .map(|_| {
            *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = *state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            let unit = (value >> 11) as f64 / ((1u64 << 53) as f64);
            ((unit * 2.0 - 1.0) as f32) * bound
        })
        .collect()
}

/// Negative log likelihood averaged over non-padding labels only.
pub(crate) fn masked_next_token_loss(logits: &Tensor, targets: &Tensor) -> Result<(Tensor, usize)> {
    let (batch, sequence, vocabulary) = logits.dims3()?;
    let (target_batch, target_sequence) = targets.dims2()?;
    if batch != target_batch || sequence != target_sequence || vocabulary == 0 {
        return Err(GenerativeError::InvalidInput {
            parameter: "targets",
            reason: "dimensi target tidak cocok dengan logits",
        });
    }
    let flattened_logits = logits.reshape((batch * sequence, vocabulary))?;
    let log_probabilities = candle_nn::ops::log_softmax(&flattened_logits, D::Minus1)?;
    let flattened_targets = targets.reshape((batch * sequence, 1))?;
    let selected = log_probabilities
        .gather(&flattened_targets, 1)?
        .squeeze(1)?
        .neg()?;
    let target_values = targets.to_vec2::<u32>()?;
    let mask_values = target_values
        .iter()
        .flat_map(|row| {
            row.iter()
                .map(|target| if *target == PAD_ID { 0.0_f32 } else { 1.0_f32 })
        })
        .collect::<Vec<_>>();
    let valid_tokens = mask_values.iter().filter(|value| **value != 0.0).count();
    if valid_tokens == 0 {
        return Err(GenerativeError::InvalidInput {
            parameter: "targets",
            reason: "batch hanya memuat label padding",
        });
    }
    let mask = Tensor::from_vec(mask_values, (batch * sequence,), &Device::Cpu)?;
    let loss = selected
        .broadcast_mul(&mask)?
        .sum_all()?
        .affine(1.0 / valid_tokens as f64, 0.0)?;
    Ok((loss, valid_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::generative::tokenizer::CharacterTokenizer;

    fn tiny_model() -> (CausalLanguageModel, CharacterTokenizer) {
        let tokenizer = CharacterTokenizer::fit(["makan nasi", "minum air"]).unwrap();
        let model = CausalLanguageModel::new(
            TransformerConfig {
                hidden_size: 8,
                layer_count: 1,
                head_count: 2,
                max_context: 8,
                seed: 17,
                device: ComputeDevice::Cpu,
            },
            tokenizer.vocabulary_size(),
        )
        .unwrap();
        (model, tokenizer)
    }

    #[test]
    fn decoder_outputs_one_logit_vector_per_input_token() {
        let (model, tokenizer) = tiny_model();
        let ids = tokenizer.encode_document("makan").unwrap();
        let input = Tensor::from_vec(
            ids[..ids.len() - 1].to_vec(),
            (1, ids.len() - 1),
            &Device::Cpu,
        )
        .unwrap();
        let logits = model.forward(&input).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, ids.len() - 1, tokenizer.vocabulary_size()]
        );
    }

    #[test]
    fn parameter_initialization_is_deterministic_for_a_seed() {
        let (first, tokenizer) = tiny_model();
        let same_seed =
            CausalLanguageModel::new(first.config.clone(), tokenizer.vocabulary_size()).unwrap();
        let mut other_config = first.config.clone();
        other_config.seed += 1;
        let other_seed =
            CausalLanguageModel::new(other_config, tokenizer.vocabulary_size()).unwrap();
        assert_eq!(
            first.parameter_snapshot().unwrap(),
            same_seed.parameter_snapshot().unwrap()
        );
        assert_ne!(
            first.parameter_snapshot().unwrap(),
            other_seed.parameter_snapshot().unwrap()
        );
    }

    #[test]
    fn causal_mask_prevents_future_tokens_from_changing_prefix_logits() {
        let (model, tokenizer) = tiny_model();
        let mut left = tokenizer.encode_prompt("ma").unwrap();
        left.push(tokenizer.encode_characters("k").unwrap()[0]);
        let mut right = left.clone();
        right[3] = tokenizer.encode_characters("n").unwrap()[0];
        let left = Tensor::from_vec(left, (1, 4), &Device::Cpu).unwrap();
        let right = Tensor::from_vec(right, (1, 4), &Device::Cpu).unwrap();
        let left_logits = model.forward(&left).unwrap().to_vec3::<f32>().unwrap();
        let right_logits = model.forward(&right).unwrap().to_vec3::<f32>().unwrap();
        for token in 0..3 {
            for class in 0..left_logits[0][token].len() {
                assert!(
                    (left_logits[0][token][class] - right_logits[0][token][class]).abs() < 1e-5
                );
            }
        }
    }

    #[test]
    fn masked_loss_ignores_pad_labels_and_is_finite() {
        let (model, tokenizer) = tiny_model();
        let doc = tokenizer.encode_document("makan").unwrap();
        let mut input = doc[..doc.len() - 1].to_vec();
        let mut targets = doc[1..].to_vec();
        input.resize(input.len() + 2, PAD_ID);
        targets.resize(targets.len() + 2, PAD_ID);
        let input = Tensor::from_vec(input, (1, doc.len() + 1), &Device::Cpu).unwrap();
        let targets = Tensor::from_vec(targets, (1, doc.len() + 1), &Device::Cpu).unwrap();
        let logits = model.forward(&input).unwrap();
        let (loss, count) = masked_next_token_loss(&logits, &targets).unwrap();
        assert_eq!(count, doc.len() - 1);
        assert!(loss.to_scalar::<f32>().unwrap().is_finite());
    }
}

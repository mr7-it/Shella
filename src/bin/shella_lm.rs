use serde::Serialize;
use shella::{
    CorpusSplit, DatasetSplitConfig, GeneratedText, GenerationConfig, GenerationRepetitionMetrics,
    GenerativeLanguageModel, TrainingConfig, TransformerConfig,
};
use std::{collections::BTreeMap, error::Error, fs, path::PathBuf, time::Instant};

fn main() {
    if let Err(error) = run() {
        eprintln!("shella-lm: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let command = args
        .next()
        .ok_or("usage: shella-lm <train|generate|evaluate|metrics> --option value")?;
    let mut options = BTreeMap::new();
    while let Some(key) = args.next() {
        if !key.starts_with("--") {
            return Err(format!("opsi harus diawali --: {key}").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("nilai untuk {key} belum diberikan"))?;
        if options.insert(key.clone(), value).is_some() {
            return Err(format!("opsi duplikat: {key}").into());
        }
    }
    match command.as_str() {
        "train" => train_command(options),
        "generate" => generate_command(options),
        "evaluate" => evaluate_command(options),
        "metrics" => metrics_command(options),
        _ => Err("perintah harus train, generate, evaluate, atau metrics".into()),
    }
}

fn train_command(options: BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    reject_unknown(
        &options,
        &[
            "--dataset",
            "--checkpoint",
            "--epochs",
            "--batch-size",
            "--learning-rate",
            "--weight-decay",
            "--gradient-clip",
            "--seed",
            "--split-seed",
            "--hidden-size",
            "--layers",
            "--heads",
            "--context",
        ],
    )?;
    let dataset_path = value_or(
        &options,
        "--dataset",
        "data/language/nusax/sentiment/indonesian/train.csv",
    );
    let checkpoint_path = required(&options, "--checkpoint")?;
    let seed = parse_or(&options, "--seed", 2026_u64)?;
    let split_config = DatasetSplitConfig {
        seed: parse_or(&options, "--split-seed", 42_u64)?,
        ..DatasetSplitConfig::default()
    };
    let started = Instant::now();
    let corpus = if dataset_path.ends_with(".json") {
        let prepared = CorpusSplit::load_prepared_json(&dataset_path)?;
        if prepared.manifest.split.seed != split_config.seed {
            return Err("--split-seed harus sama dengan seed pada manifest korpus JSON".into());
        }
        prepared
    } else {
        GenerativeLanguageModel::load_nusax_training_corpus(&dataset_path, split_config)?
    };
    let config = TransformerConfig {
        hidden_size: parse_or(&options, "--hidden-size", 32_usize)?,
        layer_count: parse_or(&options, "--layers", 1_usize)?,
        head_count: parse_or(&options, "--heads", 2_usize)?,
        max_context: parse_or(&options, "--context", 64_usize)?,
        seed,
        ..TransformerConfig::default()
    };
    let mut model = GenerativeLanguageModel::from_corpus(config, &corpus)?;
    let training_config = TrainingConfig {
        epochs: parse_or(&options, "--epochs", 6_usize)?,
        batch_size: parse_or(&options, "--batch-size", 8_usize)?,
        learning_rate: parse_or(&options, "--learning-rate", 0.002_f64)?,
        weight_decay: parse_or(&options, "--weight-decay", 0.01_f64)?,
        gradient_clip_norm: parse_or(&options, "--gradient-clip", 1.0_f64)?,
        seed,
    };
    // Validation can be compared before/after training; held-out test remains untouched
    // until all optimizer updates and checkpoint serialization are complete.
    let baseline_validation = model.evaluate(&corpus.validation, training_config.batch_size)?;
    let report = model.train(&corpus, training_config)?;
    let final_validation = model.evaluate(&corpus.validation, report.config.batch_size)?;
    let checkpoint_sha256 = model.save_checkpoint(checkpoint_path)?;

    // Test split baru dipakai sesudah semua optimizer step selesai dan checkpoint ditulis.
    let restored = GenerativeLanguageModel::load_checkpoint(checkpoint_path)?;
    let final_test = restored.evaluate(&corpus.test, report.config.batch_size)?;
    let training_model_tokens = encoded_token_count(&restored, &corpus.training)?;
    let validation_model_tokens = encoded_token_count(&restored, &corpus.validation)?;
    let test_model_tokens = encoded_token_count(&restored, &corpus.test)?;
    let output = TrainRunReport {
        source: &corpus.manifest.source,
        split: &corpus.manifest.split,
        source_file_sha256: &corpus.manifest.source.source_file_sha256,
        source_license: &corpus.manifest.source.license_id,
        training_characters: corpus.manifest.training_characters,
        validation_characters: corpus.manifest.validation_characters,
        test_characters: corpus.manifest.test_characters,
        training_model_tokens,
        validation_model_tokens,
        test_model_tokens,
        baseline_validation,
        final_validation,
        training: report,
        final_test,
        parameter_count: restored.parameter_count(),
        checkpoint_path: PathBuf::from(checkpoint_path),
        checkpoint_sha256,
        training_and_evaluation_seconds: started.elapsed().as_secs_f64(),
        notes: [
            "Source, license basis, checksums, and document split are recorded in the corpus manifest.",
            "Tokenizer is fit on the model training split only.",
            "Model-token counts use CharacterTokenizer encoding and include BOS/EOS per document.",
            "Test split is evaluated only after training; validation baseline/final metrics are paired.",
            "Checkpoint contains model weights and provenance/metrics, not AdamW optimizer state.",
        ],
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn generate_command(options: BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    reject_unknown(
        &options,
        &[
            "--checkpoint",
            "--prompt",
            "--max-new-tokens",
            "--temperature",
            "--top-k",
            "--top-p",
            "--seed",
        ],
    )?;
    let checkpoint = required(&options, "--checkpoint")?;
    let prompt = required(&options, "--prompt")?;
    let temperature = parse_or(&options, "--temperature", 0.8_f64)?;
    let top_k_value: usize = parse_or(&options, "--top-k", 8_usize)?;
    let top_p_value: f64 = parse_or(&options, "--top-p", 0.9_f64)?;
    let config = GenerationConfig {
        max_new_tokens: parse_or(&options, "--max-new-tokens", 80_usize)?,
        temperature,
        top_k: if temperature == 0.0 || top_k_value == 0 {
            None
        } else {
            Some(top_k_value)
        },
        top_p: if temperature == 0.0 || top_p_value == 0.0 {
            None
        } else {
            Some(top_p_value)
        },
        seed: parse_or(&options, "--seed", 2026_u64)?,
    };
    let model = GenerativeLanguageModel::load_checkpoint(checkpoint)?;
    let generated = model.generate(prompt, config)?;
    println!("{}", serde_json::to_string_pretty(&generated)?);
    Ok(())
}

#[derive(Serialize)]
struct GenerationMetricsReport {
    repetition: GenerationRepetitionMetrics,
}

fn metrics_command(options: BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    reject_unknown(&options, &["--input-json"])?;
    let input = required(&options, "--input-json")?;
    if fs::metadata(input)?.len() > 1_000_000 {
        return Err("berkas JSON generasi melebihi batas 1 MiB".into());
    }
    let generated: GeneratedText = serde_json::from_slice(&fs::read(input)?)?;
    let report = GenerationMetricsReport {
        repetition: generated.repetition_metrics(),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn evaluate_command(options: BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    reject_unknown(&options, &["--checkpoint", "--dataset", "--batch-size"])?;
    let checkpoint_path = required(&options, "--checkpoint")?;
    let dataset_path = value_or(
        &options,
        "--dataset",
        "data/language/nusax/sentiment/indonesian/train.csv",
    );
    let model = GenerativeLanguageModel::load_checkpoint(checkpoint_path)?;
    let manifest = model
        .corpus_manifest()
        .ok_or("checkpoint tidak memiliki manifest corpus")?;
    let split = DatasetSplitConfig {
        seed: manifest.split.seed,
        training_fraction: manifest.split.training_fraction_ppm as f64 / 1_000_000.0,
        validation_fraction: manifest.split.validation_fraction_ppm as f64 / 1_000_000.0,
        test_fraction: manifest.split.test_fraction_ppm as f64 / 1_000_000.0,
    };
    let corpus = if dataset_path.ends_with(".json") {
        CorpusSplit::load_prepared_json(&dataset_path)?
    } else {
        CorpusSplit::load_nusax_training(
            dataset_path,
            &shella::language::LicenseRegistry::verified_sources(),
            split,
        )?
    };
    if corpus.manifest != *manifest {
        return Err("dataset/provenance/split tidak sama dengan checkpoint".into());
    }
    let batch_size = parse_or(&options, "--batch-size", 8_usize)?;
    let metrics = model.evaluate(&corpus.test, batch_size)?;
    println!("{}", serde_json::to_string_pretty(&metrics)?);
    Ok(())
}

#[derive(Serialize)]
struct TrainRunReport<'a> {
    source: &'a shella::CorpusSourceManifest,
    split: &'a shella::SplitManifest,
    source_file_sha256: &'a str,
    source_license: &'a str,
    training_characters: usize,
    validation_characters: usize,
    test_characters: usize,
    training_model_tokens: usize,
    validation_model_tokens: usize,
    test_model_tokens: usize,
    baseline_validation: shella::EvaluationMetrics,
    final_validation: shella::EvaluationMetrics,
    training: shella::TrainingReport,
    final_test: shella::EvaluationMetrics,
    parameter_count: usize,
    checkpoint_path: PathBuf,
    checkpoint_sha256: String,
    training_and_evaluation_seconds: f64,
    notes: [&'static str; 5],
}

fn encoded_token_count(
    model: &GenerativeLanguageModel,
    records: &[shella::CorpusRecord],
) -> Result<usize, Box<dyn Error>> {
    let mut total = 0_usize;
    for record in records {
        let encoded = model.tokenizer().encode_document(&record.text)?.len();
        total = total
            .checked_add(encoded)
            .ok_or("jumlah token melampaui kapasitas integer")?;
    }
    Ok(total)
}

fn required<'a>(
    options: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, Box<dyn Error>> {
    options
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("opsi wajib {key}").into())
}

fn value_or(options: &BTreeMap<String, String>, key: &str, default: &str) -> String {
    options
        .get(key)
        .cloned()
        .unwrap_or_else(|| default.to_owned())
}

fn parse_or<T>(
    options: &BTreeMap<String, String>,
    key: &str,
    default: T,
) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    match options.get(key) {
        Some(value) => Ok(value.parse::<T>()?),
        None => Ok(default),
    }
}

fn reject_unknown(
    options: &BTreeMap<String, String>,
    allowed: &[&str],
) -> Result<(), Box<dyn Error>> {
    if let Some(unknown) = options.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("opsi tidak dikenal: {unknown}").into());
    }
    Ok(())
}

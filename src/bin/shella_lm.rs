use serde::Serialize;
use shella::{
    CorpusSplit, DatasetSplitConfig, GenerationConfig, GenerativeLanguageModel, TrainingConfig,
    TransformerConfig,
};
use std::{collections::BTreeMap, error::Error, path::PathBuf, time::Instant};

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
        .ok_or("usage: shella-lm <train|generate|evaluate> --option value")?;
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
        _ => Err("perintah harus train, generate, atau evaluate".into()),
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
    let corpus = GenerativeLanguageModel::load_nusax_training_corpus(&dataset_path, split_config)?;
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
    let output = TrainRunReport {
        source: &corpus.manifest.source,
        split: &corpus.manifest.split,
        source_file_sha256: &corpus.manifest.source.source_file_sha256,
        source_license: &corpus.manifest.source.license_id,
        training_characters: corpus.manifest.training_characters,
        validation_characters: corpus.manifest.validation_characters,
        test_characters: corpus.manifest.test_characters,
        baseline_validation,
        final_validation,
        training: report,
        final_test,
        parameter_count: restored.parameter_count(),
        checkpoint_path: PathBuf::from(checkpoint_path),
        checkpoint_sha256,
        training_and_evaluation_seconds: started.elapsed().as_secs_f64(),
        notes: [
            "NusaX sentiment text only; label column is not a generation target.",
            "Tokenizer is fit on the model training split only.",
            "No official validation/test text and no NusaX-MT duplicate column enters optimizer updates.",
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
    let corpus = CorpusSplit::load_nusax_training(
        dataset_path,
        &shella::language::LicenseRegistry::verified_sources(),
        split,
    )?;
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

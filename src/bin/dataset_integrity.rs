use shella::{DatasetId, DatasetIntegrityManifest, LicenseRegistry};
use std::{env, error::Error, fs::File, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = env::args_os().skip(1).collect::<Vec<_>>();
    match arguments.first().and_then(|value| value.to_str()) {
        Some("verify") if arguments.len() == 3 => {
            let manifest = DatasetIntegrityManifest::load(&arguments[2])?;
            let report = manifest.verify_directory(Path::new(&arguments[1]))?;
            println!(
                "verified_files={} verified_bytes={}",
                report.files_verified, report.total_bytes
            );
        }
        Some("generate") if arguments.len() == 3 => {
            let mut manifest = DatasetIntegrityManifest::create(Path::new(&arguments[1]))?;
            manifest.external_files = DatasetIntegrityManifest::bundled()?.external_files;
            manifest.save(&arguments[2])?;
            println!("manifest_files={}", manifest.files.len());
        }
        Some("verify-source") if arguments.len() == 4 => {
            let dataset_id = arguments[1].to_str().ok_or("dataset ID harus UTF-8")?;
            let dataset_id: DatasetId = serde_json::from_str(&format!("\"{dataset_id}\""))?;
            let source_path = arguments[2].to_str().ok_or("source path harus UTF-8")?;
            let file = File::open(&arguments[3])?;
            let length = file.metadata()?.len();
            LicenseRegistry::verified_sources().verify_file_reader(
                dataset_id,
                source_path,
                length,
                file,
            )?;
            println!("verified_source={dataset_id:?}:{source_path}");
        }
        _ => {
            return Err("usage: dataset-integrity <verify|generate> <dataset-directory> <manifest.json>; dataset-integrity verify-source <dataset-id> <source-path> <file>".into());
        }
    }
    Ok(())
}

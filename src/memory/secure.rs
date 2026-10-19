use super::{
    persistence,
    record::{MemoryRecord, StoreData},
};
use crate::SimError;
use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct Envelope {
    format_version: u32,
    nonce: Vec<u8>,
    ciphertext: Vec<u8>,
    signature: Vec<u8>,
}

/// Store terenkripsi dan bertanda tangan; lock hanya diambil selama transaksi mutasi.
#[derive(Debug)]
pub struct SecureMemoryStore {
    path: PathBuf,
    encryption_key: [u8; 32],
    signing_key: SigningKey,
    data: StoreData,
}

impl SecureMemoryStore {
    /// Membuka store secure. Kunci tidak pernah ditulis ke disk.
    pub fn open(
        path: impl AsRef<Path>,
        encryption_key: [u8; 32],
        signing_key: [u8; 32],
    ) -> Result<Self, SimError> {
        let path = normalize_path(path.as_ref())?;
        let signing_key = SigningKey::from_bytes(&signing_key);
        let data = if path.exists() {
            read_secure(&path, &encryption_key, &signing_key.verifying_key())?
        } else {
            empty_data()
        };
        let store = Self {
            path,
            encryption_key,
            signing_key,
            data,
        };
        if !store.path.exists() {
            store.commit(store.data.clone())?;
        }
        Ok(store)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.records.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.records.is_empty()
    }
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MemoryRecord> {
        self.data.records.get(key)
    }

    /// Transaksi read-modify-write dengan lock eksklusif per operasi; writer lain tidak ditolak saat open.
    pub fn put(
        &mut self,
        key: impl Into<String>,
        value: Value,
        tags: &[String],
    ) -> Result<String, SimError> {
        let key = key.into();
        persistence::validate_field("key", &key)?;
        let tags = persistence::normalize_tags(tags)?;
        self.transaction(|data| {
            let id = if let Some(record) = data.records.get(&key) {
                record.id.clone()
            } else {
                let id = format!("rec-{:#016x}", data.next_id).replace("0x", "");
                data.next_id = data
                    .next_id
                    .checked_add(1)
                    .ok_or(SimError::CounterOverflow {
                        counter: "ID secure memori",
                    })?;
                id
            };
            data.records.insert(
                key.clone(),
                MemoryRecord {
                    id: id.clone(),
                    key: key.clone(),
                    value: value.clone(),
                    tags: tags.clone(),
                    embedding: None,
                    expires_at_unix: None,
                },
            );
            Ok(id)
        })
    }

    fn transaction<T>(
        &mut self,
        operation: impl FnOnce(&mut StoreData) -> Result<T, SimError>,
    ) -> Result<T, SimError> {
        let mut lock_path = self.path.as_os_str().to_os_string();
        lock_path.push(".secure.lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(PathBuf::from(lock_path))
            .map_err(|error| io_error("buka secure lock", error))?;
        lock.lock_exclusive()
            .map_err(|error| io_error("kunci secure store", error))?;
        let mut data = if self.path.exists() {
            read_secure(
                &self.path,
                &self.encryption_key,
                &self.signing_key.verifying_key(),
            )?
        } else {
            empty_data()
        };
        let result = operation(&mut data)?;
        persistence::validate_store(&data)?;
        self.write_secure(&data)?;
        self.data = data;
        let _ = FileExt::unlock(&lock);
        Ok(result)
    }

    fn commit(&self, data: StoreData) -> Result<(), SimError> {
        self.write_secure(&data)
    }

    fn write_secure(&self, data: &StoreData) -> Result<(), SimError> {
        let plaintext =
            serde_json::to_vec(data).map_err(|error| SimError::SerializationFailed {
                subsystem: "secure memori",
                reason: error.to_string(),
            })?;
        let cipher = Aes256Gcm::new_from_slice(&self.encryption_key)
            .map_err(|_| invalid("kunci AES invalid"))?;
        let mut nonce_bytes = [0_u8; 12];
        getrandom::fill(&mut nonce_bytes)
            .map_err(|_| invalid("sumber nonce secure tidak tersedia"))?;
        let ciphertext = cipher
            .encrypt(Nonce::from_slice(&nonce_bytes), plaintext.as_ref())
            .map_err(|_| invalid("enkripsi secure store gagal"))?;
        let signature = self.signing_key.sign(&ciphertext).to_bytes().to_vec();
        let envelope = Envelope {
            format_version: FORMAT_VERSION,
            nonce: nonce_bytes.to_vec(),
            ciphertext,
            signature,
        };
        let bytes =
            serde_json::to_vec(&envelope).map_err(|error| SimError::SerializationFailed {
                subsystem: "secure memori",
                reason: error.to_string(),
            })?;
        let parent = self.path.parent().unwrap_or(Path::new("."));
        let mut name = self
            .path
            .file_name()
            .ok_or_else(|| invalid("path secure invalid"))?
            .to_os_string();
        name.push(format!(
            ".tmp.{}.{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let temp = parent.join(name);
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)
            .map_err(|error| io_error("buat secure temporary", error))?;
        use std::io::Write;
        file.write_all(&bytes)
            .map_err(|error| io_error("tulis secure store", error))?;
        file.sync_all()
            .map_err(|error| io_error("sync secure store", error))?;
        drop(file);
        fs::rename(&temp, &self.path).map_err(|error| io_error("pasang secure store", error))?;
        Ok(())
    }
}

fn empty_data() -> StoreData {
    StoreData {
        format_version: persistence::FORMAT_VERSION,
        next_id: 1,
        records: BTreeMap::new(),
    }
}
fn read_secure(
    path: &Path,
    key: &[u8; 32],
    verifying: &VerifyingKey,
) -> Result<StoreData, SimError> {
    let bytes = fs::read(path).map_err(|error| io_error("baca secure store", error))?;
    let envelope: Envelope =
        serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
    if envelope.format_version != FORMAT_VERSION
        || envelope.nonce.len() != 12
        || envelope.signature.len() != 64
    {
        return Err(SimError::UnsupportedDataVersion {
            subsystem: "secure memori",
            version: envelope.format_version,
        });
    }
    let signature = Signature::from_slice(&envelope.signature)
        .map_err(|_| invalid("signature secure invalid"))?;
    verifying
        .verify(&envelope.ciphertext, &signature)
        .map_err(|_| invalid("signature secure tidak cocok"))?;
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| invalid("kunci AES invalid"))?;
    let plaintext = cipher
        .decrypt(
            Nonce::from_slice(&envelope.nonce),
            envelope.ciphertext.as_ref(),
        )
        .map_err(|_| invalid("dekripsi secure store gagal"))?;
    let data: StoreData =
        serde_json::from_slice(&plaintext).map_err(|error| invalid(error.to_string()))?;
    persistence::validate_store(&data)?;
    Ok(data)
}
fn normalize_path(path: &Path) -> Result<PathBuf, SimError> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| invalid("path secure invalid"))?;
    parent
        .canonicalize()
        .map(|p| p.join(name))
        .map_err(|error| io_error("normalisasi secure path", error))
}
fn invalid(reason: impl Into<String>) -> SimError {
    SimError::CorruptData {
        subsystem: "secure memori",
        reason: reason.into(),
    }
}
fn io_error(operation: &'static str, error: impl Into<std::io::Error>) -> SimError {
    let error = error.into();
    SimError::Io {
        operation,
        kind: error.kind(),
    }
}

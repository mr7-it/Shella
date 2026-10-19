use serde_json::json;
use shella::{
    IsotonicCalibrator, LearningMemory, MemoryStore, PlattCalibrator, SecureMemoryStore,
    TextEmbedding,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::Duration,
};

static ID: AtomicU64 = AtomicU64::new(0);
fn path(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "shella-{prefix}-{}-{}.db",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ))
}
fn cleanup(path: &PathBuf, secure: bool) {
    let _ = fs::remove_file(path);
    let mut lock = path.as_os_str().to_os_string();
    lock.push(if secure { ".secure.lock" } else { ".lock" });
    let _ = fs::remove_file(lock);
}

#[test]
fn embedding_semantik_mengurutkan_teks_serupa() {
    let first = TextEmbedding::from_text("kucing hitam tidur", 256).expect("embedding");
    let second = TextEmbedding::from_text("kucing hitam sedang tidur", 256).expect("embedding");
    let unrelated = TextEmbedding::from_text("pesawat terbang tinggi", 256).expect("embedding");
    let similar = first
        .cosine_similarity(second.values())
        .expect("similarity");
    let unlike = first
        .cosine_similarity(unrelated.values())
        .expect("similarity");
    assert!(similar > unlike);

    let path = path("semantic");
    let mut store = MemoryStore::open(&path).expect("store");
    store
        .put_with_embedding("cat", json!({"kind":"cat"}), &[], Some(first.values()))
        .expect("put");
    store
        .put_with_embedding(
            "plane",
            json!({"kind":"plane"}),
            &[],
            Some(unrelated.values()),
        )
        .expect("put");
    let result = store.search_embedding(second.values(), 1);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0.key(), "cat");
    drop(store);
    cleanup(&path, false);
}

#[test]
fn ttl_dihapus_otomatis_saat_restart_dan_text_learning_memakai_embedding() {
    let ttl_path = path("ttl");
    {
        let mut store = MemoryStore::open(&ttl_path).expect("store");
        store
            .put_with_ttl("short", json!(1), &[], Duration::from_secs(0))
            .expect("ttl");
    }
    thread::sleep(Duration::from_millis(1100));
    let store = MemoryStore::open(&ttl_path).expect("reopen");
    assert!(store.get("short").is_none());
    drop(store);
    cleanup(&ttl_path, false);

    let learning_path = path("learning-text");
    let mut memory = LearningMemory::open(&learning_path).expect("memory");
    memory
        .record_text_feedback("bahasa", "kucing hitam tidur", Some(1), Some(1))
        .expect("record");
    memory
        .record_text_feedback("bahasa", "pesawat terbang tinggi", Some(2), Some(2))
        .expect("record");
    let recalls = memory
        .recall_text_similar("kucing hitam sedang tidur", 1)
        .expect("recall");
    assert_eq!(recalls[0].expected_action(), Some(1));
    drop(memory);
    cleanup(&learning_path, false);
}

#[test]
fn secure_store_mengenkripsi_menandatangani_dan_mengizinkan_writer_bergantian() {
    let path = path("secure");
    let encryption_key = [7_u8; 32];
    let signing_key = [9_u8; 32];
    {
        let mut first =
            SecureMemoryStore::open(&path, encryption_key, signing_key).expect("secure open");
        first
            .put("secret", json!({"message":"plaintext-rahasia"}), &[])
            .expect("put");
        let bytes = fs::read(&path).expect("ciphertext");
        assert!(!String::from_utf8_lossy(&bytes).contains("plaintext-rahasia"));
        let mut second =
            SecureMemoryStore::open(&path, encryption_key, signing_key).expect("multi writer open");
        second
            .put("second", json!({"v":2}), &[])
            .expect("transaction second");
    }
    let reopened =
        SecureMemoryStore::open(&path, encryption_key, signing_key).expect("secure reopen");
    assert_eq!(reopened.len(), 2);
    assert_eq!(
        reopened.get("secret").expect("secret").value(),
        &json!({"message":"plaintext-rahasia"})
    );
    drop(reopened);
    let mut bytes = fs::read(&path).expect("read");
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&path, bytes).expect("tamper");
    assert!(SecureMemoryStore::open(&path, encryption_key, signing_key).is_err());
    cleanup(&path, true);
}

#[test]
fn platt_dan_isotonic_benar_benar_fit_split_validasi() {
    let samples = [
        (-2.0, false),
        (-1.0, false),
        (0.0, false),
        (1.0, true),
        (2.0, true),
        (3.0, true),
    ];
    let platt = PlattCalibrator::fit(&samples, 500, 1.0).expect("fit platt");
    assert!(platt.predict(2.0) > platt.predict(-2.0));
    let isotonic = IsotonicCalibrator::fit(&samples).expect("fit isotonic");
    assert!(isotonic.predict(2.0) >= isotonic.predict(-2.0));
    assert!(
        isotonic
            .blocks()
            .iter()
            .all(|(_, probability)| (0.0..=1.0).contains(probability))
    );
}

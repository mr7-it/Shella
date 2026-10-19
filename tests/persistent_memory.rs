use serde_json::json;
use shella::{MemoryStore, SimError};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
static ID: AtomicU64 = AtomicU64::new(0);
fn path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "shella-memory-{}-{}.json",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn remove_store_files(file: &PathBuf) {
    fs::remove_file(file).expect("database dihapus");
    let mut lock_path = file.as_os_str().to_os_string();
    lock_path.push(".lock");
    fs::remove_file(PathBuf::from(lock_path)).expect("sidecar lock dihapus");
}

#[test]
fn memory_crud_search_ids_stabil_dan_restart() {
    let file = path();
    let tags = vec!["kucing".to_owned(), "hewan".to_owned(), "hewan".to_owned()];
    let id;
    {
        let mut store = MemoryStore::open(&file).expect("database baru");
        id = store
            .put("cat/1", json!({"name":"Mimi","weight":3}), &tags)
            .expect("insert");
        assert_eq!(store.len(), 1);
        assert_eq!(
            store.get_by_id(&id).expect("ID ada").tags(),
            &["hewan", "kucing"]
        );
        let updated_id = store
            .put("cat/1", json!({"name":"Mimi","weight":4}), &[])
            .expect("upsert");
        assert_eq!(updated_id, id);
        store
            .put("cat/2", json!({"name":"Oyen"}), &["kucing".to_owned()])
            .expect("cat2");
        assert_eq!(store.search_prefix("cat/").len(), 2);
        assert_eq!(store.search_tag("kucing").len(), 1);
    }
    {
        let mut reopened = MemoryStore::open(&file).expect("dibuka setelah drop store");
        assert_eq!(reopened.get("cat/1").expect("record pulih").id(), id);
        assert_eq!(
            reopened.get("cat/1").expect("record pulih").value(),
            &json!({"name":"Mimi","weight":4})
        );
        assert!(reopened.delete("cat/2").expect("hapus"));
        assert!(!reopened.delete("missing").expect("hapus idempotent"));
        assert_eq!(reopened.len(), 1);
    }
    let final_store = MemoryStore::open(&file).expect("delete bertahan setelah restart");
    assert_eq!(final_store.len(), 1);
    drop(final_store);
    remove_store_files(&file);
}

#[test]
fn memory_store_menolak_writer_kedua_dan_melepas_lock_setelah_drop() {
    let file = path();
    let store = MemoryStore::open(&file).expect("writer pertama mendapat lock");
    let alias = file
        .parent()
        .expect("direktori temp")
        .join(".")
        .join(file.file_name().expect("nama file"));
    assert!(matches!(
        MemoryStore::open(alias),
        Err(SimError::Io {
            operation: "kunci memori",
            kind: std::io::ErrorKind::WouldBlock,
        })
    ));

    drop(store);
    MemoryStore::open(&file).expect("lock dilepas setelah store ditutup");
    remove_store_files(&file);
}

#[test]
fn memory_menolak_input_invalid_tanpa_mutasi_dan_mendeteksi_korupsi() {
    let file = path();
    let mut store = MemoryStore::open(&file).expect("database baru");
    assert!(matches!(
        store.put("", json!(1), &[]),
        Err(SimError::InvalidInput { .. })
    ));
    assert_eq!(store.len(), 0);
    store.put("ok", json!({"v":1}), &[]).expect("record");
    drop(store);
    let mut bytes = fs::read(&file).expect("baca store");
    let index = bytes
        .iter()
        .position(|byte| *byte == b'{')
        .expect("payload byte");
    bytes[index] = b'[';
    fs::write(&file, bytes).expect("rusakkan store");
    assert!(matches!(
        MemoryStore::open(&file),
        Err(SimError::CorruptData {
            subsystem: "memori",
            ..
        })
    ));
    remove_store_files(&file);
}

#[test]
fn memory_menolak_versi_schema_yang_tidak_dikenal() {
    let file = path();
    let store = MemoryStore::open(&file).expect("database baru");
    drop(store);
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&fs::read(&file).expect("baca database")).expect("envelope JSON");
    envelope["format_version"] = json!(99);
    fs::write(
        &file,
        serde_json::to_vec(&envelope).expect("encode envelope"),
    )
    .expect("tulis schema versi mendatang");
    assert!(matches!(
        MemoryStore::open(&file),
        Err(SimError::UnsupportedDataVersion {
            subsystem: "memori",
            version: 99
        })
    ));
    remove_store_files(&file);
}

#[test]
fn memory_store_child_process_reopens() {
    let Ok(file) = std::env::var("SHELLA_MEMORY_REOPEN_TEST") else {
        return;
    };
    let mut reopened = MemoryStore::open(file).expect("database dibuka di proses baru");
    assert_eq!(
        reopened.get("restart/record").expect("record ada").value(),
        &json!({"counter":1})
    );
    reopened
        .put(
            "restart/record",
            json!({"counter":2}),
            &["child-process".to_owned()],
        )
        .expect("update dari proses baru");
}

#[test]
fn memory_store_child_writer_ditolak() {
    let Ok(file) = std::env::var("SHELLA_MEMORY_LOCK_TEST") else {
        return;
    };
    assert!(matches!(
        MemoryStore::open(file),
        Err(SimError::Io {
            operation: "kunci memori",
            kind: std::io::ErrorKind::WouldBlock,
        })
    ));
}

#[test]
fn memory_store_lock_mencegah_writer_dari_proses_lain() {
    let file = path();
    let store = MemoryStore::open(&file).expect("writer utama mendapat lock");
    let child = Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "memory_store_child_writer_ditolak",
            "--nocapture",
        ])
        .env("SHELLA_MEMORY_LOCK_TEST", &file)
        .output()
        .expect("proses writer kedua berjalan");
    assert!(
        child.status.success(),
        "writer child tidak ditolak: {}",
        String::from_utf8_lossy(&child.stderr)
    );
    drop(store);
    let reopened = MemoryStore::open(&file).expect("lock dilepas setelah writer tutup");
    drop(reopened);
    remove_store_files(&file);
}

#[test]
fn memory_store_survives_real_process_restart_and_new_write() {
    let file = path();
    let id = {
        let mut store = MemoryStore::open(&file).expect("store parent");
        store
            .put("restart/record", json!({"counter":1}), &[])
            .expect("record awal")
    };
    let child = Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "memory_store_child_process_reopens",
            "--nocapture",
        ])
        .env("SHELLA_MEMORY_REOPEN_TEST", &file)
        .output()
        .expect("proses pemulihan dijalankan");
    assert!(
        child.status.success(),
        "child gagal: {}",
        String::from_utf8_lossy(&child.stderr)
    );
    let reopened = MemoryStore::open(&file).expect("hasil child dibuka parent");
    let record = reopened
        .get("restart/record")
        .expect("record setelah restart");
    assert_eq!(record.id(), id);
    assert_eq!(record.value(), &json!({"counter":2}));
    assert_eq!(record.tags(), &["child-process"]);
    drop(reopened);
    remove_store_files(&file);
}

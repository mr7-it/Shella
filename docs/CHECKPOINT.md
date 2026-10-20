# Checkpoint dan pemulihan

API `Checkpoint::save(path, &network, evaluator)` menyimpan `LifNetwork` serta evaluator opsional. `Checkpoint::restore(path)` memeriksa JSON skema versi 1, batas ukuran 256 MiB, checksum FNV-1a, deserialisasi, lalu invariants internal sebelum mengembalikan `RestoredState`. `save_runtime`/`restore_runtime` menyediakan jalur untuk `LifRuntime` satu-neuron. Implementasi dipisah ke `checkpoint/schema.rs`, `save.rs`, `restore.rs`, dan `integrity.rs`.

`Checkpoint::save_knowledge(path, network, class_knowledge, action_neurons, evaluator)` menambahkan satu payload komposit berisi LIF/STDP penuh, model ClassKnowledge/version, mapping tepat tiga action neuron unik, dan evaluator opsional. `restore_knowledge` memverifikasi checksum serta invariants network/model/mapping sebelum mengembalikan `RestoredKnowledgeState`. Regression test membandingkan output inference dan network state sebelum/sesudah round-trip. Memory file tetap store terpisah; salinan class model di checkpoint adalah snapshot konsisten untuk resume bersama network.

State network mencakup konfigurasi heterogen tiap neuron, membran/refractory, penghitung dan riwayat spike, synapse/batas bobot, antrean event tertunda dan waktu event, jam jaringan, STDP/reward config, serta pre/post/eligibility traces. Evaluator opsional membawa reward config dan metrik kumulatif. Konfigurasi statis SDK (encoder/policy) serta file MemoryStore bukan bagian snapshot jaringan.

Penulisan menggunakan temp file eksklusif di direktori tujuan, `sync_all` atas isi file, lalu `rename`. Pada Unix file temp dibuat mode pemilik `0600`. Sync direktori sesudah rename best-effort; durability terhadap semua kegagalan daya tidak dijanjikan. Atomic replace mengikuti semantik OS/filesystem; perilaku diverifikasi pada Linux sesi ini. `serde_json` memakai `float_roundtrip` untuk mempertahankan bit f64 pada round-trip.

Checksum mendeteksi kerusakan tak disengaja; ia bukan MAC, signature, atau proteksi dari penyerang yang dapat memperbarui checksum. Jangan menerima checkpoint dari sumber tidak tepercaya tanpa validasi kebijakan aplikasi. Format baru memerlukan versi baru/migrasi eksplisit; versi tidak dikenal ditolak. Semua invariants diperiksa setelah deserialize, sehingga tidak ada state parsial yang dikembalikan.

Contoh:

```rust
use shella::{Checkpoint, LifNetwork, RewardEvaluator};
# fn save(path: &str, network: &LifNetwork, evaluator: Option<&RewardEvaluator>) -> Result<(), Box<dyn std::error::Error>> {
Checkpoint::save(path, network, evaluator)?;
let restored = Checkpoint::restore(path)?;
# let _ = restored;
# Ok(())
# }
```

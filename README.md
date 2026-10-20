# Shella

**Shella** adalah library dan runtime neuromorfik dalam Rust. Implementasi mencakup neuron LIF, jaringan multi-neuron sinkron, STDP dan reward-modulated learning, checkpoint, memori JSON file-backed, keputusan berbasis skor yang dapat diaudit, preprocessing teks deterministik, serta SDK Rust bertipe. Paket Cargo, crate library, dan binari utama semuanya bernama `shella`.

## Kemampuan yang tersedia

- **Simulasi:** integrasi LIF eksak per interval, refraktori, catatan spike, parameter heterogen per neuron, sinaps berarah, delay, agregasi arus, serta commit jaringan dua fase.
- **Pembelajaran:** STDP pair-based, eligibility trace, reward/penalti signed, batas bobot, dan evaluator otomatis yang membaca spike output aktual.
- **Checkpoint:** format JSON berversi untuk runtime tunggal atau state jaringan lengkap, evaluator opsional, checksum integritas, validasi invariants, dan penulisan melalui file sementara/rename.
- **Memori persisten dan adaptif:** upsert/CRUD, ID stabil, pemulihan restart, embedding hashing token/character n-gram, cosine retrieval, TTL otomatis, injeksi arus ke LIF, replay relevan berbasis error, feedback, konsolidasi importance, pelupaan terkontrol, serta deteksi konflik asosiasi.
- **Secure storage:** `SecureMemoryStore` dengan AES-256-GCM, nonce OS, signature Ed25519, dan transaksi read-modify-write dengan lock per mutasi untuk writer bergantian. Kunci dikelola aplikasi, tidak disimpan ke disk.
- **Keputusan:** skor fitur linear, jejak kontribusi, tie-break stabil, abstain berdasarkan ambang, evaluator TP/FP/FN/FPR/FNR, `CalibrationEvaluator`, Platt scaling, dan isotonic regression. Skor/margin bukan probabilitas.
- **Pipeline teks dasar:** normalisasi Unicode lowercase, tokenisasi berbasis aturan, vocabulary deterministik dari korpus, ID token, dan proyeksi bag-of-token ke arus jaringan.
- **Library bahasa Indonesia:** API lexicon terversi dengan `WordID` persisten dan loader source-pinned untuk stopwords-id, UD Indonesian-PUD, NusaX (MT/sentiment/lexicon), serta IndoSum. Data asli dan notice lisensinya disimpan di `data/language/`; GSD/CSUI dan IndoNLI tetap tidak diaktifkan karena izin corpus belum terverifikasi.
- **SDK:** `Engine`/`Session` untuk input arus atau teks, output bertipe, evaluator aksi, decision policy, memory, dan checkpoint jaringan.
- **Reasoning:** `ReasoningEngine` memadukan context window terbatas, goal-directed retrieval, perhatian temporal dari timestamp absolut, memory consolidation/decay, deteksi kontradiksi dan asosiasi label berbukti, dekomposisi uncertainty, compute budget adaptif, replay, kalibrasi Platt ber-ID, serta checkpoint reasoning.
- **Evaluasi/benchmark:** conditioning, interferensi/replay, continual-learning dengan konflik konteks, benchmark M10 12 seed × 4 varian × 16 repetisi dengan temporal ensemble dan adaptive fallback untuk noise 30/50/70 nA, split validation/test, confusion metrics, kalibrasi, timing jaringan, serta holdout UCI Iris nyata.

## Keterbatasan yang disengaja

`LanguageNetwork` menyediakan ID kata/subword/neuron/sinaps stabil, representasi kata terdistribusi, koneksi sinaptik aktual, dan checkpoint bersama jaringan. Lookup vocabulary memakai hashbrown SwissTable, tetapi nilai hash tidak dipakai sebagai ID. `TextEncoder` lama tetap memakai proyeksi modulo untuk kompatibilitas dan dapat mengalami collision. Pipeline teks bukan model semantik penuh, pemahaman bahasa, generator, atau decoder; `TextEmbedding` tetap hashing lexical dan dapat collision. `MemoryStore` lama tetap JSON single-writer; gunakan `SecureMemoryStore` untuk enkripsi/signature dan writer bergantian. Rotasi kunci, backup, serta key management tetap tanggung jawab aplikasi. Benchmark mengukur perilaku protokol sintetis dan tidak membuktikan generalisasi, performa aplikasi, ataupun efisiensi energi.

Integritas berkas dataset terbundel dapat diperiksa dengan
`cargo run --bin dataset-integrity -- verify data/language data/language-integrity.json`.
Manifest mencakup SHA-256, ukuran, format, dan encoding; lihat
[inventaris dan batasan dataset](docs/DATASET_BAHASA_INDONESIA.md).

`ReasoningEngine::confidence()` adalah skor heuristik; bukan probabilitas.
`calibrated_confidence()` hanya tersedia setelah Platt fit menggunakan ID validation
dan ID evaluation non-overlap, dengan Brier/MAE diukur pada data evaluation.
Reasoning checkpoint mengikat state network/context/kalibrator ke fingerprint
LearningMemory terpisah dan menolak restore bila fingerprint berbeda. Asosiasi fitur
dan label hanyalah ko-occurence terukur, bukan bukti kausal intervensional.

## Memulai

Prasyarat: Rust 1.85 atau lebih baru (MSRV di `Cargo.toml` adalah 1.85; edition 2024).

```sh
cargo run --release --locked
cargo run --example interference --locked
cargo run --example generalization --locked
cargo run --example milestone9_continual --release --locked
cargo run --example milestone10_long_term --release --locked
cargo run --example iris_real_dataset --release --locked
cargo build --release --lib --locked
cargo test --all-targets --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo bench --bench learning --locked
cargo bench --bench system --locked
cargo bench --bench language --locked
```

Dalam proyek Rust lain, tambahkan dependensi bernama `shella`, lalu impor API dari crate `shella`. Contoh integrasi ada di [Panduan API](docs/API.md) dan [SDK](docs/SDK.md). Dependency langsung dan transitif tercatat di `Cargo.toml` serta `Cargo.lock`.

## Peta dokumentasi

- [Arsitektur](docs/ARSITEKTUR.md) dan [referensi API](docs/API.md)
- [Model matematis](docs/MODEL_MATEMATIS.md), [jaringan](docs/NETWORK.md), [STDP dan reward](docs/STDP.md)
- [Checkpoint](docs/CHECKPOINT.md), [memori](docs/MEMORI.md), [keputusan](docs/KEPUTUSAN.md)
- [Pipeline teks](docs/BAHASA.md), [SDK](docs/SDK.md), [keamanan](docs/KEAMANAN.md)
- [Evaluator dan benchmark](docs/EVALUATOR_BENCHMARK.md), [pengujian](docs/PENGUJIAN.md), dan [laporan M10](LAPORAN_MILESTONE_10.md)
- [Audit Milestone 5](LAPORAN_MILESTONE_5.md), [progres Milestone 6](LAPORAN_MILESTONE_6.md), dan [hasil validasi Milestone 5](LAPORAN_VALIDASI.md)

Laporan Milestone 1–4 adalah snapshot historis. `LAPORAN_VALIDASI.md` mencatat validasi sebelum eksperimen M6; hasil continual-learning dan gerbang mutu M6 ada di `LAPORAN_MILESTONE_6.md`, sedangkan implementasi dan hasil aktual M10 ada di `LAPORAN_MILESTONE_10.md`.

## Milestone 11 — Self-Learning Brain

M11 menambahkan modul `benchmark::self_learning` dengan split Iris stratified train/validation/test, scaler train-only, stratified 5-fold cross-validation pada pool non-test, deterministic noise, serta `SelfEvaluation`. Runner `examples/milestone11_self_learning.rs` membandingkan LIF/STDP tanpa memory, LIF/STDP dengan `LearningMemory`, dan retrieval eksternal; runner juga menguji persistent memory setelah restart, continual phase, replay relevan, autonomous memory formation, adaptive memory current, dan self-improvement loop.

Jalankan dengan:

```bash
cargo run --release --example milestone11_self_learning --locked
```

Hasil aktual tersimpan pada `results/milestone11_self_learning_release_2026-10-19.csv`. Status M11 dan keterbatasan performa ada di `LAPORAN_MILESTONE_11.md`; pure LIF/STDP belum mengungguli retrieval eksternal dan self-improvement loop belum menunjukkan delta validation positif.

### Hasil perbaikan M11

Setelah baseline di-commit, prototype readout train-only pada jalur LIF/STDP meningkatkan test accuracy neural tanpa memory menjadi `0.886666`, lebih tinggi daripada retrieval eksternal `0.673333`. Self-improvement validation-gated menaikkan validation rata-rata dari `0.900000` menjadi `0.966667` (`+0.066667`) melalui prototype adaptation, feedback memory, replay kandidat, dan rollback bila skor turun. Prototype konsolidasi juga dipersistenkan dan dipulihkan setelah restart.

### Class Knowledge M11

Modul `knowledge` mengganti jalur single-center di runner M11 dengan schema fitur Iris terversi: empat fitur asli dan enam turunan, scaler train-only, statistik, k-means deterministik, pemilihan jumlah prototype per kelas memakai validation (`k = 2, 3, 4, 5, 6, 8`), dan exemplar representatif/boundary. Migrasi eksplisit menyimpan tiga center lama lossless sebagai snapshot, kemudian membangun model aktif dari train/validation.

`ClassKnowledgeModel` disimpan di `LearningMemory` atau `SecureMemoryStore`; `infer_with_lif` mengukur probe baseline dan probe dengan knowledge pada jaringan aktual. Benchmark release lima seed mencatat test accuracy rata-rata `0.973333` untuk mode multiprototype-LIF. Hasil per seed, kandidat validation, metrik per kelas/confusion matrix, dan calibration berada di `results/class_knowledge_enterprise_2026-10-19.csv`, `results/class_knowledge_validation_2026-10-19.csv`, `results/class_knowledge_per_class_2026-10-19.csv`, dan `results/class_knowledge_calibration_2026-10-19.csv`. Ini terbatas pada Iris dan tidak membuktikan kesiapan produksi.

`Checkpoint::save_knowledge` mengemas network LIF/STDP, ClassKnowledge, mapping neuron aksi, dan evaluator dalam satu snapshot tervalidasi untuk restart.

Confidence Class Knowledge memakai Platt one-vs-rest yang fit hanya pada validation; Brier/MAE test dicatat di `results/class_knowledge_calibration_2026-10-19.csv`. Sampel validation terbatas, jadi kalibrasi hanya teruji pada benchmark Iris. Drift detection, conflict registry, Mahalanobis scoring, replay budget khusus knowledge, model registry approval/rollback formal, dan audit trail actor belum tersedia. Rincian status ada di [schema Class Knowledge](docs/KNOWLEDGE_SCHEMA.md).

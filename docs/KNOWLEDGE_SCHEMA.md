# Schema Class Knowledge Iris

Status: **TERBATAS PADA BENCHMARK**. Modul ini memigrasikan jalur prototype Iris M11 ke knowledge model multiprototype dan inference LIF. Ia tidak menyatakan sistem production-ready.

## Versi dan fitur

- `MODEL_SCHEMA_VERSION = 1`; `FEATURE_SCHEMA_VERSION = 2`.
- Kelas ber-ID stabil: `0 Iris-setosa`, `1 Iris-versicolor`, `2 Iris-virginica`.
- Fitur asli: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (cm).
- Fitur turunan: `sepal_area`, `petal_area` (cm^2); `sepal_ratio`, `petal_ratio`, `petal_sepal_length_ratio`, `petal_sepal_area_ratio` (rasio tak berdimensi).
- Formula berurutan: `sepal_length * sepal_width`; `petal_length * petal_width`; `sepal_length / sepal_width`; `petal_length / petal_width`; `petal_length / sepal_length`; `petal_area / sepal_area`.
- Schema mendeskripsikan nama, tipe numerik kontinu, unit, formula, versi, dan status aktif. Ukuran mentah harus finite dan positif; hasil turunan harus finite.
- Min/max scaler fit dari train saja. Fitur konstan dipetakan ke nol. Validation hanya digunakan untuk memilih jumlah prototype.

## Data model dan algoritme

`ClassKnowledgeModel` menyimpan `model_id`, versi schema/model, parent version, approval state awal (`Candidate`), dataset version, training run, seed, created/updated timestamp, scaler, statistik training, validation accuracy, score setiap kandidat k, kalibrator, dan tiga `ClassKnowledge`. Setiap kelas menyimpan ID/nama, jumlah sampel, statistik distribusi, prototype, dan exemplar dengan `ExemplarPolicy` configurable (default dua per kelas, batas API 1..=64). Urutan prioritas: representatif keseluruhan, boundary margin terendah, lalu representatif cluster lain. Exemplar merekam predicted label, confidence, feedback, usage dan retrieval timestamp; saat ini usage/last-retrieved adalah snapshot awal, belum dimutasi saat inference.

Prototype menyimpan vector fitur 10D, member count, statistik lokal (mean, variance, standard deviation, min/max, kuartil, covariance, outlier/borderline count), embedding 10D, versi, importance, usage count, dan last-retrieved timestamp opsional. Exemplar menyimpan data sumber, derived vector, embedding, label/prediksi, confidence Platt, uncertainty, boundary score, importance, feedback, usage, timestamp retrieval, serta dataset/run asal. K-means memakai inisialisasi farthest-first deterministik dari seed, maksimum 100 iterasi, dan tie-break indeks stabil. Kandidat per kelas adalah `k = 2, 3, 4, 5, 6, 8`; pemilihan berdasar validation accuracy dari scorer hybrid (70% squared distance prototype + 30% squared distance exemplar) dan tie memilih k terkecil. Kalibrator Platt one-vs-rest per kelas fit dari validation menggunakan scorer hybrid yang sama. Score/margin bukan probabilitas; calibrated confidence hanya merujuk outcome one-vs-rest pada benchmark.

## Migrasi dan persistence

`LearningMemory::migrate_legacy_prototypes` memerlukan tiga record single-center lama dan mempertahankan nilainya persis di `legacy_snapshot`. Model aktif baru di-fit dari train/validation; center lama tidak diperlakukan sebagai sampel training. Persistence biasa menggunakan checksum dan atomic write `MemoryStore`. `ClassKnowledgeModel::save_secure` memakai `SecureMemoryStore` dengan AES-256-GCM dan Ed25519.

`Checkpoint::save_knowledge` menyimpan satu snapshot LIF/STDP + ClassKnowledge + mapping tiga action-neuron + evaluator. `restore_knowledge` memvalidasi semua komponennya sebelum mengembalikan state. Test restart memastikan inference result dan trajectory network identik. `LearningMemory`/SecureMemoryStore tetap punya persistence tersendiri.

## Jalur LIF

`infer_with_lif` menjalankan probe baseline dan probe berarus knowledge pada clone jaringan, memilih prototype, menginjeksikan afinitas ke action neurons, lalu mengembalikan respons LIF aktual. State jaringan pemanggil hanya diganti setelah seluruh probe knowledge berhasil. Ini tidak menyamakan prediksi prototype dengan output neural.

## Hasil aktual

Runner release pada `data/iris_uci.csv`, lima seed, split stratified, dan scaler train-only mencatat test accuracy mode `multi_prototype_lif`: `17: 0.966667`, `31: 1.000000`, `47: 0.966667`, `73: 0.933333`, `101: 1.000000`; mean `0.973333`. Rata-rata validation score hybrid kandidat: `k=2: 0.920000`, `k=3: 0.966667`, `k=4: 0.946667`, `k=5: 0.933333`, `k=6: 0.933333`, `k=8: 0.920000`. Per-seed result ada di `results/class_knowledge_enterprise_2026-10-19.csv`; pilihan k ada di `results/class_knowledge_validation_2026-10-19.csv`; confusion/per-class metrics ada di `results/class_knowledge_per_class_2026-10-19.csv`; Brier per seed `0.129789..0.151739` dan MAE `0.331431..0.364428` ada di `results/class_knowledge_calibration_2026-10-19.csv`.

## Belum tersedia

- Isotonic calibration untuk ClassKnowledge, per-class Brier/ECE, serta threshold abstain class-specific.
- Pemilihan jarak Euclidean/cosine/Mahalanobis; predictor saat ini memakai squared Euclidean pada fitur min-max.
- Conflict registry, boundary memory tersendiri, exemplar feedback/usage update setelah retrieval, serta priority replay policy.
- Neural signature lengkap (latency/order, synaptic contribution, reward/eligibility history), drift alerts, frequency monitoring, dan OOD evaluation.
- Model registry approval/parent/rollback formal (model hanya menyimpan kandidat/status dasar), actor audit trail, tenant isolation, dan transaksi batch.
- Benchmark lengkap noise-level, OOD, drift, calibration, conflict, forgetting policy, dan replay budget.

Status: schema/features, deterministic clustering, legacy snapshot, persistence/restart, secure round-trip, LIF injection, dan test split isolation **LULUS** pada regression test lokal. Kualitas model **TERBATAS PADA BENCHMARK Iris**. Hal yang belum terimplementasi berstatus **BELUM DIUJI**.
# Memori persisten, embedding, dan secure storage M10

## Class Knowledge M11

`LearningMemory` menyimpan `ClassKnowledgeModel` pada key `class-knowledge/model/<training_run_id>` memakai checksum dan atomic write `MemoryStore`. Model memvalidasi versi schema, class, prototype, dan metadata saat restore. `migrate_legacy_prototypes` memerlukan tiga center lama, mempertahankan nilainya di snapshot, lalu fit multiprototype dari train/validation. `ClassKnowledgeModel::save_secure`/`load_secure` menggunakan `SecureMemoryStore`; jalur enkripsi dan signature existing tidak dilemahkan.

Model memuat sepuluh fitur, min/max training, statistik kelas/prototype, skor kandidat k validation, exemplar ber-budget configurable, metadata model candidate/version, serta kalibrator Platt one-vs-rest yang fit hanya pada validation. `infer_with_lif` menjalankan baseline dan knowledge-injected probe pada network aktual. Drift detector, conflict registry, Mahalanobis, replay budget khusus knowledge, serta formal registry/approval multi-writer belum tersedia. Untuk angka terbaru gunakan [benchmark Class Knowledge](../results/class_knowledge_enterprise_2026-10-19.csv), [per-class metrics](../results/class_knowledge_per_class_2026-10-19.csv), dan [kalibrasi test](../results/class_knowledge_calibration_2026-10-19.csv); angka milestone di bagian bawah adalah snapshot eksperimen lama.

## MemoryStore backward-compatible

`MemoryStore::open(path)` tetap menyediakan format JSON v1, checksum FNV-1a, atomic rename, dan mode single-writer untuk kompatibilitas data lama. `put_with_embedding` menyimpan vektor embedding numerik dan `search_embedding` mengurutkan record berdasarkan cosine similarity. `put_with_ttl` menyimpan TTL absolut; record expired dibersihkan saat open dan dapat dibersihkan eksplisit melalui `purge_expired`.

`TextEmbedding::from_text(text, dimensions)` membuat embedding hashing deterministik dari token dan character 3-gram. `LearningMemory::record_text_feedback` dan `recall_text_similar` memakai embedding 256 dimensi ini. Representasi ini adalah semantic-like lexical retrieval yang reproducible; bukan embedding neural, model bahasa, atau bukti pemahaman semantik.

## SecureMemoryStore

`SecureMemoryStore::open(path, encryption_key, signing_key)` adalah jalur secure baru. Payload record dienkripsi dengan AES-256-GCM menggunakan nonce acak dari OS, lalu ciphertext ditandatangani Ed25519. Signature diverifikasi sebelum dekripsi; modifikasi ciphertext, signature, atau nonce ditolak. Kunci 32-byte tidak pernah ditulis ke file dan harus datang dari secret manager aplikasi.

Berbeda dari `MemoryStore::open`, `SecureMemoryStore` tidak menahan lock selama umur object. Setiap `put` melakukan read-modify-write di bawah exclusive sidecar lock, membaca snapshot terbaru sebelum mutasi, dan menulis atomic encrypted snapshot. Dua instance dapat dibuka bergantian dan tidak menimpa update yang sudah dibuat writer lain. Ini adalah transaksi multi-writer serializable per mutasi, bukan transaksi batch lintas beberapa mutasi.

Secure storage belum melakukan rotasi kunci otomatis, backup, key escrow, atau migrasi otomatis dari JSON plaintext. Aplikasi harus memilih jalur secure sejak awal dan mengelola kunci di luar repository.

## Retrieval dan pembelajaran

`LearningMemory` menyimpan experience dengan features, target, feedback, importance, dan embedding opsional. `recall_similar` memakai cosine feature similarity; `recall_text_similar` memakai embedding teks. `infer_with_memory` menginjeksikan support hasil recall ke neuron aksi LIF, sedangkan `replay_relevant` menjalankan experience melalui input, teacher spike, evaluator reward, dan reward-modulated STDP.

`infer_task_with_memory_adaptive` menolak injeksi ketika similarity di bawah ambang validasi. Pada benchmark M10, temporal voting lima observasi noisy digabungkan dengan fallback tanpa memori pada noise ekstrem. Ini menjaga akurasi ketika evidence memory tidak cukup kuat, alih-alih memaksakan asosiasi lama.

`find_conflicts` mendeteksi target berbeda pada fitur mirip. `consolidate` meningkatkan importance dan melindungi asosiasi. `decay_irrelevant` melakukan pelupaan terkontrol berdasarkan similarity dan importance policy. Retrieval score bukan probabilitas.

## Kalibrasi

`DecisionMetrics` menyediakan accuracy, precision, recall, FPR, FNR, abstain, dan alasan abstain. `CalibrationEvaluator` menghitung Brier score dan mean absolute calibration error dari confidence probabilistik tervalidasi. Untuk melatih kalibrasi dari split validasi, gunakan `PlattCalibrator::fit` atau `IsotonicCalibrator::fit`; keduanya memiliki `predict` yang menghasilkan nilai pada `0..1`. Score linear dan margin decision tidak boleh dianggap probabilitas.

## Batasan

Hash embedding sensitif terhadap bahasa, tokenisasi, dan collision dimensi. SecureMemoryStore bergantung pada keamanan kunci yang diberikan aplikasi dan tidak menjadikan plaintext store lama otomatis aman. Benchmark M10 diperbesar menjadi 12 seed × 4 varian × 16 repetisi dan menguji noise 30/50/70 nA dengan temporal ensemble. `examples/iris_real_dataset.rs` juga menjalankan 120 train/30 test pada dataset UCI Iris di `data/iris_uci.csv`; hasil aktual dicatat di `results/iris_uci_generalization_2026-10-19.csv`. Iris adalah dataset kecil dan mudah, sehingga bukan bukti generalisasi produksi universal.

## M11 self-learning

Runner M11 membentuk memory secara autonomous dari salah prediksi, abstain, evidence rendah, dan sebagian pengalaman benar yang dipilih deterministik. Validation feedback disimpan pada task terpisah lalu dipakai oleh replay relevan sebelum validation diukur ulang. Store dibuka ulang setelah restart untuk menguji persistence.

Hasil aktual M11 menunjukkan memory-assisted LIF meningkatkan accuracy terhadap jalur LIF/STDP tanpa memory, tetapi retrieval eksternal masih lebih tinggi. Self-improvement loop juga belum selalu memperbaiki validation; pada hasil lima seed terakhir delta rata-rata adalah `-0.166667`. Angka ini dipertahankan sebagai hasil negatif yang terukur.

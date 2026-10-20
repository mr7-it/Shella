# Pengujian perilaku dan benchmark

## Gerbang mutu

Perintah utama yang digunakan dari root repository:

```sh
cargo fmt --all -- --check
cargo check --all-targets --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --locked
cargo test --release --locked
cargo run --release --locked
cargo run --release --example milestone10_long_term --locked
cargo bench --bench learning --locked
cargo bench --bench system --locked
cargo bench --bench language --locked
```

MSRV di manifest adalah Rust 1.85. Jalankan gerbang di stable dan MSRV jika perubahan menyentuh code/dependency. `cargo test --all-targets` membangun/menjalankan target benchmark dengan konfigurasi test; perintah `cargo bench` terpisah dipakai untuk angka performa Release. Jangan menyamakan hasil tes kebenaran dengan timing benchmark.

## Peta tes

- `tests/lif_runtime.rs`: persamaan LIF, input invalid, spike/reset/refractory, atomisitas, overflow, dan determinisme.
- `tests/network_stdp.rs`, `tests/milestone3.rs`: topologi, update dua fase, delay, propagasi/inhibisi, konfigurasi heterogen, STDP/eligibility/reward dan rollback.
- `tests/milestone4.rs`: evaluator, reward berdasarkan output aktual, metrik atomik, serta recall dibanding kontrol tanpa update.
- `tests/generalization.rs`: asosiasi cue-to-action baru, held-out amplitudo+distractor, kontrol tanpa learning, dan cue yang tak dilatih.
- `tests/continual_learning.rs`: empat asosiasi dengan fitur cue umum yang berbagi lintas kelas, dua fase belajar, lima seed, split amplitudo terpisah, determinisme, dan forgetting task lama.
- `tests/checkpoint_recovery.rs`: round-trip jaringan/runtime, evaluator, pending event, trace STDP, trajectory resume pada proses baru, file corrupt/truncated, versi tidak didukung, dan I/O.
- `tests/persistent_memory.rs`: CRUD/upsert, ID stabil, prefix/tag retrieval, restart proses nyata, input invalid dan data rusak.
- `tests/decision_engine.rs`: skor/kontribusi terhitung, tie-break, abstain, unknown features, invalid input, overflow skor, dan overflow margin.
- `tests/language_pipeline.rs`: normalisasi, aturan tokenizer Unicode, vocabulary deterministik/unknown, pemisahan train/eval, arus encoder, serta penolakan input tanpa token.
- `tests/language_network.rs`: WordID persisten dan migrasi vocabulary legacy, ID neuron/sinaps stabil, representasi word ke banyak neuron, koneksi sinaps aktual dengan pembaruan STDP, checkpoint restart penuh, dan integrasi reasoning teks-memory.
- `tests/indonesian_vocabulary.rs`, `tests/indonesian_datasets.rs`: normalisasi, ID stabil, lookup, persistensi, penolakan file yang hash-nya tidak cocok, metadata lisensi, serta parser/stopword ke input LanguageNetwork.
- `tests/indonesian_upstream_datasets.rs`: file stopwords, PUD, seluruh 50 CSV NusaX, dan arsip IndoSum yang dibundel diverifikasi terhadap pin sumber; juga menguji parse, tokenizer, evaluasi held-out, round-trip, integrasi vocabulary neural, korupsi input, dan batas fold.
- `tests/indonesian_stopwords_upstream.rs` dan `tests/indonesian_hunspell_upstream.rs` tetap tes upstream `#[ignore]` untuk jalur file eksternal; tes baru memakai aset yang sudah dipin dan dibundel.
- `tests/sdk_integration.rs`: teks -> input neuron -> network terlatih -> reward evaluator -> held-out action -> decision -> memory -> checkpoint.
- `tests/reasoning_engine.rs`: kalibrasi ber-ID dan split leakage, timestamp/migrasi, context window dan isolasi context, consolidation/decay/penggunaan, asosiasi-kontradiksi, uncertainty decomposition, compute budget, goal retrieval, STDP/reward-STDP/replay, determinisme, serta checkpoint/fingerprint restart.
- Unit test evaluator kalibrasi: confidence di luar `0..1` ditolak dan Brier/MAE dihitung dari confidence probabilistik yang diberikan.
- `tests/milestone10_secure_adaptive.rs`: embedding semantic retrieval, TTL restart, enkripsi AES-GCM, signature Ed25519, writer bergantian, Platt scaling, dan isotonic regression.
- `examples/milestone10_long_term.rs`: runner release 12 seed × 4 varian × 16 repetisi untuk restart, retensi, konflik, replay error, pelupaan, noise holdout, confusion metrics, dan kalibrasi; runner memakai jaringan dan memori aktual.
- `examples/iris_real_dataset.rs`: runner data nyata UCI Iris dengan 120 train/30 test, normalisasi train-only, retrieval memory, dan evaluasi noise holdout.

Pada verifikasi M10 di sandbox dengan Rust 1.97.1, `cargo fmt --all -- --check`, `cargo check --all-targets --all-features`, Clippy `-D warnings`, test debug, dan test release lulus setelah adaptive ensemble/gate. Runner M10 dan runner Iris juga lulus serta outputnya cocok dengan CSV tersimpan. Klaim ini tidak menggantikan verifikasi pada MSRV atau perangkat deployment.

Tes integrasi sumber eksternal yang masih berlaku dapat dijalankan terpisah
setelah memperoleh berkas yang tepat pada revisi allowlist:

```sh
SHELLA_STOPWORDS_ID=/path/stopwords-id.json \
cargo test --test indonesian_stopwords_upstream -- --ignored

SHELLA_HUNSPELL_DIC=/path/Indonesia.dic \
SHELLA_HUNSPELL_AFF=/path/Indonesia.aff \
cargo test --test indonesian_hunspell_upstream -- --ignored
```

Tes tersebut menolak byte yang tidak cocok dengan Git blob hash pada registry.
Data stopwords, PUD, NusaX, dan IndoSum yang lisensinya diverifikasi berada di
`data/language/`; GSD, CSUI, dan IndoNLI tidak dimuat karena izin data belum
dapat diverifikasi. Lihat [inventaris sumber dan lisensi](./DATASET_BAHASA_INDONESIA.md).

## Benchmark

`benches/learning.rs` mengukur tugas conditioning 32 epoch / dua pola dan evaluator reward, membandingkan recall learned dengan kontrol tanpa weight update. Harness melakukan warm-up dan tujuh batch × 100 run; hasil spesifik mesin tercatat di `docs/EVALUATOR_BENCHMARK.md`.

`examples/milestone6_continual.rs` menjalankan 5 seed × 32 epoch per fase untuk 4 asosiasi yang berbagi cue; ini benchmark akurasi/retensi, bukan timing. Runner menulis hasil per seed ke `results/milestone6_continual_2026-10-19.csv`.

`benches/system.rs` mengukur median lima batch untuk network 16/64/256 neuron dengan ring synapses dan delayed event, checkpoint save/restore 64 neuron, prefix/tag retrieval store berisi 256 record, serta reasoning 3 langkah pada jaringan 16 neuron dengan 16 pengalaman. Tidak ada warm-up terpisah pada benchmark system. Hasil bukan peak memory, beban multi-proses, atau jaminan performa deployment.

`benches/language.rs` memuat dan mem-parse NusaX-MT train yang dipin (500
kalimat), membangun vocabulary aktual berisi 2.795 token, lalu mengukur lookup,
encoding, serta membandingkan algoritme reverse lookup linear lama dengan indeks
`WordID` hashbrown. Pada satu run Linux Release, median decoder adalah 1.391,3
ns/ID untuk linear dan 9,6 ns/ID untuk indeks (sekitar 145× lebih cepat pada
sampel tersebut); direct lookup hashbrown 27,5 ns/op, `Vocabulary::encode`
7.900,3 ns/teks, dan `encode_word_ids` 5.333,4 ns/teks. Angka ini pengukuran
lokal berukuran terbatas dan bukan jaminan performa lintas mesin/aplikasi.

`examples/capacity_probe.rs` dan `scripts/run_capacity_probe.sh` menjalankan workload terpisah untuk RSS/peak RSS Linux, build, median step, checkpoint save dan byte size sampai 16,384 neuron/65,536 sinaps. Hasil aktual Sandbox disimpan pada `results/capacity_sandbox_2026-10-19.csv` dan raw NDJSON; hasil tersebut bukan pengukuran perangkat pengguna.

## Pengujian Milestone 11

Regression `tests/milestone11_self_learning.rs` juga menguji fitur turunan/schema, scaler train-only, kandidat k deterministik, migrasi snapshot single-prototype, restart model, SecureMemoryStore, injeksi ke LIF, budget exemplar, calibration refit validation, dan deteksi payload/schema corrupt memakai `data/iris_uci.csv`. Runner release menulis hasil per seed ke `results/class_knowledge_enterprise_2026-10-19.csv`, skor kandidat k ke `results/class_knowledge_validation_2026-10-19.csv`, confusion/per-class metrics ke `results/class_knowledge_per_class_2026-10-19.csv`, dan Brier/MAE test dari kalibrator Platt validation ke `results/class_knowledge_calibration_2026-10-19.csv`. Pengujian belum mencakup drift/OOD, Mahalanobis singular fallback, ECE, dan benchmark lengkap noise/replay budget.

`examples/milestone11_self_learning.rs` menjalankan lima seed dan menghasilkan 35 baris: tiga jalur test bersih, memory pada noise ±0.25, continual old-task setelah replay, self-loop validation, dan persistence setelah restart. Test Iris tidak dimasukkan ke memory sebelum evaluasi test. Scaler dan pemilihan memory current memakai train/validation saja; test hanya dievaluasi di akhir.

Regression test `tests/milestone11_self_learning.rs` memverifikasi split disjoint, label stratified, train-only scaler, k-fold tanpa overlap, dan self-evaluation. Hasil M11 lengkap ada pada `LAPORAN_MILESTONE_11.md`.

### Verifikasi perbaikan keterbatasan

Benchmark final lima seed mengukur accuracy neural tanpa memory `0.886666` versus retrieval eksternal `0.673333`. Validation self-improvement meningkat dari `0.900000` menjadi `0.966667`; acceptance gate membuang kandidat yang menurunkan skor. Prototype neural disimpan ke memory, store dibuka ulang, prototype dipulihkan, lalu dievaluasi kembali. Hasil CSV final ada di `results/milestone11_self_learning_release_2026-10-19.csv`.

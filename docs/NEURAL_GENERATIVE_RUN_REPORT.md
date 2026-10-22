# Laporan engine generatif neural Shella — 2026-10-22

## Audit dan integrasi

Shella telah memiliki `Vocabulary`/`WordID`, `LanguageNetwork`, `SemanticMemory`, `LearningMemory`, checkpoint, serta SDK/session. Tokenisasi `Vocabulary` yang ada bersifat token kata dan bukan tokenizer next-token yang memadai untuk decoder bahasa. Implementasi karena itu menambahkan tokenizer karakter Unicode yang terpisah dan mendokumentasikan bahwa karakter/OOV tidak berarti kata telah dipahami. Decoder memakai `candle-core` dan `candle-nn` versi tepat 0.8.4; `Cargo.toml` menetapkan Rust 1.85 dan edition 2024.

`GenerativeLanguageModel` memuat parameter Transformer aktual: token embedding, positional embedding, masked causal self-attention, residual/normalization, feed-forward, output projection, dan cross-entropy next-character. Training menggunakan autograd Candle, AdamW, batching, clipping gradien, validasi numerik, serta seed. Inference autoregresif menyertakan token keluaran ke konteks berikutnya dan tidak mengubah bobot. Checkpoint menyimpan bobot, arsitektur, tokenizer, provenance, konfigurasi dan report; checksum divalidasi saat dimuat. State AdamW tidak disimpan, sehingga resume training tidak didukung.

SDK kini menyediakan `Session::generate_text(&GenerativeLanguageModel, prompt, config)`. Ini sengaja merupakan delegasi opsional: model generatif **tidak** membaca `SemanticMemory`, `LearningMemory`, `LanguageNetwork`, WordNet, atau output reasoning. Jalur fakta tetap memakai `SemanticMemory` dan abstention yang sudah ada. Output generator berlabel `unverified`; ia tidak boleh dipromosikan sebagai jawaban berbukti. Dengan demikian belum ada mekanisme neural retrieval-grounded generation.

## Dataset dan split

Training memakai file resmi NusaX-Senti Indonesia `train.csv` pada revisi `75b0be5d982348332509f296e10d9bf9cb66d60f`, dengan lisensi data CC BY-SA 4.0 dan atribusi IndoNLP NusaX contributors. Sumber dan lisensi diperiksa melalui provenance loader Shella dan file notice upstream [1] [2]. File berukuran 84,054 byte; SHA-256 `698c4802111085c14fbe35122957e1145338d307bcc295764026a23095a32984`; Git blob SHA-1 `1be82eec8b97232838644566f56e6326fe038937`.

Loader menormalkan NFC/lowercase, memvalidasi teks, lalu mendeduplikasi sebelum pembagian deterministik. Dari 500 record sumber tidak ada exact duplicate atau near-duplicate yang dibuang. Split seed 42 dan proporsi 80/10/10 menghasilkan 400 train, 50 validation, dan 50 test internal. ID disjoint dan disimpan dalam manifest; tokenizer hanya di-fit pada 400 dokumen training. Gradien hanya melihat train internal. Kolom label sentiment bukan target. Official NusaX validation/test, NusaX-MT, IndoSum, dan WordNet tidak menjadi data training. Test internal baru dievaluasi setelah optimizer selesai; hyperparameter tidak dipilih dari test. Perlu diingat bahwa test ini adalah holdout internal yang dibentuk dari upstream train, bukan official NusaX test.

## Run dan metrik

Run aktual memakai model CPU dengan hidden size 32, satu layer, dua attention head, context 64, 18,131 parameter, seed model/training 2026, enam epoch, batch 8, learning rate 0.002, weight decay 0.01, gradient clip 1.0. Setiap epoch menjalankan 144 optimizer step, total 864 step. Waktu training dan evaluasi tercatat 105.50 detik.

Pada validation yang sama, baseline sebelum training menghasilkan loss 4.6024 dan perplexity 99.7192 pada 7,592 target posisi. Setelah training, validation menghasilkan loss 2.6201 dan perplexity 13.7372. Loss/perplexity training epoch keenam adalah 2.5975 / 13.4302. Test internal final, yang tidak dipakai untuk update, menghasilkan loss 2.5768 dan perplexity 13.1549 pada 8,286 target posisi. Setiap dokumen menyertakan EOS sebagai target; BOS bukan label target dan padding diabaikan oleh loss.

Metrik ini menunjukkan model mempelajari statistik karakter dari korpus kecil tersebut. Metrik tidak mengukur ketepatan fakta, kualitas jawaban, atau generalisasi bahasa.

## Generasi dan pemeriksaan kualitatif

Checkpoint dimuat ulang, lalu diuji dengan prompt `Bagaimana cara merawat tanaman di rumah?`. Pemeriksaan literal terhadap semua teks dalam NusaX train tidak menemukan prompt tersebut. Greedy decoding menghasilkan tepat 100 karakter-token dan tidak berhenti oleh EOS:

```text
n san anga  s ma san s san sanganganga sa san san an an an an an a a a a a a a a a a a a a an a a a
```

Keluaran ini berulang dan tidak koheren sebagai jawaban. Pengulangan run dengan prompt dan seed sama memberikan JSON identik, tetapi stabilitas bukan mutu. Contoh yang kurang baik ini dicatat apa adanya; tidak ada klaim kelancaran atau factuality.

## Perubahan dan verifikasi

Modul generatif baru: `src/language/generative/{checkpoint.rs,dataset.rs,generation.rs,mod.rs,model.rs,tokenizer.rs,training.rs}` dan CLI `src/bin/shella_lm.rs`. Integrasi/ekspor: `src/sdk/session.rs`, `src/language/mod.rs`, `src/lib.rs`, serta tes SDK `tests/sdk_integration.rs`. Perubahan dependency: `Cargo.toml`, `Cargo.lock`. Dokumentasi: `docs/API.md`, `docs/BAHASA.md`, `docs/DATASET_BAHASA_INDONESIA.md`, dan laporan ini. Perubahan kompatibilitas lain pada working tree: `src/knowledge/class_knowledge.rs`, `src/language/indonesian/mod.rs`, `src/language/indonesian/wordnet.rs`, dan `src/learning/long_term.rs`.

Validasi final pada Rust 1.85.0: `cargo fmt --check` lulus; `cargo test --all-targets --locked` lulus dengan 231 passed, 0 failed, 2 ignored; `cargo clippy --all-targets --all-features --locked -- -D warnings` lulus. Tes generatif mencakup causal mask, input-target, masking PAD/loss, perubahan bobot nyata lewat gradient update, konfigurasi, tokenizer, decoding, checkpoint, serta pemanggilan SDK.

Checkpoint tersedia sebagai `target/nusax-pretrained-20261022.sllm` (80,522 byte), SHA-256 `c78e16bad5e94be514f1b72a52e732253fe40f538551957c15bf02b19aae259b`. Report JSON run dan JSON generasi disimpan terpisah. Artefak berada di `target/` dan tidak otomatis masuk Git.

## Batasan yang tersisa

Korpus training berisi hanya 500 review pendek dari satu dataset tugas sentimen; model hanya 18 ribu parameter dan tokenizer karakter. Contoh prompt holdout memperlihatkan repetisi dan ketidakkoherenan. Tidak ada pengujian generalisasi bahasa luas atau evaluasi factuality generatif. Integrasi SDK tidak menghubungkan generator ke evidence retrieval; untuk jawaban fakta, gunakan jalur semantik yang ada. Checkpoint dapat dipakai ulang untuk inference, tetapi training resume tidak lengkap tanpa optimizer state.

## References

[1]: https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f "NusaX source revision used by Shella"
[2]: https://github.com/IndoNLP/nusax/blob/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/LICENSE "NusaX dataset license notice"

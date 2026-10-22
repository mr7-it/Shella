# Laporan engine generatif neural Shella — 2026-10-22

## Audit dan integrasi

Shella telah memiliki `Vocabulary`/`WordID`, `LanguageNetwork`, `SemanticMemory`, `LearningMemory`, checkpoint, serta SDK/session. Tokenisasi `Vocabulary` yang ada bersifat token kata dan bukan tokenizer next-token yang memadai untuk decoder bahasa. Implementasi karena itu menambahkan tokenizer karakter Unicode yang terpisah dan mendokumentasikan bahwa karakter/OOV tidak berarti kata telah dipahami. Decoder memakai `candle-core` dan `candle-nn` versi tepat 0.8.4; `Cargo.toml` menetapkan Rust 1.85 dan edition 2024.

`GenerativeLanguageModel` memuat parameter Transformer aktual: token embedding, positional embedding, masked causal self-attention, residual/normalization, feed-forward, output projection, dan cross-entropy next-character. Training menggunakan autograd Candle, AdamW, batching, clipping gradien, validasi numerik, serta seed. Inference autoregresif menyertakan token keluaran ke konteks berikutnya dan tidak mengubah bobot. Checkpoint menyimpan bobot, arsitektur, tokenizer, provenance, konfigurasi dan report; checksum divalidasi saat dimuat. State AdamW tidak disimpan, sehingga resume training tidak didukung.

Pada run awal ini, SDK menyediakan `Session::generate_text(&GenerativeLanguageModel, prompt, config)` sebagai delegasi opsional tanpa retrieval. Follow-up di bagian akhir laporan menambahkan method memory-conditioned; kedua jalur neural tetap berlabel `unverified` dan bukan pengganti jawaban berbukti dari `SemanticMemory`.

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

Pada run awal, korpus training berisi hanya 500 review pendek dari satu dataset tugas sentimen; model sekitar 18 ribu parameter dengan tokenizer karakter. Contoh holdout memperlihatkan repetisi dan ketidakkoherenan. Tidak ada klaim generalisasi bahasa luas atau evaluasi factuality generatif. Batas integrasi terbaru dan uji conditioning dicatat pada follow-up di bawah. Checkpoint dapat dipakai ulang untuk inference, tetapi training resume tidak lengkap tanpa optimizer state.

## References

[1]: https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f "NusaX source revision used by Shella"
[2]: https://github.com/IndoNLP/nusax/blob/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/LICENSE "NusaX dataset license notice"


## Follow-up audit dan integrasi SemanticMemory — 2026-10-22

Bagian sebelumnya menggambarkan keadaan awal sebelum permintaan integrasi. Audit read-only saat itu menemukan bahwa `Session::generate_text` hanya meneruskan prompt langsung ke `GenerativeLanguageModel::generate`; model memuat tokenizer karakter dari checkpoint, menambahkan output token secara autoregresif, dan tidak membaca `SemanticMemory`. Checkpoint `SHLMLM01` sudah memvalidasi checksum/provenance dan tokenizer, sehingga dipertahankan tanpa perubahan format. `SemanticMemory::retrieve` memberi skor ranking heuristik atas pernyataan konteks berbasis WordID/relasi, tetapi tidak membuktikan entailment atau factuality keluaran generator. Dataset neural hanya 500 review NusaX-Senti pendek; output baseline pada prompt tanaman menunjukkan repetisi berat.

Implementasi menambahkan API SDK additive `Session::generate_text_with_semantic_memory`. Ia menggunakan SemanticMemory yang sama pada `Engine`, memanggil retrieval berdasarkan prompt dan `context_id`, membatasi retrieval sampai 32, membuang duplikat teks, dan hanya menyertakan pernyataan lengkap yang muat dalam context window. Prompt conditioning berlabel `catatan/tanya/jawab` masuk ke pemanggilan `model.generate`, jadi decoding tetap autoregresif. Recall yang ditemukan dan recall yang benar-benar dimasukkan dikembalikan terpisah. Tidak ada evidence yang relevan atau tidak ada evidence yang muat menghasilkan abstention (`generated: None`); tidak ada fallback diam-diam ke generasi bebas. Output yang berhasil tetap `unverified`: recall score bukan confidence, dan tidak dilakukan verifikasi klaim terhadap evidence. Karena itu jalur ini belum menjamin jawaban faktual.

### Repetisi dan holdout generasi

Perubahan decoder menghindari kandidat karakter yang mengulang 4-gram terdahulu jika alternatif tersedia; bila seluruh kandidat akan berulang, decoder memakai kembali kandidat awal agar tidak dead-end. Checkpoint dan `GenerationConfig` tidak berubah. `GeneratedText::repetition_metrics()` dan CLI `shella-lm metrics --input-json` memberikan distinct-character ratio, repeated n-gram rates, adjacent repeats, EOS, dan prompt truncation.

Pada **checkpoint, prompt, seed, jumlah token, dan greedy strategy identik**, metrik untuk prompt `Bagaimana cara merawat tanaman di rumah?` berubah sebagai berikut:

| Metrik (100 token) | Sebelum filter 4-gram | Sesudah filter 4-gram |
|---|---:|---:|
| Rasio karakter unik | 0.0600 | 0.1400 |
| Repeated bigram rate | 0.8788 | 0.6364 |
| Repeated trigram rate | 0.7857 | 0.3367 |
| Repeated 4-gram rate | 0.6804 | 0.0000 |
| Adjacent repeat rate | 0.0101 | 0.0202 |
| EOS / prompt truncated | false / false | false / false |

Repeated 4-gram nol adalah akibat langsung batas decoding, bukan peningkatan terukur pada koherensi. Adjacent repeat sedikit naik dan keluaran tetap bukan kalimat Indonesia yang koheren. Setelah perubahan, prompt tersebut menghasilkan `n san anga  s ma sak s bangi sa si siang ba s saman sin s dan mangeran bakan da st sasangu saan t s `. Dua prompt tambahan yang tidak terdapat dalam semua split NusaX internal—`Apa saja manfaat berjalan kaki setiap pagi?` dan `Mengapa kita harus mematikan lampu yang tidak dipakai?`—juga menghasilkan fragmen karakter yang tidak koheren; keluaran dan diagnostiknya disimpan terpisah di `target/nusax-holdout-walk-20261022.json` dan `target/nusax-holdout-energy-20261022.json`. Tes Rust `tests/generative_evaluation.rs` memeriksa bahwa ketiga prompt tidak muncul sebagai substring pada train, validation, maupun test internal, dan memvalidasi ID split disjoint.

### Eksperimen context=128 dan perplexity

Satu run tambahan melatih ulang model baru dengan NusaX train internal yang sama (seed split 42; 400/50/50 dokumen; 0 exact dan 0 near duplicate dibuang), seed 2026, 6 epoch, batch 8, learning rate 0.002, weight decay 0.01, gradient clip 1.0, satu layer, hidden size 32, dua head, context 128, 20.179 parameter. Baseline validation sebelum update: loss/perplexity **4.4517 / 85.7685**. Sesudah training: **2.6306 / 13.8827**. Test internal yang dievaluasi sesudah training: **2.6108 / 13.6099** pada 8.286 target token. Checkpoint baru berukuran sekitar 87 KiB dengan SHA-256 `691e63bccadffae410dc0fe8ed423e5f578ba63044288753ed00f9fc2bf85843`; report lengkap ada di `target/nusax-context128-pretraining-20261022-r1.json`.

Eksperimen ini tidak menunjukkan perbaikan perplexity dibanding run context=64 terdahulu (final validation 13.7372, test 13.1549); arsitektur/jendela berbeda dan hanya satu seed/run, jadi tidak cukup untuk klaim kausal. Checkpoint context=64 terdahulu dipertahankan sebagai pembanding; checkpoint context=128 juga tidak dihapus.

### Conditioning dan pemulihan proses

Tes integrasi menggunakan checkpoint context=64 terlatih yang sebenarnya, menyimpan context SemanticMemory `kucing makan nasi`, memanggil query `kucing makan apa?`, lalu membuka ulang store dan checkpoint pada proses test child yang terpisah. Retrieval aktual menghasilkan satu recall dengan relevance **1.0** (alasan: konsep WordID sama dan konteks sama), dan satu statement itu benar-benar masuk ke prompt tanpa truncation. Hasil generasi identik setelah process restart; tes lulus. Namun continuation aktual `a dan s ` (8 token) bukan jawaban yang relevan. Ini membuktikan jalur retrieval/conditioning/persistence bekerja, **bukan** bahwa generator mengikuti atau memverifikasi memory. Respons tetap ditandai unverified. Tes tanpa recall relevan juga membuktikan jalur menahan generasi.

### Verifikasi

Validasi final pada Rust 1.85.0: `cargo fmt --check` lulus; `cargo +1.85.0 clippy --all-targets --all-features --locked -- -D warnings` lulus tanpa warning; `cargo +1.85.0 test --all-targets --locked` lulus dengan **237 passed, 0 failed, 2 ignored** dalam 36 harness. Tes khusus juga lulus: 5 unit test decoder/metrik; 1 tes prompt holdout no-leakage; tes abstention untuk context tanpa recall dan evidence yang terlalu besar; serta uji restart proses terpisah dengan checkpoint terlatih aktual. `git diff --check` bersih.

Batas utama tetap: tokenizer per-karakter, dataset kecil dan khusus review sentimen, keluaran holdout yang belum koheren, model tidak dilatih sebagai instruction-following, conditioning teks bukan grounding faktual, skor relevance bukan confidence, serta state optimizer tidak masuk checkpoint. Jangan gunakan jalur neural ini sebagai jawaban fakta terverifikasi; untuk respons berbukti gunakan `SemanticMemory::respond_in_context`.


## Follow-up: pipeline LIF–Transformer dan korpus legal Indonesia — 2026-10-22

### Alur jawaban terverifikasi yang diimplementasikan

SDK kini menyediakan `Session::answer_with_neural_review` di `src/sdk/verified_answer.rs`. API ini memanggil reasoning LIF saat menerima pertanyaan, memakai `SemanticMemory` yang terpasang pada `Session` untuk mengambil bukti, lalu hanya meminta Transformer membuat draf jika bukti sumber/konteks tersedia, tidak ambigu, memiliki ID evidence dan statement pendukung, serta LIF memilih aksi `draft`. Kapasitas prompt diperiksa agar pertanyaan atau bukti tidak terpotong.

Draf Transformer dikirim kembali ke `ReasoningEngine` untuk jejak keputusan LIF, lalu diperiksa oleh verifier konservatif: kecocokan istilah konten dengan pertanyaan, tumpang tindih istilah evidence/acuan jawaban, istilah yang tidak didukung, dan konsistensi polaritas ya/tidak. Jika gagal atau LIF meminta `revise`, sistem memberi paling banyak satu kesempatan revisi. Jawaban hanya dilepas jika LIF memilih `approve` dan pemeriksaan konservatif lulus; selain itu sistem abstain dengan alasan yang tercatat. Empat indeks aksi (`draft`, `approve`, `revise`, `abstain`) harus berbeda, dipetakan ke neuron aksi, serta sudah dipelajari/diatur pada `ReasoningEngine`.

Ini adalah jaringan neuron digital LIF, bukan simulasi biologis lengkap. Skor LIF merupakan heuristik dan pemeriksaan teks **bukan entailment umum, verifikasi fakta dunia luar, ataupun jaminan kebenaran**. `SemanticMemory` tetap harus berisi bukti yang relevan; korpus pretraining tidak otomatis menjadi fakta terindeks. Tes yang tersedia mencakup unit verifier untuk klaim/polaritas, integrasi abstention sebelum Transformer saat evidence tidak ada, dan tes SDK terhadap korpus aktual. Jalur approve/revise tidak boleh dianggap telah membuktikan kualitas faktual model.

### Korpus sumber nyata, pemrosesan, dan ukuran

Subkorpus yang dibangun berisi teks OCR lima undang-undang resmi Indonesia dari JDIH BPK pada lima topik: hak cipta, keterbukaan informasi publik, pengelolaan sampah, pendidikan nasional, dan pendidikan tinggi. Pemakaian dibatasi pada teks peraturan menurut UU No. 28 Tahun 2014 Pasal 42(b)—bukan lisensi menyeluruh atas semua isi situs pemerintah. URL sumber, judul/status, PDF asli, SHA-256, dan dasar hak dicatat pada `data/indonesian_legal_corpus/sources.json`; audit sumber lain dan alasan eksklusinya ada di `docs/AUDIT_LISENSI_KORPUS_INDONESIA.md`.

Pipeline nyata di `tools/build_indonesian_legal_corpus.py` memakai Tesseract Indonesia/Inggris PSM 4, normalisasi NFC/lowercase, pembersihan artefak layout/fragmen scan/pengulangan, deduplikasi exact dan near-duplicate, lalu split berdasarkan dokumen dengan seed 42 sebelum training. Snapshot final memiliki **392.350 karakter-token** (tokenizer karakter Unicode Shella; bukan BPE), yaitu train **180.599** token/3 dokumen, validasi **111.153**/1 dokumen, test **100.598**/1 dokumen, serta 52.603 kata whitespace. Tidak ada duplicate yang terdeteksi. Train-only sekitar **2,94×** baseline NusaX internal 61.339 karakter-token/400 dokumen dengan algoritme split dan tokenisasi yang sama; NusaX tidak dicampur ke korpus hukum ini. OCR tetap dapat salah dan belum diperiksa manual pasal demi pasal.

Loader `CorpusSplit::load_prepared_json` memeriksa format, normalisasi, ID disjoint, jumlah karakter-token, identitas split, path relatif, serta ukuran/checksum SHA-256 dan Git blob atas inventaris sumber sebelum dataset bisa dipakai optimizer. Tes `tests/prepared_legal_corpus.rs` memuat JSON dan inventaris nyata, memeriksa 3/1/1 split, lalu membangun objek model Shella dari korpus tersebut; tidak menggunakan mock atau data placeholder. Perintah training tersedia di panduan korpus, tetapi tidak ada checkpoint pretraining baru yang diklaim selesai dari dataset kecil ini.

### Validasi dan batas skala

Validasi Rust penuh terakhir: `cargo test --all-targets --locked` menghasilkan **241 passed, 0 failed, 2 ignored** pada 37 harness; tes loader korpus aktual dijalankan kembali terhadap snapshot final dan lulus. `cargo fmt --all --check`, Clippy dengan `-D warnings`, serta `git diff --check` juga lulus.

Permintaan **model 2 miliar parameter belum diwujudkan**. Implementasi yang ada dibatasi hingga 4 juta parameter, CPU-only, tokenizer karakter, dan context maksimal 256; loader JSON membaca korpus penuh ke memori dengan batas 512 MiB. Snapshot 392.350 karakter-token yang legal dan sempit ini juga jauh dari skala korpus umum pra-pelatihan. Model 2B memerlukan perubahan arsitektur/backend dan jalur data sharded/streaming serta korpus Indonesia yang jauh lebih besar dengan hak atas teks yang benar-benar diperiksa. Perubahan ini tidak menaikkan batas secara palsu atau menghasilkan checkpoint yang tidak dapat digunakan.

### Follow-up: ekspansi korpus prosa koheren Indonesia — 2026-10-22

Pemindaian dump resmi Wikipedia Indonesia snapshot 2026-07-01 selesai. Checksum SHA-1 arsip yang diumumkan Wikimedia cocok dengan unduhan aktual; dump dibaca streaming, disaring ke artikel berbahasa Indonesia yang memenuhi syarat prosa, dipilih deterministik hingga 20.000 artikel, lalu digabung dengan lima teks undang-undang yang sudah dikurasi. Korpus akhir berisi **20.005 dokumen** dan **71.297.343 karakter-token** (CharacterTokenizer Shella; bukan BPE), sekitar **9.865.107 kata whitespace**. Split dokumen menghasilkan 69.927.479 karakter-token train/19.604 dokumen, 708.033 validasi/200, dan 661.831 test/201. Deduplikasi exact serta near-duplicate tidak menemukan penghapusan pada snapshot ini. Statistik, hash dan provenance berada di `data/indonesian_coherent_corpus/` dan panduan di `docs/KORPUS_BAHASA_INDONESIA_KOHEREN.md`.

Lisensi Wikipedia bersifat per artikel dan campuran (umumnya CC BY-SA 4.0 dan/atau GFDL, dengan kemungkinan atribusi/lisensi impor tambahan). Record mencatat judul, page ID/revision ID, URL artikel/riwayat, lisensi, serta modifikasi; materi dengan penanda pengecualian hak eksplisit dibuang. Namun ini bukan review hukum manual seluruh kutipan atau rantai atribusi, sehingga korpus belum diklaim cleared untuk distribusi komersial dan efek ShareAlike/GFDL terhadap bobot model belum disimpulkan.

Tes integrasi eksplisit pada korpus final lulus: loader memvalidasi inventory, checksum/provenance, jumlah karakter, dan split untuk 20.005 dokumen lalu berhasil membentuk model Shella (`1 passed`). Pemeriksaan Rust penuh, format, Clippy tanpa warning, dan semua target telah lulus sebelum test khusus ini; test khusus kemudian juga lulus pada JSON final. **Optimizer/pretraining atas korpus baru belum dijalankan.** Ekspansi data tidak mengubah batas model: Transformer saat ini tetap maksimal 4 juta parameter dan target 2 miliar parameter belum direalisasikan.


## Follow-up: head LIF terdiferensiasi dan feedback terverifikasi — 2026-10-23

Model decoder kini memuat recurrent LIF readout 16 neuron di atas hidden states Transformer. Forward memakai spike biner dan backward memakai fast-sigmoid surrogate. `train_on_verified_feedback` mengoptimalkan next-character loss serta action loss (`Answer`, `Revise`, `Abstain`) pada graph Candle yang sama. `Session::learn_from_verified_feedback` menerima label dari `HumanConfirmed` atau `TrustedEvaluator`, mencatat fitur pengalaman ke `LearningMemory` bila dipasang, dan menyimpan checkpoint model/checksum. Feedback harus diberikan eksplisit; interaksi biasa, jawaban model yang belum ditinjau, dan auto-labeling tidak mengubah model.

Head hybrid terhubung sebagai pemeriksa draf di `answer_with_neural_review`, setelah kandidat dihasilkan, agar distribusi input pada review cocok dengan contoh feedback (prompt pertanyaan+bukti dan kandidat). Aksi `Abstain` menghentikan pelepasan/revisi draf. Gerbang ini tambahan atas `ReasoningEngine` LIF, SemanticMemory, dan pemeriksa leksikal/polaritas. Pemeriksa tersebut bukan entailment umum, verifikasi fakta eksternal, ataupun jaminan kebenaran. Detail kontrak API ada di `docs/PEMBELAJARAN_HIBRIDA_ONLINE.md`.

Audit gradien menemukan batas penting di backend: fused `LayerNorm::forward` Candle 0.8.4 memakai `apply_op3_no_bwd`. Sebelum diperbaiki, next-character loss hanya mengirim gradien ke output projection dan tidak mencapai blok decoder; karena itu klaim bahwa Transformer dapat belajar bersama belum didukung. Decoder kini menggunakan ekspresi `layer_norm_slow` yang tetap terdiferensiasi. Tes regresi membuktikan language loss mencapai embedding token dan bobot attention QKV, sementara action loss mencapai QKV Transformer serta proyeksi input LIF.

Verifikasi sesudah perbaikan: `cargo fmt --all --check`, `cargo test --all-targets --locked` (**248 passed, 0 failed, 3 ignored** pada 37 harness), `cargo clippy --all-targets --all-features --locked -- -D warnings`, dan `git diff --check` lulus. Tes SDK baru juga menjalankan training feedback satu langkah, memeriksa record `LearningMemory`, lalu memuat kembali checkpoint hybrid.

**Pretraining penuh pada korpus 71.297.343 karakter-token belum dijalankan**; jumlah itu karakter Unicode menurut tokenizer Shella, bukan token BPE. Belum ada checkpoint atau metrik bahasa yang diklaim dari snapshot koheren tersebut. Model tetap CPU-only, maksimal 4 juta parameter, tokenizer karakter, context maksimum 256; target 2 miliar parameter dan tokenizer subword belum diwujudkan. Checkpoint training tidak menyimpan state AdamW sehingga momentum tidak berlanjut saat resume.

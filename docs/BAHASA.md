# Pipeline teks

Pipeline bahasa hanya preprocessing dan representasi sederhana. `normalize_text` melakukan Unicode NFC, lowercase, dan merapatkan whitespace. `tokenize` menjadikan run karakter Unicode alfanumerik sebagai token dan tanda baca sebagai separator. `Vocabulary::fit(corpus, min_frequency, max_tokens)` membangun vocabulary dari korpus yang diberikan; urutan token adalah frekuensi menurun lalu leksikografis, dengan `<unk>` pada ID 0. `Vocabulary::encode` memetakan kata di luar kamus ke `<unk>`.

`TextEncoder::encode(text, neuron_count, current_per_token_na)` mengembalikan token, ID positional lama (`token_ids`), `WordID` stabil (`word_ids`), dan vektor arus. Setiap occurrence menambah arus positif pada neuron `token_id % neuron_count`. Ini bag-of-token deterministik dengan collision; tidak merepresentasikan tata bahasa, makna, konteks, embedding, subword, klasifikasi terlatih, atau generasi teks. Vocabulary harus di-fit pada data latih oleh pemanggil sebelum dipakai pada data evaluasi; jangan fit ulang dari data uji bila ingin evaluasi tanpa kebocoran.

`Vocabulary::lookup_word` menormalisasi input satu kata untuk lookup, sedangkan `id`/`word_id` menyediakan jalur lookup cepat atas token yang sudah kanonis. `Vocabulary::encode_word_ids` dan `decode_word_ids` mendukung round-trip melalui identitas kata stabil. Decoder mengembalikan ejaan kanonis yang tersimpan; tokenisasi mengubah input menjadi NFC/lowercase, memisahkan tanda baca, dan OOV memakai `WordID(0)`/`<unk>`. ID lain yang tidak ditemukan adalah error. `Vocabulary::save`/`load` memakai snapshot JSON tervalidasi dan atomic replace; ID lama dipertahankan saat kosakata diperluas setelah pemulihan.


`TextEncoder::encode` menolak input yang setelah normalisasi/tokenisasi tidak menghasilkan token (termasuk teks kosong, whitespace-only, dan punctuation-only). `tokenize` tetap dapat dipakai sebagai tokenizer umum dan mengembalikan daftar kosong untuk input seperti itu. Token OOV tetap dipetakan ke ID `<unk>` 0; encoder belum menyediakan rasio OOV atau keputusan unknown.

## Library kosakata Indonesia

`IndonesianVocabulary` tetap merupakan library yang dapat diperluas. Dataset
resmi yang terverifikasi sekarang disimpan terpisah dari source code di
[`data/language/`](../data/language/). Registry mem-pin stopwords-id (MIT),
Indonesian-PUD (CC BY-SA 3.0), dataset NusaX (CC BY-SA 4.0), dan IndoSum
(CC BY-SA 4.0); versi, ukuran, atribusi, scope, dan keterbatasan pemakaian
dicatat dalam [inventaris dataset](./DATASET_BAHASA_INDONESIA.md). Loader
memeriksa ukuran, Git blob SHA-1, dan SHA-256 yang dipin sebelum parsing. Semua
63 file yang dibundel juga terdaftar dalam manifest integritas versi 1
[`data/language-integrity.json`](../data/language-integrity.json), yang mencatat
ukuran, SHA-256, format, dan encoding. `DatasetIntegrityManifest::bundled()`
dan `verify_directory("data/language")` memvalidasi file hilang, file ekstra,
checksum, UTF-8, dan struktur format dasar. CLI Rust dapat menjalankan
`cargo run --bin dataset-integrity -- verify data/language data/language-integrity.json`.
Manifest memberi deteksi perubahan terhadap nilai yang dikomit, bukan tanda tangan
kriptografis; perubahan manifest sendiri tetap memerlukan review.
Manifest tersebut juga menyimpan ukuran dan SHA-256 dua file Hunspell yang
dimuat dari luar bundle; loader mengecek Git blob SHA-1 dan SHA-256 keduanya.

PUD dapat diimpor sebagai anotasi CoNLL-U ke lexicon/checkpoint. Loader NusaX
membaca alignment terjemahan, data sentimen, dan lexicon multibahasa; tokenizer
dan evaluator memakai fungsi bahasa Shella yang ada. IndoSum dibaca per fold
JSONL dan hanya menyediakan oracle berlabel serta pengukuran token-level
ROUGE-1. Tidak satu pun dataset ini melatih model dengan sendirinya.

UD Indonesian-GSD dan CSUI belum diaktifkan karena hak atas teks sumber belum
terverifikasi terpisah dari lisensi anotasi; IndoNLI tidak memiliki lisensi
dataset yang cukup jelas. WordNet Bahasa r24 tersedia pada revisi yang dipin dan
lisensi MIT yang diverifikasi. Dataset itu hanya diimpor dengan label Y/O;
relasi dihasilkan dari keanggotaan synset bersama untuk lema yang tidak ambigu.
Sumber yang tetap tertahan tidak diberi provenance resmi dan tidak digunakan
untuk mengklaim kemampuan inferensi atau pemahaman bahasa.

Entry yang ditambahkan melalui `upsert`/`import_json` tetap wajib membawa nama
sumber, URI entri, identitas lisensi, URI lisensi, dan ID asal entri. Validasi
field generik tidak membuktikan keaslian sumber atau izin; provenance dataset
allowlisted berasal dari `DatasetLoader`.

API loader:

- `DatasetLoader::load_ud_conllu` memvalidasi dan mengimpor struktur CoNLL-U,
  termasuk PUD yang diizinkan; ID GSD/CSUI tetap ditolak registry.
- `DatasetLoader::load_stopwords_json` memuat JSON upstream yang terverifikasi;
  stopword/frasa dapat dipakai oleh `LanguageNetwork` sebelum proyeksi arus.
- `DatasetLoader::load_nusax_mt_csv`, `load_nusax_sentiment_csv`, dan
  `load_nusax_lexicon_csv` membaca CSV berlisensi yang hash-pinned. Evaluasi
  sentimen menerima prediksi dari pemanggil dan menolak split train.
- `DatasetLoader::load_indosum_archive` memverifikasi arsip terkompresi, lalu
  membaca hanya satu split JSONL. `IndoSumArticle::evaluate_summary` menghitung
  ROUGE-1 atas kandidat yang diberikan, bukan keluaran model Shella.
- `DatasetLoader::load_hunspell_files` memuat pasangan Indonesia `.dic`/`.aff`
  yang terverifikasi. Root dan aturan satu-afiks tersimpan dalam lexicon; analisis
  tidak mengimplementasikan seluruh semantik Hunspell (misalnya kombinasi aturan
  lintas-afiks/continuation class), dan kata majemuk bertanda hubung dihitung
  sebagai dilewati.

Gunakan `IndonesianVocabulary::new(LexiconMetadata::indonesian(name, version))`
kemudian `upsert(LexiconEntryInput)` atau `import_json` untuk penambahan tervalidasi.
JSON persisten memiliki `schema_version`, metadata dataset, entri, ID kata, sumber,
lisensi, dan metadata linguistik opsional. Upsert mempertahankan `WordID`; batch
import bersifat atomik. Variasi ejaan masuk indeks alias NFC/case-insensitive,
sementara kelas kata, lema, imbuhan, definisi, dan rujukan definisi tetap opsional.
Definisi harus menunjuk `source_entry_id` yang tercatat pada entri.

`to_json`/`from_json` menyerialisasi dan memvalidasi library. Untuk persistensi
filesystem, `Checkpoint::save_indonesian_vocabulary` dan
`restore_indonesian_vocabulary` memakai checkpoint JSON checksummed dan penulisan
atomik. `segment_unlisted` hanya mengembalikan karakter Unicode ternormalisasi;
fitur ini tidak menyimpulkan lemma, morfologi, ataupun makna. Schema JSON
[`indonesian-vocabulary-v1.schema.json`](../schemas/indonesian-vocabulary-v1.schema.json)
mendokumentasikan bentuk import dan data tersimpan. `to_language_network` membawa
vocabulary dan ID yang sama ke `LanguageNetwork`, tempat aplikasi dapat mengikat
`WordID` ke satu atau lebih neuron. Neural binding tetap tanggung jawab aplikasi
dan tidak otomatis belajar dari definisi.

## Semantic memory dan analisis berbukti

`SemanticMemory::open(path, lexicon)` membungkus `MemoryStore` yang sudah ada dan
menyimpan lexicon, relasi, istilah multi-kata, konteks, serta status penghapusan
dalam satu record JSON atomik. File tetap memakai single-writer lock, checksum,
dan write-temp/rename dari `MemoryStore`; `WordID` yang ada tidak dihitung ulang.
`learn_entry` memakai `IndonesianVocabulary::upsert`, jadi kata baru memperoleh ID
append-only. Penghapusan konsep menonaktifkan ID tanpa mendaur ulangnya serta
membersihkan edge, phrase, dan rujukan konteks terkait.

`SemanticToken` menghubungkan token hasil `normalize_text`/`tokenize` ke `WordID`,
definisi bila tersedia, status dikenal, dan span byte pada teks ternormalisasi.
`add_phrase` menerima phrase, `PhraseKind`, makna yang diberikan caller, dan
`SemanticEvidence`; semua kata komponennya harus ada pada lexicon. Longest-match
memilih phrase terdaftar, tidak menebak nama entitas baru.

`ConceptRelation` menghubungkan dua `WordID` dengan jenis sinonim, antonim,
jenis-dari, bagian-dari, sebab-akibat, atau terkait. Setiap edge menyimpan sumber,
ID entri asal, status `SourceAssertion`/`Fact`/`Hypothesis`, timestamp opsional,
dan confidence opsional `0..=1`. API memvalidasi field dan ID, tetapi tidak
mengautentikasi sumber caller; jangan menandai input sebagai fakta tanpa verifikasi
aplikasi. Relasi leksikal yang tersedia pada lexicon diimpor dengan provenance
asal dan tanpa confidence rekaan. WordNet Bahasa dapat diimpor melalui loader yang
sama: hanya anggota Indonesia bermutu Y/O satu-token, dengan relasi dalam synset
yang tidak ambigu dan bukti sumber. File membership tidak menyediakan graf relasi
antar-synset; edge yang diimpor hanya menyatakan keanggotaan pada synset yang sama.

`analyze` memakai POS/lemma yang terdaftar untuk memilih verba pertama sebagai
predikat, token sebelum predikat sebagai subjek/pelaku, token sesudahnya sebagai
objek, dan `kepada`/`untuk` sebagai petunjuk penerima. Negasi terbatas pada
`tidak`, `bukan`, `belum`, `jangan`, `tak`, dan `nggak` sebelum predikat. Role
berlabel `RuleHeuristic`, bukan anotasi sintaksis terverifikasi. `bisa` dan
pronomina konteks hanya ditandai ambigu bila ada lebih dari satu lemma/kandidat;
resolusi pronomina satu langkah memilih subjek noun pada kalimat sebelumnya dan
tetap diberi label heuristik. Ini bukan parser dependency atau coreference umum.

`analyze_in_context` menyimpan paling banyak 128 kalimat per ID konteks.
`retrieve` memberi skor berbasis WordID yang sama atau satu edge relasi langsung
yang memiliki confidence tercatat atau status non-hipotesis; `SourceAssertion`
tanpa confidence dapat membantu ranking, tetapi skornya tetap heuristik, bukan
confidence probabilistik. Kecocokan ID atau relasi wajib ada sebelum bonus konteks
dapat berlaku. `infer` menelusuri edge `is-a`/`bagian-dari`
secara transitif dan sinonim dua arah; trace mengembalikan path, ID bukti, status,
produk confidence bila semua edge memilikinya, dan indikator konflik sinonim /
antonim. Ia tidak menyimpulkan sebab-akibat yang belum tercatat.

`resonate` dapat dipanggil lewat `Session::resonate_semantics` dengan
`LanguageNetwork` yang vocabulary dan `WordID`-nya cocok dengan lexicon. Token
yang belum mempunyai binding neuron dilewati dan dilaporkan. Sinonim hanya
menambah arus LIF bila confidence tercatat dan statusnya bukan `Hypothesis`; lalu
jaringan LIF/STDP aktual menjalankan satu langkah. `ReasoningEngine::with_semantic_memory`
memasang store yang sama untuk `reason_text`; `learn_from_semantic_feedback`
menyimpan analisis konteks, melaporkan gap, membentuk currents semantic, menulis
feedback ke `LearningMemory`, dan memperbarui readout LIF/STDP aktual. Experience
tetap disimpan sebagai fitur currents/neural, bukan record token atau teks mentah.

`SemanticMemory::respond_in_context(context_id, text, retrieval_limit)` menjalankan
analisis token/role, resolusi satu-langkah pada konteks, retrieval, pencocokan frame,
policy evidence/kontradiksi, lalu realisasi jawaban. `SemanticResponse` memuat intent,
kategori evidence, alasan, gap, kutipan konteks, ID bukti, dan provenance/lisensi.
Definisi memerlukan `definition_source_entry_id` yang cocok dengan source record;
relasi memakai trace `infer` dan tidak dipromosikan ke jawaban fakta bila trace masih
`Hypothesis` atau confidence eksplisitnya nol. Sinonim untuk frame/retrieval harus
memiliki status non-hipotesis. Jawaban ya/tidak membandingkan subjek, predikat, objek,
polaritas negasi, dan seluruh pernyataan deklaratif pada konteks yang sama; hanya
overlap kata atau pertukaran subjek/objek tidak cukup. Relasi sinonim WordNet r24
tanpa confidence dapat menjadi bukti sumber untuk kecocokan leksikal, tetapi tidak
menciptakan nilai confidence.

Pertanyaan `Siapa yang ...?` dapat dijawab bila predikat (dan objek bila disebut)
cocok dengan pernyataan positif pada konteks yang sama; nama subjek dan supporting
statement diambil dari konteks. Bentuk WH lain, pertanyaan sebab, sintaksis inversi,
dan banyak paraphrase belum diimplementasikan dan akan abstain. Pertanyaan tidak
menjadi evidence untuk dirinya sendiri. Konteks bukan verifikasi fakta dunia luar.
`confidence_band` adalah kelas evidence, bukan skor probabilistik atau kalibrasi.
Perakitan kalimat tetap deterministik, bukan LLM atau generator bebas.

Loader WordNet r24 memetakan POS suffix synset ke UPOS hanya jika lema menunjuk tepat
satu synset dan tidak memiliki anotasi POS yang sudah ada; ini memberi `is_noun` /
`is_verb` dasar tanpa menebak untuk lema polisemik. Impor revision tes ke lexicon
kosong memberi POS pada **4.589** entry. Definisi hanya dipasang ketika
lema juga unik dan tidak ada konflik; gloss membawa `source_entry_id` serta provenance
sumber/lisensi. Pada snapshot tes, 1.345 entry menerima gloss yang lolos
aturan tersebut dari 12.668 baris definisi yang diparse. Contoh holdout “Apa arti
heterotrof?” dijawab dari gloss upstream. Keberadaan gloss yang diimpor bukan bukti
neural learning.

## Neural autoregressive text generation

`GenerativeLanguageModel` adalah decoder-only Transformer causal berbasis Candle
0.8.4 yang menghitung next-character cross-entropy dan belajar dengan AdamW serta
autograd di CPU. Tokenizer Unicode karakter dibangun hanya dari teks split train;
vocabulary ini khusus model neural dan tidak mengganti `Vocabulary`/`WordID`
leksikal Shella. Model dan checkpoint menyimpan provenance serta manifest split.

`load_nusax_training_corpus` menerima file resmi NusaX-Senti Bahasa Indonesia
split train yang hash-pinned dan CC BY-SA 4.0. Loader menormalisasi NFC/lowercase,
menghapus dokumen duplikat persis dan near-duplicate konservatif sebelum
deterministic split 80/10/10 (seed default 42). Gradien hanya melihat train;
validation dipakai mengukur loss tiap epoch dan membandingkan baseline dengan model
akhir. Test dibaca hanya setelah optimizer selesai, satu kali untuk metrik final.
Label sentimen bukan target, NusaX-MT tidak ditambahkan, dan official NusaX valid/test
tidak dibaca. Hyperparameter tidak dipilih dari test. CLI train menyimpan checkpoint
checksummed dengan manifest, provenance, serta metrik; model AdamW optimizer state
tidak disimpan sehingga training tidak dapat di-resume tepat di tengah epoch.

Satu run aktual pada 2026-10-22 memakai commit NusaX
`75b0be5d982348332509f296e10d9bf9cb66d60f`, SHA-256 source train
`698c4802111085c14fbe35122957e1145338d307bcc295764026a23095a32984`, seed
split 42 dan training 2026. Dari 500 record sumber, 0 exact dan 0 near-duplicate
dibuang; split menjadi 400/50/50 dokumen (train/validation/test), dengan 61,259 /
7,592 / 8,286 karakter-token yang dievaluasi termasuk token khusus. Model memiliki
18,131 parameter, satu layer, hidden size 32, dua head, context 64, enam epoch,
144 optimizer step per epoch. Loss dihitung atas posisi target karakter berikutnya;
EOS termasuk satu target per dokumen dan BOS bukan label target. Baseline validation
loss/perplexity **4.6024 / 99.72**;
sesudah training **2.6201 / 13.74**. Training loss/perplexity epoch terakhir
**2.5975 / 13.43**. Test internal yang baru disentuh sesudah training memberi
loss/perplexity **2.5768 / 13.15** pada 8,286 token. Checkpoint aktual berukuran
80,522 byte dan fingerprint SHA-256
`c78e16bad5e94be514f1b72a52e732253fe40f538551957c15bf02b19aae259b`.

Generasi greedy aktual dari checkpoint dengan prompt literal holdout
`Bagaimana cara merawat tanaman di rumah?` (string tersebut tidak ditemukan di
NusaX train) menghasilkan kelanjutan 100 karakter-token berikut:

```text
n san anga  s ma san s san sanganganga sa san san an an an an an a a a a a a a a a a a a a an a a a
```

Pengulangan pola ini bukan jawaban yang koheren. Ulangan inferensi dengan seed
sama menghasilkan output byte-for-byte identik; determinisme tidak berarti mutu
bahasa memadai. Ini contoh yang dilaporkan apa adanya, bukan bukti kemampuan
menjawab pertanyaan atau memanfaatkan evidence semantic.

Model bahasa karakter kecil ini menghasilkan kelanjutan teks statistik, bukan
fakta terverifikasi. Jalur lama `Session::generate_text` tidak membaca
`SemanticMemory`, `LearningMemory`, `LanguageNetwork`, atau output reasoning;
pemuatan WordNet tidak menjadi bobotnya. API tambahan yang didokumentasikan di
bawah dapat menyisipkan recall `SemanticMemory` sebagai teks conditioning, tetapi
tidak membuktikan klaim keluaran dan tetap berlabel unverified. `GeneratedText::factual_grounding`
menyatakan batas provenance ini. Untuk pertanyaan faktual gunakan `SemanticMemory`
response berbukti; jangan menampilkan teks generator sebagai jawaban fakta tanpa verifikasi.
Loss/perplexity holdout hanya mengukur prediksi karakter pada korpus ini, bukan
koherensi, pemahaman, atau generalisasi bahasa Indonesia secara luas.

`ReasoningEngine::respond_text` mengorkestrasi response semantic dan `reason_text`:
input dipakai untuk jawaban berbukti, di-encode oleh `LanguageNetwork`, lalu melewati
recall `LearningMemory` dan transisi LIF. Hasil `ReasoningResult` saat ini adalah
prediksi aksi/ketidakpastian untuk task caller; tanpa mapping eksplisit dari aksi ke
proposisi, ia tidak dijadikan sumber fakta atau diubah menjadi kalimat jawaban.
Semantic response tetap hanya mengklaim isi yang didukung `SemanticMemory`. SDK menyediakan delegasi
`Session::respond_with_reasoning` dan `Session::learn_from_semantic_feedback`; caller
memasok `ReasoningEngine` dan `LanguageNetwork` dengan identitas neuron cocok.
Pembelajaran hanya terjadi bila caller memberikan expected/predicted action melalui
feedback API. Gap token unknown, ambigu, atau konflik menghentikan jalur belajar
semantic tanpa update. Memuat WordNet sendiri tidak melatih jaringan.

Tes respons terpisah pada fixture lokal melaporkan: intent **5/5**, factuality
terhadap evidence berlabel **5/5**, reasoning/negasi/kontradiksi/abstention **5/5**,
kriteria bentuk kalimat **6/6**, dan satu phrasing holdout **1/1**. Ini sampel kecil
yang ditulis untuk tes, bukan estimasi akurasi populasi. Regression test tambahan
mencakup pembalikan role, predicate mismatch, synonym edge berstatus source assertion,
resolusi satu pronomina, pertanyaan who, dan kontradiksi dengan retrieval limit 1.
Tes upstream terpisah juga meminta sinonim dari pasangan noun yang sungguh diimpor dari
WordNet dan memeriksa evidence ID/provenance MIT. Ini kontrak terbatas, bukan estimasi
akurasi populasi. Benchmark polaritas 100 contoh di bawah mengukur task terstruktur
yang berbeda. Holdout WH terpisah memakai pertanyaan alami `Siapa yang makan ikan?`
yang tidak pernah disimpan sebagai fakta: baseline tanpa konteks **0/1**, setelah
pernyataan `Kucing makan ikan.` ditambahkan lewat `analyze_in_context` menjadi **1/1**.
Ini mengukur retrieval jawaban dari input konteks, bukan peningkatan neural melalui
feedback; pertanyaan baseline dan pertanyaan evaluasi juga dipisahkan ke context ID.

Penghapusan konsep pada `ReasoningEngine::remove_semantic_concept` menjurnal task
konteks terkait di semantic store, lalu menghapus experience task itu dan context
frame neural. Jurnal bersifat retryable; operasi reasoning/learning berikutnya
menyelesaikan jurnal sebelum bekerja. Checkpoint menolak state dengan jurnal
penghapusan tertunda. SDK menyediakan `save_integrated_checkpoint` dan
`restore_integrated_checkpoint`, yang mengikat neural snapshot ke fingerprint
SHA-256 `MemoryStore`/`SemanticMemory` dan fingerprint `LearningMemory`; store tetap
file terpisah dan harus dibuka di path yang sama sebelum restore. Restore menolak
fingerprint berbeda sebelum mengubah jaringan. Operasi antar-file bukan satu
transaksi filesystem; jurnal penghapusan membantu retry, bukan atomic commit umum.

`evaluate_splits(training, validation, test)` menghitung exact-frame accuracy,
slot accuracy, dan coverage dari label acuan yang diberikan pemanggil. ID contoh
dan teks ternormalisasi harus unik antar split. Fungsi ini tidak melatih model,
tidak memilih konfigurasi dari validation, dan tidak membuktikan split eksternal
tidak bocor ke sumber pengetahuan; pemanggil tetap wajib memisahkan data sebelum
proses belajar. Tes integrasi repository memakai kalimat berbeda pada tiap split;
jumlahnya kecil dan hasilnya hanya mengukur aturan role/negasi di atas, bukan
generalisasi bahasa Indonesia secara luas.

`evaluate_grouped_splits` menambah syarat `group_id` sumber/semantik tidak boleh
muncul di lebih dari satu split. Evaluator tidak dapat membuktikan group yang
diberikan benar-benar mewakili provenance atau bahwa vocabulary/graph tidak pernah
melihat test; caller tetap bertanggung jawab pada provenance split.

Benchmark `benchmark_polaritas_100_test_group_held_out_membandingkan_baseline_dan_feedback`
menggunakan 50 train (5×5 pasangan noun dengan afirmasi/negasi), 50 validation
dari pasangan noun lain, dan 100 test dari 10×5 pasangan baru. Vocabulary dibangun
hanya dari train; semua 15 noun validation/test dikonfirmasi OOV. Group pasangan
subjek-objek disjoint lintas split. Setiap contoh train diprediksi lebih dulu;
label feedback aktual lalu memperbarui model secara incremental. Validation dan
test tidak pernah dipakai sebagai feedback. Test identik untuk baseline dan pascafeedback,
dan setiap prediksi memakai clone network serta snapshot LearningMemory sendiri
agar usage count tidak mengadaptasi state sepanjang evaluasi. Hasil aktual:
baseline **0/100**, validation setelah feedback **50/50**, test setelah feedback
**100/100**. Task hanya klasifikasi polaritas dari keberadaan `tidak` pada pola
kalimat sederhana dan data terstruktur ini; hasil tersebut bukan bukti paraphrase
umum, pemahaman bahasa, atau generalisasi lintas sumber. Benchmark ada pada
`tests/semantic_language.rs` dan dapat diulang dengan `cargo test --test
semantic_language benchmark_polaritas_100_test_group_held_out_membandingkan_baseline_dan_feedback --locked`.


## SemanticMemory-conditioned Transformer inference

`Session::generate_text` tetap merupakan API lama tanpa retrieval. Jalur baru
`Session::generate_text_with_semantic_memory(model, prompt, context_id, retrieval_limit,
config)` memakai `SemanticMemory` yang terpasang pada `Engine`, memanggil retrieval
kontekstual, lalu menyertakan hanya pernyataan recall utuh yang muat sebelum prompt
pengguna di jendela autoregresif Transformer. Nilai `retrieval_limit` dibatasi sampai
32. `MemoryConditionedGeneration` mengembalikan kandidat recall, subset yang benar-benar
dimasukkan, prompt pengguna asli, status, dan hasil model bila generasi dilakukan.

Jika tidak ada recall relevan atau pernyataannya tidak muat pada context window,
`generated` bernilai `None` dan status menunjukkan abstention; jalur ini tidak diam-diam
beralih ke generasi tanpa memory. Bila generasi terjadi, output tetap berlabel
`unverified`: skor recall adalah ranking heuristik, bukan confidence atau entailment,
dan model tidak memverifikasi klaim keluaran terhadap memory. `SemanticMemory::retrieve`
terutama mencari pernyataan konteks berdasarkan WordID/relasi langsung; hasil ini tidak
menambahkan provenance sumber leksikal ke klaim generator. Gunakan `respond_in_context`
untuk jalur jawaban semantik berbukti dan abstention yang sudah ada.

Decoding kini menolak kandidat yang akan mengulang karakter 4-gram dari kelanjutan
sebelumnya bila kandidat alternatif tersedia. Bila seluruh kandidat akan mengulang,
decoder kembali ke himpunan awal agar tidak dead-end; ini heuristik permukaan, bukan
jaminan koherensi dan dapat membatasi pola bahasa yang sah. `GeneratedText::repetition_metrics()`
melaporkan rasio karakter unik, repeated bigram/trigram/4-gram, adjacent repeat, EOS, dan
prompt truncation. Metrik tersebut bukan skor kelancaran. CLI `shella-lm metrics
--input-json <hasil-generate.json>` menghitung diagnostik ini dari JSON generasi yang
nyata. Perplexity tetap dihitung terpisah melalui `shella-lm evaluate` pada split yang
tercatat di manifest.

Checkpoint `SHLMLM01` tidak diubah oleh integrasi ini; store SemanticMemory tetap
persisten secara terpisah. Tes integrasi memulai executable test kedua, membuka ulang
store dan checkpoint, serta membandingkan hasil generasi deterministik. Generator masih
merupakan next-character LM kecil yang dilatih pada NusaX-Senti saja; recall tidak
mengubah tujuan training menjadi instruction tuning.

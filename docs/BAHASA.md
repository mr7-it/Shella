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
59 file yang dibundel juga terdaftar dalam manifest integritas versi 1
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
dataset yang cukup jelas. WordNet Bahasa pada URL terdahulu juga tidak tersedia
(404 saat diperiksa). Sumber-sumber yang tertahan ini tidak diberi provenance
resmi dan tidak dipakai untuk mengklaim kemampuan inferensi, relasi semantik,
atau pemahaman bahasa.

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
asal dan tanpa confidence rekaan. WordNet Bahasa tidak diaktifkan, sehingga tidak
ada graf relasi upstream yang diklaim tersedia secara default.

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
dengan confidence; edge hipotesis diberi bobot lebih rendah. Kecocokan ID atau
relasi wajib ada sebelum bonus konteks dapat berlaku. Nilai relevansi adalah skor
urut heuristik, bukan probabilitas. `infer` menelusuri edge `is-a`/`bagian-dari`
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

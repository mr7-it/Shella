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

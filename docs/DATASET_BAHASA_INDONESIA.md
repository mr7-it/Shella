# Dataset Bahasa Indonesia

Inventaris ini memisahkan lisensi kode dari lisensi data. Hanya data yang
memiliki izin penggunaan yang dapat diidentifikasi dan direvisi secara tepat
yang masuk allowlist runtime. Source text dan anotasi tidak dianggap memiliki
lisensi yang sama kecuali sumber menyatakannya.

## Status dan provenance

| Dataset | Revisi/identitas | Format dan ukuran yang diperiksa | Lisensi data | Status Shella |
|---|---|---|---|---|
| [stopwords-id](https://github.com/stopwords-iso/stopwords-id) | Commit `6109494eb4cfebd75c59bd832848374d02f4e907` | JSON UTF-8, 7,962 byte | MIT | Dibundel bersama pemberitahuan MIT. Loader memverifikasi Git blob SHA-1; stopwords dapat diterapkan sebelum input masuk `LanguageNetwork`. |
| [UD Indonesian-GSD](https://github.com/UniversalDependencies/UD_Indonesian-GSD) | Repository tersedia; revisi yang diperiksa `bc23d75691fd97255bbd420c4dbacd423b0ede95` | CoNLL-U; tidak diunduh | Lisensi anotasi treebank CC BY-SA 4.0, tetapi izin teks sumber tidak terbukti mencakup keseluruhan file | Tidak diaktifkan dan tidak ada datanya yang dibundel. |
| [UD Indonesian-CSUI](https://github.com/UniversalDependencies/UD_Indonesian-CSUI) | Repository tersedia | CoNLL-U; tidak diunduh | Metadata UD menyebut CC BY-SA, tetapi provenance mengarah ke korpus `idn-treebank` dan izin teks sumber belum dapat diverifikasi | Tidak diaktifkan dan tidak ada datanya yang dibundel. |
| [UD Indonesian-PUD](https://github.com/UniversalDependencies/UD_Indonesian-PUD) | Commit `1d8741838d6b8fa64dbda85a33c4af0700fb7430` | `id_pud-ud-test.conllu`, 1,161,372 byte; 1,000 kalimat | CC BY-SA 3.0, termasuk teks menurut pemberitahuan upstream | Dibundel dengan `LICENSE.txt`; loader memverifikasi Git blob SHA-1 dan mengimpor token, lemma satu-token, UPOS, FEATS, serta dependency ke `IndonesianVocabulary`. Ini split test, bukan data training. |
| [NusaX](https://github.com/IndoNLP/nusax) | Commit `75b0be5d982348332509f296e10d9bf9cb66d60f` | Data tree 4,162,995 byte; 50 CSV, seluruhnya dipin dengan Git blob SHA-1 | CC BY-SA 4.0 pada `datasets/LICENSE`; lisensi kode repository dicatat terpisah dan tidak digunakan sebagai lisensi data | Dataset yang memiliki parser tercantum pada tiga baris berikut. Source files dan notice asli dibundel. |
| NusaX-MT | Commit NusaX yang sama | 3 multilingual CSV; 1,871,004 byte | CC BY-SA 4.0 | Loader memvalidasi skema dan alignment 12 bahasa serta mengembalikan pasangan terjemahan. Tidak ada model penerjemah atau klaim peningkatan hasil. |
| NusaX-Senti | Commit NusaX yang sama | 36 CSV untuk 12 bahasa × train/valid/test; 2,019,311 byte | CC BY-SA 4.0 | Loader memvalidasi label `positive`/`neutral`/`negative`, ID, bahasa, dan split. Teks memakai tokenizer Shella; evaluator menerima prediksi eksternal dan menolak split train. Tidak ada classifier yang dilatih. |
| NusaX-Lexicon | Commit NusaX yang sama | 11 CSV Indonesian-to-language; 250,390 byte | CC BY-SA 4.0 | Loader menyediakan lookup frasa dan pencarian Indonesian; serialisasi memulihkan indeks. API dapat mengintern token pasangan ke vocabulary `LanguageNetwork`, tetapi tidak membuat sinaps atau belajar otomatis. |
| [IndoNLI](https://github.com/ir-nlp-csui/indonli) | Repository tersedia; tidak ada revisi data yang dipin | JSONL dan arsip terjemahan tersedia upstream | Tidak ditemukan lisensi dataset yang mencakup anotasi dan teks sumber | Ditolak oleh registry; tidak diunduh, tidak diparse, dan tidak digunakan untuk evaluasi. |
| [IndoSum](https://github.com/kata-ai/indosum) | Dataset Google Drive `1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco`; changelog `2019-03-20`; SHA-256 `34e5c504f711dd8ec0e87dbf10eb1dbaefca86717194f95a14476afeb8810108` | `tar.gz` 96,037,108 byte; JSONL per fold/split; ukuran arsip terurai 404,039,680 byte | CC BY-SA 4.0, tercantum di `README.txt` yang ikut dalam arsip; setiap record mempertahankan `source` dan `source_url` | Arsip dan notice data asli dibundel. Loader memverifikasi hash, membaca satu JSONL split per panggilan, memvalidasi record, menyediakan oracle extractive berlabel, dan menghitung token-level ROUGE-1 untuk kandidat pemanggil. Tidak ada summarizer yang dilatih. |

WordNet Bahasa pada URL yang diajukan tidak ditemukan tersedia saat diperiksa
(404). Kamus Hunspell bukan salah satu dari sepuluh dataset permintaan ini;
loader/provenance yang sudah ada tetap terpisah.

## API runtime

`LicenseRegistry::verified_sources()` hanya menyetujui stopwords-id, PUD,
NusaX, IndoSum, dan sumber Hunspell terdahulu. `DatasetLoader` menolak berkas
yang Git blob SHA-1-nya tidak cocok dengan revisi registry. Batas pembacaan dan
validasi skema diterapkan sebelum hasil dikembalikan.

Contoh path dari root repository:

```rust
let registry = LicenseRegistry::verified_sources();
let stopwords = DatasetLoader::load_stopwords_json(
    "data/language/stopwords-id.json",
    &registry,
)?;
let sentiment = DatasetLoader::load_nusax_sentiment_csv(
    "data/language/nusax/sentiment/indonesian/test.csv",
    "indonesian",
    NusaXSplit::Test,
    &registry,
)?;
let indosum = DatasetLoader::load_indosum_archive(
    "data/language/indosum/indosum.tar.gz",
    1,
    IndoSumPartition::Test,
    &registry,
)?;
```

`load_ud_conllu` dapat digunakan pada PUD yang dipin untuk menyimpan anotasi
struktural dalam lexicon dan checkpoint Shella. NusaX lexicon dapat dicari
langsung atau diintern ke `LanguageNetwork`; hubungan bahasa yang dihasilkan
tetap berupa WordID, belum asosiasi semantik atau bobot neural.

Dataset asli dipertahankan di `data/language/` agar tersedia setelah restart.
Dataset yang sudah diparse dapat dibaca ulang dari sumber lokal; vocabulary PUD
memiliki jalur serialisasi/checkpoint yang sudah ada, dan NusaX lexicon
menyediakan JSON round-trip dengan validasi provenance. Evaluasi NusaX dan
IndoSum hanya melaporkan metrik dari prediksi/teks kandidat yang diberikan
pemanggil. Hasil tersebut tidak menunjukkan Shella telah belajar, dan tidak
mengukur pemahaman atau reasoning umum.

## Batasan lisensi dan cakupan

- GSD dan CSUI tetap tidak tersedia pada allowlist sampai izin atas seluruh
  teks sumber dapat dipastikan secara terpisah dari lisensi anotasi.
- IndoNLI tetap tidak tersedia sampai ada lisensi dataset yang mencakup teks
  dan anotasinya. Tidak ada metrik inferensi IndoNLI yang diklaim.
- Loader IndoSum membaca arsip terverifikasi ke struktur satu split di memori;
  arsip 96 MB disimpan terkompresi dan tidak diekstrak permanen. Memuat split
  penuh membutuhkan memori sebanding ukuran JSONL split tersebut.
- Tidak ada sumber yang digunakan untuk melatih model dalam perubahan ini.

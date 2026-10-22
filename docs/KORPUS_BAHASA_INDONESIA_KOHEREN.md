# Korpus prosa Bahasa Indonesia koheren untuk Shella

## Ringkasan dan sumber yang dipilih

Korpus ini memperluas subkorpus awal lima undang-undang menjadi gabungan **prosa ensiklopedis Indonesia dan teks undang-undang**. Sumber utama adalah dump artikel Wikipedia bahasa Indonesia yang dipin ke snapshot resmi **1 Juli 2026**. Arsip `idwiki-20260701-pages-articles-multistream.xml.bz2` berukuran **1.293.461.998 byte**; checksum SHA-1 resmi dan lokal cocok (`522c21bfe2bc9bfb8361a0cb6ac43f809c72021e`), dan SHA-256 lokal disimpan dalam inventaris korpus. Arsip mentah besar tidak disimpan di Git; script mengunduh ulang snapshot yang sama dengan `--download` dan menolak arsip yang checksum-nya berbeda.

Dump mentah tidak langsung digunakan. Hanya halaman namespace utama yang bukan redirect yang dipertimbangkan. Builder mengubah wikitext menjadi prosa, mengambil maksimum **20.000 artikel** dengan prioritas hash deterministik (seed 42) dan batas **100.000.000 karakter-token**, lalu menggabungkannya dengan teks lima undang-undang dari subkorpus awal. Karena ada batas dokumen dan volume, angka akhir tercatat dalam [`corpus_stats.json`](../data/indonesian_coherent_corpus/corpus_stats.json), bukan diasumsikan sama dengan ukuran dump atau jumlah halaman Wikipedia.

Wikipedia dipilih sebagai sumber utama karena menyediakan artikel kontemporer dengan prosa multi-paragraf dan skala lebih besar daripada koleksi kalimat terisolasi. Audit sumber paralel menemukan Project Gutenberg tidak memiliki eBook berlabel bahasa `id` pada katalog resmi saat diperiksa; SIBI belum memberi lisensi terbuka yang jelas untuk ekstraksi/pelatihan model; Tatoeba berisi kalimat pendek tanpa konteks; Wikisource dan Wikibuku memerlukan audit hak serta mutu per karya/halaman. Sumber-sumber itu tidak dicampurkan ke dataset ini.

## Pengolahan dan mutu

Implementasi yang dapat dijalankan ulang adalah [`tools/build_indonesian_coherent_corpus.py`](../tools/build_indonesian_coherent_corpus.py); versinya dipin di [`tools/requirements-indonesian-corpus.txt`](../tools/requirements-indonesian-corpus.txt). Pipeline membaca dump bz2/XML secara streaming, memeriksa checksum sebelum parse, tidak menyimpan username kontributor atau data pengguna lain, dan memetakan setiap record Wikipedia ke judul, page ID, revision ID/timestamp, URL artikel/riwayat, lisensi, metode atribusi, dan catatan perubahan pada [`sources.json`](../data/indonesian_coherent_corpus/sources.json).

Filter kualitas yang diterapkan meliputi:

- namespace `0` saja; redirect, halaman nonartikel dan revisi kosong dibuang;
- teks hasil parse harus terdeteksi sebagai bahasa Indonesia (`id`) oleh `langid` 1.1.6;
- sekurang-kurangnya 900 karakter, dua paragraf substansial, 100 kata, dan lima kalimat;
- rasio alfabetik minimal 0,75 dan keragaman kata unik minimal 0,20;
- templat, tabel, daftar, kategori/media, referensi, URL, serta bagian referensi/pranala luar dan struktur nonprosa dibuang; baris/teks dinormalisasi NFC, lowercase, dan whitespace;
- halaman dengan templat peringatan hak cipta/fair use yang eksplisit dikecualikan; artikel lebih dari 1 MiB setelah ekstraksi juga dikecualikan;
- duplikasi exact dihapus melalui SHA-256 teks normalisasi, lalu near-duplicate melalui SimHash 64-bit atas word 5-gram (jarak Hamming ≤3 dan rasio panjang ≥0,80).

Filter ini membantu mempertahankan unit artikel yang koheren; ia **bukan** penyuntingan fakta, koreksi tata bahasa, penjaminan netralitas, maupun evaluasi kesesuaian setiap artikel untuk semua penggunaan. [`sample_review.json`](../data/indonesian_coherent_corpus/sample_review.json) berisi sampel artikel nyata dengan tautan sumber untuk tinjauan manusia.

## Split dan pengukuran token

Deduplikasi dilakukan sebelum split. Satu artikel penuh adalah satu record dan satu unit split, sehingga bagian artikel yang sama tidak terpecah ke train/validasi/test. Pembagian deterministik menggunakan seed 42 dengan target 98% train, 1% validasi, dan 1% test; ID semua dokumen dicatat di manifest. Lima undang-undang tetap satu dokumen per sumber resmi. Jumlah record dan token setiap split serta total berada di `corpus_stats.json`.

Angka `model_character_tokens` memakai tokenizer Shella saat ini: satu Unicode code point setelah NFC/lowercase/normalisasi whitespace dihitung sebagai satu karakter-token. Angka ini bukan BPE/subword dan tidak boleh dibandingkan langsung dengan ukuran token model lain. `encoded_tokens_with_bos_eos` menambahkan BOS dan EOS untuk tiap dokumen; `target_tokens_with_eos` menghitung token teks dan EOS tanpa BOS. Definisi ini ditulis di statistik build.

### Hasil build yang benar-benar selesai

| Split | Dokumen | Karakter-token Shella | Token termasuk BOS/EOS | Kata whitespace |
|---|---:|---:|---:|---:|
| Train | 19.604 | 69.927.479 | 69.966.687 | 9.675.157 |
| Validasi | 200 | 708.033 | 708.433 | 98.859 |
| Test | 201 | 661.831 | 662.233 | 91.091 |
| **Total** | **20.005** | **71.297.343** | **71.337.353** | **9.865.107** |

Jumlah tersebut terdiri dari maksimum 20.000 artikel Wikipedia terpilih dan lima teks undang-undang. Dari 1.863.345 halaman yang dibaca, 186.640 artikel lolos pemeriksaan bahasa/kualitas sebelum pemilihan deterministik; 20.000 artikel terpilih menyumbang 70.904.993 karakter-token. Setelah gabungan, deduplikasi exact/near-duplicate menghapus **0** dokumen. JSON prepared berukuran 72.759.549 byte dengan SHA-256 `abcdce857d44746e70974af6505e063c6dd4f6828365b599ab2de01660cb7fa2`; daftar ID split, source/license/revision metadata, dan checksum dicatat dalam manifest/inventaris.

Validasi aktual: `cargo test --test prepared_legal_corpus --locked -- --ignored korpus_koheren_indonesia_aktual_lolos_loader_dan_dapat_membentuk_model` lulus (1 passed). Tes memuat berkas JSON dan inventaris 20.005 dokumen melalui validator Rust serta berhasil membentuk objek Transformer Shella. Ini belum menjalankan optimizer/pretraining model pada seluruh korpus.

## Lisensi dan atribusi

Korpus ini **berlisensi campuran per dokumen**; rincian per record terdapat di `sources.json`, dan pemberitahuan ringkas disediakan dalam [`LICENSE.txt`](../data/indonesian_coherent_corpus/LICENSE.txt).

Teks asli komunitas Wikimedia pada umumnya dilisensikan di bawah CC BY-SA 4.0 dan GFDL, tetapi sebagian teks mungkin hanya CC BY-SA atau mempunyai notasi/syarat atribusi tambahan karena materi impor. Wikimedia Terms of Use menerima atribusi melalui URL artikel/riwayat (yang mencantumkan kontributor), salinan stabil setara, atau daftar penulis; Terms juga meminta pemberitahuan lisensi, penandaan perubahan, dan pemenuhan share-alike saat mendistribusikan teks turunan. Pipeline menyimpan URL artikel dan revisi, mencatat bahwa markup/struktur telah diubah, dan tidak mengambil media.

Wikimedia memperingatkan dump mungkin memuat pengecualian fair use dan pelanggaran hak cipta yang belum terdeteksi. Filter peringatan eksplisit tidak dapat membuktikan bahwa setiap kutipan/impor telah terselesaikan. Karena itu provenance dan batas ini disimpan; peninjauan hukum tetap perlu sebelum redistribusi dataset atau model. Status CC BY-SA/GFDL terhadap bobot model hasil training **tidak ditetapkan** oleh pipeline ini.

Lima teks undang-undang berasal dari PDF resmi JDIH BPK yang checksum-nya diverifikasi. Dasar hak dibatasi pada teks peraturan perundang-undangan berdasarkan Pasal 42(b) UU No. 28 Tahun 2014; ini tidak mengklaim tata letak PDF, gambar, komentar, atau semua materi situs pemerintah sebagai domain publik. Audit dan provenance subkorpus legal ada pada [`KORPUS_HUKUM_INDONESIA.md`](KORPUS_HUKUM_INDONESIA.md).

## Bangun ulang, validasi, dan training

Dari root repository, instal dependency yang dipin lalu bangun ulang dari unduhan resmi (arsip sekitar 1,3 GB):

```sh
python3 -m pip install -r tools/requirements-indonesian-corpus.txt
python3 tools/build_indonesian_coherent_corpus.py --download \
  --archive /tmp/idwiki-20260701-pages-articles-multistream.xml.bz2 \
  --seed 42 --max-documents 20000 --max-characters 100000000
```

Hasil utama yang dibaca oleh Shella adalah `data/indonesian_coherent_corpus/prepared_corpus.json`. Loader Rust memeriksa checksum/Git-blob provenance `sources.json`, proporsi/ID split, jumlah karakter-token, normalisasi, serta batas 1 MiB per dokumen dan 512 MiB per berkas JSON.

Contoh menjalankan training yang sesuai CLI saat ini (ini hanya contoh penggunaan; **belum** berarti model 2 miliar parameter atau training besar sudah dijalankan):

```sh
cargo run --locked --bin shella-lm -- train \
  --dataset data/indonesian_coherent_corpus/prepared_corpus.json \
  --checkpoint target/indonesian-coherent.sllm \
  --split-seed 42 --epochs 1 --batch-size 8 \
  --hidden-size 32 --layers 1 --heads 2 --context 64
```

Korpus yang lebih besar tidak otomatis menghasilkan model lebih besar atau lebih baik. Implementasi Transformer Shella saat ini masih CPU-only dan dibatasi hingga 4.000.000 parameter; target 2 miliar parameter memerlukan perubahan arsitektur/backend, perangkat training dan rancangan data yang terpisah. Ukuran training aktual juga dapat mahal meski contoh di atas memakai arsitektur kecil.

## Referensi primer

- [Indeks dump resmi idwiki snapshot 2026-07-01](https://dumps.wikimedia.org/idwiki/20260701/)
- [Informasi lisensi resmi dump Wikimedia](https://dumps.wikimedia.org/legal.html)
- [Wikimedia Terms of Use — Section 7, Licensing of Content](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)
- [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- [GNU Free Documentation License 1.3](https://www.gnu.org/licenses/fdl-1.3.html)
- [Dokumentasi MediaWiki API: Parsing wikitext](https://www.mediawiki.org/wiki/API:Parsing_wikitext)
- [UU Hak Cipta No. 28 Tahun 2014 pada situs resmi DJKI](https://www.dgip.go.id/unduhan/download/uu-nomor-28-tahun-2014-tentang-hak-cipta-34-2014)

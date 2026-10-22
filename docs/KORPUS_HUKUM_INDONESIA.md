# Subkorpus hukum Indonesia untuk Shella

## Cakupan dan dasar hak

Koleksi awal ini berisi lima undang-undang Republik Indonesia berbahasa Indonesia dari JDIH BPK: Hak Cipta (2014), Keterbukaan Informasi Publik (2008), Pengelolaan Sampah (2008), Sistem Pendidikan Nasional (2003), dan Pendidikan Tinggi (2012). Setiap dokumen memiliki halaman metadata, URL PDF, status sebagaimana tercatat pada sumber, dan SHA-256 PDF di [`sources.json`](../data/indonesian_legal_corpus/sources.json). PDF asli disimpan di `data/indonesian_legal_corpus/raw/`.

Dasar domain publiknya dibatasi pada **teks peraturan perundang-undangan**: Pasal 42(b) UU No. 28 Tahun 2014 menyatakan tidak ada Hak Cipta atas peraturan perundang-undangan. Teks resmi pasalnya tersedia pada situs DJKI. Ini bukan lisensi umum atas semua materi yang ditempatkan di situs pemerintah. Karena itu koleksi hanya mengambil teks undang-undang melalui OCR dan tidak memasukkan abstrak, komentar, putusan pengadilan, artikel, atau materi pihak ketiga. Disclaimer JDIH BPK meminta pengguna mengonfirmasi ke instansi sumber bila isi berbeda. Referensi dan alasan eksklusi sumber lain dicatat di [`AUDIT_LISENSI_KORPUS_INDONESIA.md`](AUDIT_LISENSI_KORPUS_INDONESIA.md). Ini bukan pendapat hukum; minta tinjauan hukum sebelum penggunaan komersial atau redistribusi.

## Pemrosesan yang benar-benar dijalankan

[`tools/build_indonesian_legal_corpus.py`](../tools/build_indonesian_legal_corpus.py) memverifikasi hash setiap PDF terhadap inventaris, merender halaman, menjalankan Tesseract dengan model bahasa Indonesia dan Inggris (PSM 4, empat worker maksimum), membersihkan header/footer, fragmen scan dan pengulangan kata, menangani pemenggalan baris, menormalkan NFC/lowercase, menghapus duplikat exact dan near-duplicate, lalu membagi keluarga dokumen sebelum training dengan seed tetap. PDF lama berjenis scan; OCR tetap bisa salah dan pipeline ini tidak melakukan koreksi fakta manual, sehingga teks harus ditinjau terhadap PDF sumber sebelum dipakai untuk sistem produksi atau menjawab fakta hukum.

Hasil yang dibuat oleh pipeline:

- `data/indonesian_legal_corpus/extracted/*.txt`: teks OCR bersih per dokumen;
- `data/indonesian_legal_corpus/prepared_corpus.json`: split dan manifest yang dapat dibaca CLI Shella;
- `data/indonesian_legal_corpus/corpus_stats.json`: ukuran dokumen, deduplikasi, split, jumlah kata, serta hitungan token.

Metrik token mengikuti tokenizer model Shella saat ini: satu **Unicode code point** setelah NFC, lowercase, dan normalisasi whitespace dihitung sebagai satu karakter-token; statistik `model_character_tokens` tidak menghitung token khusus, sedangkan `encoded_tokens_with_bos_eos` menambahkan BOS dan EOS untuk setiap dokumen. Angka ini bukan token subword dan tidak dapat dibandingkan langsung dengan hitungan tokenizer BPE/Unigram.

Snapshot final setelah OCR/pembersihan: **392.350 karakter-token** total (52.603 kata whitespace), terdiri dari train **180.599** token pada 3 dokumen, validasi **111.153** pada 1 dokumen, dan test **100.598** pada 1 dokumen. Token input termasuk BOS/EOS berjumlah 392.360. Deduplikasi mendeteksi dan menghapus 0 dokumen pada snapshot ini. Dengan definisi tokenizer serta shuffle/deduplikasi Rust yang sama, train legal ini sekitar **2,94×** jumlah train NusaX internal yang menjadi baseline (61.339 karakter-token dari 400 dokumen); NusaX tidak dicampurkan ke koleksi legal ini dan perbandingan tersebut hanya ukuran, bukan klaim kualitas atau izin baru.

## Bangun ulang dan gunakan untuk training

Dari root repository, pastikan `pdftoppm`, `pdfinfo`, `tesseract`, dan data bahasa Tesseract `ind` tersedia. Jalankan:

```sh
python3 tools/build_indonesian_legal_corpus.py --ocr-dpi 250
```

Untuk menguji ulang aturan pembersihan pada ekstraksi yang sudah tersedia tanpa merender ulang PDF, gunakan `python3 tools/build_indonesian_legal_corpus.py --reuse-extracted`; opsi ini memverifikasi ulang hash PDF dan memakai ekstraksi serta statistik halaman dari build sebelumnya.

CLI training menerima berkas JSON prepared, memvalidasi manifest, identitas split, jumlah karakter-token, provenance, dan checksum sebelum optimizer berjalan:

```sh
cargo run --locked --bin shella-lm -- train \
  --dataset data/indonesian_legal_corpus/prepared_corpus.json \
  --checkpoint target/indonesian-legal-laws.sllm \
  --split-seed 42 --epochs 1 --batch-size 8 \
  --hidden-size 32 --layers 1 --heads 2 --context 64
```

Validasi ulang checkpoint dengan dataset yang sama:

```sh
cargo run --locked --bin shella-lm -- evaluate \
  --checkpoint target/indonesian-legal-laws.sllm \
  --dataset data/indonesian_legal_corpus/prepared_corpus.json \
  --batch-size 8
```

Seed `--split-seed` ketika training harus sama dengan seed di manifest JSON. Split test tidak digunakan pada optimizer dan evaluasi test terjadi setelah training. File JSON tidak mengubah atau menimpa dataset NusaX yang sudah ada; NusaX dan korpus hukum tetap sumber terpisah dengan provenance serta hak berbeda.

## Batas kapasitas dan cakupan

Koleksi ini adalah **subkorpus legal awal**, bukan korpus bahasa Indonesia umum dan belum mencapai miliaran token. Lima dokumen memberi variasi bidang, tetapi jumlah dokumen terlalu kecil untuk evaluasi holdout yang kuat; split berbasis dokumen juga tidak menguji generalisasi lintas portal karena semua PDF berasal dari JDIH BPK. Korpus IndoGovBERT C2 (59.729 baris pada kartu Hugging Face) tidak dimasukkan karena kartu dataset tidak menyebut lisensi spesifik dan kontennya mencampur jenis dokumen. Loader JSON saat ini membaca berkas penuh ke memori dan menolak berkas di atas 512 MiB; untuk korpus skala besar diperlukan sharding/streaming, bukan memperbesar batas ini begitu saja.

Transformer Shella yang ada dibatasi implementasinya sampai **4.000.000 parameter**, CPU-only, hidden size maksimal 256, empat layer, dan context maksimal 256 token. Jadi permintaan **2 miliar parameter belum dibuat** dan tidak dapat dicapai hanya dengan memperbesar konstanta; diperlukan perubahan arsitektur/backend dan perangkat training yang jauh lebih besar, serta korpus miliaran token yang lisensinya cleared. Kode tidak menaikkan batas itu karena akan menghasilkan konfigurasi yang tidak praktis/tidak dapat dilatih dalam lingkungan aktif.

## Sumber primer

- [Pasal 42 UU No. 28 Tahun 2014, teks pada situs resmi DJKI](https://www.dgip.go.id/unduhan/download/uu-nomor-28-tahun-2014-tentang-hak-cipta-34-2014)
- [Metadata dan PDF UU No. 28 Tahun 2014 di JDIH BPK](https://peraturan.bpk.go.id/Details/38690/uu-no-28-tahun-2014)
- [Katalog Undang-Undang Ditjen Peraturan Perundang-undangan](https://peraturan.go.id/uu)

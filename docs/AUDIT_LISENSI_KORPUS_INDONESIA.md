# Audit lisensi sumber korpus Indonesia

Audit ini menilai apakah teks dapat dimasukkan secara aman ke korpus pra-pelatihan, terutama bila model atau data turunan kelak didistribusikan secara komersial. Ini bukan pendapat hukum. Lisensi kode tidak otomatis melisensikan isi dataset, dan label lisensi sebuah kumpulan tidak membuktikan hak atas setiap teks sumber.

## Keputusan sumber yang diperiksa

- **Wikipedia bahasa Indonesia:** dump resminya berisi teks dengan lisensi yang berbeda-beda, umumnya CC BY-SA 4.0 dan/atau GFDL, tetapi materi impor, media, dan konten pihak ketiga dapat mengikuti syarat lain. Wikimedia meminta atribusi dan pemeriksaan sumber per halaman. Dampak ShareAlike/GFDL pada bobot model tidak dijelaskan secara khusus. **Tidak dimasukkan ke korpus pra-pelatihan komersial tanpa tinjauan hukum dan pencatatan provenance per halaman.** [1] [2] [3]
- **Wikibuku bahasa Indonesia:** halaman proyek menyatakan CC BY-SA 4.0 dan GFDL untuk teks, tetapi halaman/kontribusi impor dapat memiliki syarat lain. Dump bukan bukti bahwa hak atas semua isi homogen; bobot model dan ShareAlike juga belum terjawab oleh dokumentasi yang diperiksa. **Ditahan sampai ada review hukum dan pipeline atribusi yang dapat dipenuhi.** [2] [4] [5]
- **Wikisumber bahasa Indonesia:** isi dapat berupa naskah domain publik, terjemahan, hasil transkripsi, atau materi berlisensi. Kebijakan lokal yang dibuka tidak sepenuhnya konsisten mengenai versi lisensi, sementara dump bukan teks bersih dan tidak membuktikan hak atas karya sumber. **Hanya dapat dipertimbangkan dengan allowlist per karya/halaman dan verifikasi hak; dump mentah dikecualikan.** [1] [2] [6]
- **Universal Dependencies Indonesian (PUD/GSD/CSUI):** lisensi berbeda antar-treebank. PUD menggunakan CC BY-SA 3.0; GSD memakai CC BY-SA 4.0 untuk anotasi tetapi README memisahkan hak teks dasar; CSUI mempunyai rantai sumber yang mencakup lisensi nonkomersial dan larangan redistribusi. **Seluruh koleksi dikecualikan untuk pra-pelatihan komersial; PUD tetap berguna untuk evaluasi linguistik terpisah.** [7] [8] [9] [10] [11]
- **NusaX Indonesia:** dokumentasi menyebut CC BY-SA 4.0 untuk data, tetapi data sentimen berakar pada ulasan/komentar pihak ketiga dan lisensi sumber SmSA tidak konsisten antara dokumentasi yang diperiksa. Lisensi Apache di root repository mencakup kode, bukan otomatis teks. **Tidak dianggap cleared untuk pra-pelatihan komersial sebelum ada klarifikasi hak teks dari pengelola/pemegang hak.** [12] [13] [14] [15]
- **IndoSum:** repository berlisensi Apache-2.0, tetapi halaman yang diaudit tidak menetapkan lisensi terpisah untuk artikel berita dan ringkasan yang bersumber dari penerbit lain. Apache atas kode bukan izin untuk menyalin isi berita. **Dikecualikan sampai izin atas isi teks dikonfirmasi.** [16] [17] [18]
- **MADLAD-400 Indonesia:** deklarasi lisensi yang ditemukan tidak konsisten (README menyebut CC BY 4.0, metadata menampilkan ODC-By). Dokumentasi Common Crawl menegaskan bahwa hak atas isi crawled tetap pada pemilik sumber; label database tidak menyelesaikan hak cipta per halaman. **Dikecualikan dari pra-pelatihan komersial secara default.** [19] [20] [21] [22]

## Aturan penggunaan korpus Shella

1. Setiap sumber harus mempunyai URL, revisi/snapshot, lisensi dataset, lisensi teks sumber, atribusi, dan hash berkas yang dicatat terpisah.
2. Data dengan hak teks yang tidak jelas tidak boleh ikut dihitung sebagai data “cleared”, walaupun kode loader atau repository induknya open-source.
3. Dokumen dipisah sebelum training berdasarkan ID dokumen atau kelompok sumber agar salinan/varian dari sumber yang sama tidak bocor ke validasi/test.
4. Perhitungan jumlah data harus menyebut tokenisasi yang digunakan. Korpus model Shella saat ini memakai token karakter; hitungan karakter token tidak boleh disamakan dengan token subword untuk model skala besar.

## Referensi

[1]: https://dumps.wikimedia.org/legal.html "Kebijakan hukum dump Wikimedia"
[2]: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use "Ketentuan penggunaan Wikimedia"
[3]: https://creativecommons.org/licenses/by-sa/4.0/legalcode.en "Teks legal CC BY-SA 4.0"
[4]: https://id.wikibooks.org/wiki/Wikibuku:Hak_cipta "Kebijakan hak cipta Wikibuku Indonesia"
[5]: https://dumps.wikimedia.org/other/mediawiki_content_history/readme.html "Dokumentasi ekspor konten MediaWiki"
[6]: https://id.wikisource.org/wiki/Wikisumber:Hak_cipta "Kebijakan hak cipta Wikisumber Indonesia"
[7]: https://universaldependencies.org/treebanks/id_pud/index.html "Treebank Indonesian PUD"
[8]: https://github.com/UniversalDependencies/UD_Indonesian-PUD "Repository Indonesian PUD"
[9]: https://github.com/UniversalDependencies/UD_Indonesian-GSD "Repository Indonesian GSD"
[10]: https://github.com/UniversalDependencies/UD_Indonesian-CSUI "Repository Indonesian CSUI"
[11]: https://github.com/famrashel/idn-treebank "Sumber treebank idn-treebank"
[12]: https://github.com/IndoNLP/nusax "Repository dan README NusaX"
[13]: https://huggingface.co/datasets/indonlp/NusaX-MT "Kartu dataset NusaX-MT"
[14]: https://huggingface.co/datasets/indonlp/indonlu "Kartu dataset IndoNLU"
[15]: https://github.com/IndoNLP/nusa-crowd/blob/master/nusacrowd/nusa_datasets/smsa/smsa.py "Loader sumber SmSA di NusaCrowd"
[16]: https://github.com/kata-ai/indosum "Repository IndoSum"
[17]: https://github.com/kata-ai/indosum/blob/master/LICENSE.txt "Lisensi repository kode IndoSum"
[18]: https://arxiv.org/html/1810.05334v5 "Paper IndoSum"
[19]: https://huggingface.co/datasets/allenai/MADLAD-400 "Kartu dataset MADLAD-400"
[20]: https://huggingface.co/datasets/allenai/MADLAD-400/raw/main/README.md "README dan deklarasi lisensi MADLAD-400"
[21]: https://opendatacommons.org/licenses/by/1-0/ "Lisensi Open Data Commons Attribution"
[22]: https://commoncrawl.org/terms-of-use "Ketentuan penggunaan Common Crawl"


## Sumber pemerintah yang diperiksa berikutnya

Teks peraturan perundang-undangan adalah kandidat paling jelas untuk subkorpus domain publik: Pasal 42(b) UU No. 28 Tahun 2014 menyebut bahwa tidak ada Hak Cipta atas peraturan perundang-undangan. Pasal 42(c) juga menyebut pidato kenegaraan atau pidato pejabat pemerintah. JDIH BPK menerbitkan metadata peraturan dan tautan PDF, tetapi disclaimer-nya menyatakan dokumen berasal dari instansi pemerintah terkait; bila ada perbedaan, pengguna harus mengonfirmasi ke instansi sumber. Korpus yang disusun dari sini dibatasi pada teks peraturan itu sendiri, bukan abstrak, komentar, atau materi pihak ketiga. Ini adalah dasar hukum untuk subset hukum, bukan lisensi blanket bagi semua isi di situs pemerintah. [23] [24]

Portal resmi Ditjen Peraturan Perundang-undangan menampilkan 62.626 peraturan dalam katalog dan 1.931 entri kategori Undang-Undang pada saat audit. Karena ini menghitung catatan katalog, bukan dokumen bersih atau token, angka tersebut tidak diperlakukan sebagai ukuran korpus. [25]

IndoGovBERT C2 tampak menarik dari sisi skala: kartu Hugging Face menampilkan 59.729 baris dan ukuran berkas sekitar 367 MB. Namun kartu dataset tidak memuat pernyataan lisensi yang tegas. Paper menjelaskan C2 mencampur undang-undang/peraturan, laporan pemerintah, dan pidato dari beberapa situs; keterangan “open-source license” di paper/repository tidak memberi izin spesifik untuk setiap teks. Karena itu C2 tidak dimasukkan sebagai data cleared. [26] [27] [28]

Sumber-sumber tersebut masih memerlukan pemeriksaan kualitas ekstraksi dan pemisahan dokumen. PDF lama dapat berupa scan; tidak dipakai jika teks OCR tidak terbaca. Unduhan yang dipakai harus mempertahankan URL JDIH, metadata judul/nomor/tahun, hash PDF, hasil ekstraksi, dan perhitungan token model karakter Shella.

### Dokumen yang benar-benar dihimpun

Snapshot awal hanya memasukkan lima teks undang-undang dari JDIH BPK: UU No. 28 Tahun 2014 tentang Hak Cipta, UU No. 14 Tahun 2008 tentang Keterbukaan Informasi Publik, UU No. 18 Tahun 2008 tentang Pengelolaan Sampah, UU No. 20 Tahun 2003 tentang Sistem Pendidikan Nasional, dan UU No. 12 Tahun 2012 tentang Pendidikan Tinggi. URL detail dan PDF, status di halaman sumber, ukuran, dan SHA-256 masing-masing dicatat di [`sources.json`](../data/indonesian_legal_corpus/sources.json); teks OCR, split, deduplikasi, dan token count dibuat oleh pipeline yang didokumentasikan di [`KORPUS_HUKUM_INDONESIA.md`](KORPUS_HUKUM_INDONESIA.md).

Dasar penggunaan dibatasi pada teks peraturan perundang-undangan menurut Pasal 42(b) UU No. 28 Tahun 2014. Gambar, logo, tata letak PDF, abstrak, komentar, putusan, serta materi pihak ketiga tidak dinyatakan domain publik oleh keputusan ini dan tidak dimasukkan sebagai teks training. PDF disimpan sebagai sumber audit lokal; sebelum redistribusi raw PDF atau penggunaan komersial, hak atas berkas/elemen selain teks hukum harus ditinjau terpisah. Set ini belum menjadi korpus bahasa umum dan bukan bukti izin untuk dokumen pemerintah lainnya.

### Dump Wikipedia Indonesia yang diproses untuk korpus prosa

Perluasan terbaru mengambil snapshot resmi idwiki tanggal **2026-07-01**, bukan crawl bebas yang tidak berversi. Indeks resmi dan metadata `dumpstatus.json` mengidentifikasi berkas `idwiki-20260701-pages-articles-multistream.xml.bz2`; checksum SHA-1 yang dicantumkan berkas checksums resmi cocok dengan arsip lokal. Arsip berukuran 1.293.461.998 byte, SHA-1 `522c21bfe2bc9bfb8361a0cb6ac43f809c72021e`, SHA-256 `c8abefcddb80ce38a0668ac8b33199646bd515b17b17411b5da4cb5ee5360405`. Hasil artikel per revisi disimpan sebagai atribusi/provenance, bukan arsip contributor usernames. Hash, URL artikel/revisi, page ID, revision ID, lisensi, dan perubahan normalisasi dicatat per record di [`data/indonesian_coherent_corpus/sources.json`](../data/indonesian_coherent_corpus/sources.json).

Dump Wikimedia sendiri menjelaskan bahwa teks komunitas pada umumnya tersedia di bawah CC BY-SA dan/atau GFDL; Terms of Use juga memuat kewajiban atribusi, tautan lisensi, pemberitahuan perubahan, dan share-alike saat distribusi konten turunan. Namun dump dapat mengandung materi impor, lisensi tambahan atau pengecualian yang tidak terdeteksi. Karena itu lisensi korpus **campuran per artikel**, bukan satu izin universal: pengguna harus memeriksa riwayat/atribusi dan persyaratan pada halaman yang bersangkutan. Builder membuang media dan penanda pengecualian hak cipta yang eksplisit, tetapi tidak dapat menyelesaikan hak setiap materi impor. Korpus ini belum dinyatakan cleared untuk distribusi komersial; efek lisensi ShareAlike/GFDL pada bobot model juga belum diputuskan. Lihat [`LICENSE.txt`](../data/indonesian_coherent_corpus/LICENSE.txt) untuk ringkasan atribusi dan batas hak.

Sumber primer snapshot dan syarat yang diperiksa:

- [Indeks dump idwiki 2026-07-01](https://dumps.wikimedia.org/idwiki/20260701/), [checksum resmi](https://dumps.wikimedia.org/idwiki/20260701/idwiki-20260701-sha1sums.txt), dan [status dump machine-readable](https://dumps.wikimedia.org/idwiki/20260701/dumpstatus.json).
- [Ketentuan hukum dump Wikimedia](https://dumps.wikimedia.org/legal.html) dan [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).
- [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) dan [GFDL 1.3](https://www.gnu.org/licenses/fdl-1.3.html).
- [Dokumentasi parsing wikitext MediaWiki](https://www.mediawiki.org/wiki/API:Parsing_wikitext).

[23]: https://www.dgip.go.id/unduhan/download/uu-nomor-28-tahun-2014-tentang-hak-cipta-34-2014 "Teks UU Hak Cipta pada situs resmi DJKI"
[24]: https://peraturan.bpk.go.id/Details/38690/uu-no-28-tahun-2014 "Metadata dan file UU Hak Cipta di JDIH BPK"
[25]: https://peraturan.go.id/uu "Katalog resmi undang-undang Ditjen PP"
[26]: https://www.mdpi.com/2504-2289/8/11/153 "Metodologi koleksi IndoGovBERT C1 dan C2"
[27]: https://github.com/Just108/IndoGovBERT "Repository IndoGovBERT"
[28]: https://huggingface.co/datasets/just-108/C2 "Kartu dataset IndoGovBERT C2"

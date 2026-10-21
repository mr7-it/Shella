# Arsitektur Shella

## Identitas dan modul

Paket Cargo, crate library, dan binari utama bernama `shella`. `src/lib.rs` mengekspos API publik; modul implementasi berada di direktori masing-masing:

- `config/`: konfigurasi LIF dan validator domain.
- `error/`: error bertipe untuk konfigurasi, input, numerik, topologi, persistence, dan serialisasi.
- `neuron/` serta `runtime/`: state/transisi LIF, runtime satu neuron, dan clock.
- `learning/`: STDP lokal, eligibility trace, dan reward-modulated rule.
- `network/`: orkestrasi sinkron, state-neuron restore validation, sinaps, delayed event, propagation, scheduler.
- `evaluator/`: konfigurasi reward, metrik, dan penerapan reward transactional.
- `benchmark/`: conditioning, interferensi/replay, continual-learning berkonflik dengan anggaran replay terukur, generalisasi held-out/noise, dan statistik recall.
- `checkpoint/`: schema, integritas/checksum/file writer, save, dan restore.
- `memory/`: schema record, mutasi/store, retrieval, dan persistence/integritas.
- `decision/`: konfigurasi kandidat/kebijakan, seleksi/skoring, alasan abstain terstruktur, dan confusion metrics.
- `identity/`: strong types `SubwordID`, `WordID`, `NeuronID`, dan `SynapseID`.
- `language/`: normalisasi/tokenizer, vocabulary dengan hashbrown SwissTable untuk
  lookup teks, segmentasi subword ber-ID stabil, representasi kata-neuron
  terdistribusi, dan koneksi lexical yang memakai sinaps jaringan aktual.
  `indonesian/` menyediakan lexicon ID berprovenance dengan JSON terversi,
  validasi metadata linguistik, indeks alias, impor batch atomik, serta integrasi
  vocabulary ke jaringan. Submodul `datasets`, `licenses`, `relations`,
  `hunspell`, `nusax`, dan `indosum` memuat parser/policy sumber. Aset
  stopwords-id, PUD, NusaX, dan IndoSum berada terpisah di `data/language/` dan
  diverifikasi dengan hash sebelum digunakan. GSD/CSUI/IndoNLI tidak diaktifkan
  karena lisensi teks atau cakupan izinnya belum terverifikasi. `semantic.rs`
  menghubungkan `IndonesianVocabulary` ke representasi frasa, graph berprovenance,
  parsing role heuristik, resolusi konteks terbatas, retrieval, inference trace,
  dan adapter resonansi LIF.
- `sdk/`: input/output, engine, sesi, dan adapter persistence.
- `engine/`: reasoning engine modular terpisah dari SDK. `inference.rs` mengorkestrasi
  retrieval `LearningMemory` dan langkah LIF; `temporal.rs`, `calibration.rs`,
  `learning.rs`, `checkpoint.rs`, `analysis.rs`, dan `validation.rs` memisahkan
  perhatian temporal, kalibrasi, pembelajaran, persistensi, analisis bukti/budget,
  serta validasi. `mod.rs` menyimpan state dan API orkestrasi dasar,
  perhatian temporal, iterasi jaringan LIF, abstention, dan trace metrik.

## Jalur simulasi

`LifConfig` memvalidasi parameter satuan. `LifNeuron` menyiapkan transisi tanpa mutasi lalu melakukan commit setelah lolos pemeriksaan numerik. `LifRuntime` mengelola satu neuron; `LifNetwork` menyelaraskan neuron dengan konfigurasi heterogen pada satu `dt`. Network memvalidasi arus, mengagregasi event jatuh tempo, menyiapkan seluruh transisi, menjadwalkan event masa depan, menghitung perubahan learning state, lalu meng-commit langkah. Kegagalan yang terdeteksi sebelum commit tidak memajukan state jaringan.

Spike pada langkah selesai `k` menjadwalkan input pada awal langkah `k+d`. Event menyimpan snapshot bobot saat spike sumber dipancarkan, sehingga update bobot sesudah penjadwalan tidak mengubah event pending. Urutan neuron, sinaps, penjumlahan input, dan tie-break aksi deterministik.

## Learning dan evaluasi

STDP lokal mengubah bobot dari pasangan spike pre/post. Pada reward-modulated mode, update lokal masuk ke eligibility trace; reward signed yang diterima kemudian memodulasi bobot. `RewardEvaluator` memetakan target/kanal output aktual ke reward dan baru mencatat metrik setelah operasi yang diperlukan berhasil.

Benchmark conditioning memanggil `LifNetwork`, evaluator, dan reward learning produksi. Jaringan kontrol menerima pola teacher/input yang sama tanpa menerapkan update reward. Tugas kecil ini menunjukkan mekanisme pada pola tetap, bukan bukti generalisasi. `benches/system.rs` mengukur langkah jaringan 16/64/256 neuron + sinaps, serta save/restore checkpoint dan prefix/tag retrieval memori.

## State persisten

`Checkpoint` menyimpan `LifRuntime`, `LifNetwork`, atau `LanguageNetwork` (vocabulary dan ID kata/subword, binding neuron, relasi ID sinaps, bobot, parameter neuron, traces, pending events) serta evaluator opsional. JSON berversi memeriksa ukuran, checksum, dan invariants sebelum pemulihan. Network lama tanpa field identitas dimigrasikan secara deterministik dari urutan populasi/topologi saat restore; ID tersebut disimpan pada save berikutnya.

`MemoryStore` adalah store JSON dengan upsert, delete, get, ID stabil, prefix/tag dan embedding retrieval, version/checksum check, serta write-temp/rename. Advisory exclusive lock pada sidecar path mencegah lebih dari satu instance/proses membuka store yang sama untuk menulis. `SemanticMemory` menyimpan satu snapshot schema-versioned di atas store yang sama untuk mengomit lexicon, graph, phrase, konteks, dan deletion journal secara atomik. Retrieval semantiknya memakai WordID dan relasi eksplisit, bukan embedding `TextEmbedding`; kedua jalur tetap terpisah.

## Keputusan, bahasa, dan aplikasi

`DecisionPolicy` menghitung skor linear dari fitur yang disediakan aplikasi; `selection` menghasilkan kontribusi terurai, margin, alasan, tie-break, atau abstain. Skor bukan probabilitas dan confidence tidak dikalibrasi.

Pipeline teks menormalisasi Unicode lowercase dan tokenisasi karakter alfanumerik. `Vocabulary` menyimpan `WordID` terpisah dari nilai hash; hashbrown hanya menjadi indeks lookup. `LanguageNetwork` memberi tiap kata urutan `SubwordID`, dapat mengikat kata pada banyak `NeuronID`, membentuk sinaps aktual di antara representasi kata, serta memproyeksikan input ke LIF/STDP. `SemanticMemory` menautkan token/frasa ke ID lexicon dan edge dengan provenance; role kalimat, negasi, anafora satu kalimat, retrieval, dan inference dibatasi aturan yang dijelaskan di `docs/BAHASA.md`. `ReasoningEngine::reason_text` memakai semantic currents saat store dipasang; `learn_from_semantic_feedback` mengalirkan analisis, gap, currents, feedback ke `LearningMemory`, dan update readout/STDP pada LIF aktual. Pengalaman neural tetap fitur currents, bukan teks mentah.

Reasoning checkpoint mengikat LearningMemory dan semantic store melalui fingerprint. SDK Engine mempunyai payload integrated checkpoint yang mengikat generic `MemoryStore`, `LearningMemory`, dan `SemanticMemory`; file store tetap terpisah. Deletion journal pada SemanticMemory menghapus experience task terkait dan context frames pada jalur ReasoningEngine, dan checkpoint ditolak bila antrean belum selesai. Perubahan lintas file tidak memiliki transaksi atomic umum; journal membuat penghapusan dapat di-retry. Checkpoint SDK legacy tidak berubah dan tetap tidak menyertakan konfigurasi encoder/policy. `TextEncoder` lama tetap tersedia dengan proyeksi modulo agar API kompatibel. Tidak ada model bahasa, decoder/generator, atau bukti generalisasi luas.

`IndonesianVocabulary` memvalidasi entri JSON dengan schema version, ID kata
append-only, provenance sumber/lisensi per entri, metadata linguistik opsional,
dan alias ejaan NFC. Lookup alias menggunakan hashbrown; ID tidak berasal dari
hash. Checkpoint umum menyediakan checksum dan atomic replace untuk lexicon.
Validasi struktur metadata tidak mengesahkan sumber/licensing secara eksternal.
Dataset pihak ketiga yang disetujui disimpan terpisah dari kode di
`data/language/` dan divalidasi melalui registry; dataset lain tidak diaktifkan
tanpa izin yang jelas. Segmentasi kata tak terdaftar hanya bersifat ortografis.

`Engine`/`Session` mengirim input arus/teks ke network, menerima spike aktual, menghubungkan spike neuron ke kanal aksi, memanggil decision/evaluator, menulis memory, dan menyimpan/memulihkan checkpoint. `with_semantic_memory` memasang facade semantic; `with_learning_memory` memasang neural experience store. `analyze_semantics` membaca lexicon, `analyze_semantics_in_context` menyimpan konteks persisten, dan `resonate_semantics` mengorkestrasi graph semantic dengan `LanguageNetwork` ber-ID cocok. `save_integrated_checkpoint`/`restore_integrated_checkpoint` memeriksa fingerprint seluruh store sebelum mutasi neural state. SDK sinkron dalam proses; store eksternal dibuka terpisah sebelum restore.

`ReasoningEngine` memiliki `LifNetwork` dan `LearningMemory`. Setiap panggilan
`reason` mengambil pengalaman sesuai context, menggabungkan retrieval score dengan
timestamp Unix nanodetik pengalaman untuk bobot temporal, menginjeksikan dukungan aksi ke arus LIF, lalu
menjalankan jumlah langkah yang dikonfigurasi. Hasil menyertakan langkah jaringan,
recall, perhatian, dukungan aksi, entropy, evidence strength, dan confidence heuristik.
Skor heuristik bukan probabilitas. `fit_confidence_calibration` melatih Platt hanya
dengan validation samples dan menghitung Brier/MAE pada evaluation samples terpisah;
hasil kalibrasi diakses terpisah dari skor mentah. Aksi ditahan bila tidak ada
dukungan atau confidence keputusan di bawah ambang. `learn_from_feedback` menulis
feedback ke memori; bila STDP aktif, ia menyajikan input dan spike pengajar ke
jaringan lalu mengirim signed reward pada mode reward-modulated STDP. Tanpa STDP,
readout sinaps yang ada tetap dipakai.

Context window opsional membatasi riwayat transisi per context ID; compute budget
adaptif memilih 1..=configured maximum dari densitas input, distribusi recall, dan
kontradiksi. Goal-directed retrieval menggabungkan input dan fitur goal hanya untuk
query, tidak mengubah input arus LIF. Contradiction detection membandingkan label
pada pengalaman recall yang similarity-nya tinggi. Causal association API hanya
mengagregasi ko-occurence fitur-label, bukan bukti kausal intervensional.

Consolidation menggabungkan record mirip dalam task dan label sama, menghitung rerata
berbobot evidence count, dan menghapus duplikat. Time decay memakai umur sejak decay
terakhir; usage count memperpanjang effective half-life. Experience replay memakai
API replay `LearningMemory` yang menjalankan input pengajar melalui network dan
reward-modulated STDP bila aktif. Reasoning checkpoint memuat jaringan, konfigurasi,
context dan kalibrator/provenance secara checksummed; fingerprint record
`LearningMemory` mencegah restore terhadap versi store berbeda. File checkpoint dan
file memory bukan satu transaksi filesystem, sehingga mismatch menghasilkan error
alih-alih pemulihan parsial.

## Batas operasional

Tidak tersedia transport HTTP/RPC, server, multi-writer lock, migrasi schema umum,
default calibrated confidence tanpa validation fit, enkripsi file, atau model bahasa.
Checkpoint SDK menyimpan state network/evaluator, bukan konfigurasi encoder/policy
statis; aplikasi perlu memasangnya kembali. Memory mempunyai file persisten tersendiri.

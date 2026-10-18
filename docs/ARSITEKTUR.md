# Arsitektur Shella

## Identitas dan modul

Paket Cargo, crate library, dan binari utama bernama `shella`. `src/lib.rs` mengekspos API publik; modul implementasi berada di direktori masing-masing:

- `config/`: konfigurasi LIF dan validator domain.
- `error/`: error bertipe untuk konfigurasi, input, numerik, topologi, persistence, dan serialisasi.
- `neuron/` serta `runtime/`: state/transisi LIF, runtime satu neuron, dan clock.
- `learning/`: STDP lokal, eligibility trace, dan reward-modulated rule.
- `network/`: orkestrasi sinkron, state-neuron restore validation, sinaps, delayed event, propagation, scheduler.
- `evaluator/`: konfigurasi reward, metrik, dan penerapan reward transactional.
- `benchmark/`: protokol conditioning dan statistik recall.
- `checkpoint/`: schema, integritas/checksum/file writer, save, dan restore.
- `memory/`: schema record, mutasi/store, retrieval, dan persistence/integritas.
- `decision/`: konfigurasi kandidat/kebijakan serta seleksi/skoring.
- `language/`: normalisasi, tokenizer, vocabulary, dan encoder arus.
- `sdk/`: input/output, engine, sesi, dan adapter persistence.

## Jalur simulasi

`LifConfig` memvalidasi parameter satuan. `LifNeuron` menyiapkan transisi tanpa mutasi lalu melakukan commit setelah lolos pemeriksaan numerik. `LifRuntime` mengelola satu neuron; `LifNetwork` menyelaraskan neuron dengan konfigurasi heterogen pada satu `dt`. Network memvalidasi arus, mengagregasi event jatuh tempo, menyiapkan seluruh transisi, menjadwalkan event masa depan, menghitung perubahan learning state, lalu meng-commit langkah. Kegagalan yang terdeteksi sebelum commit tidak memajukan state jaringan.

Spike pada langkah selesai `k` menjadwalkan input pada awal langkah `k+d`. Event menyimpan snapshot bobot saat spike sumber dipancarkan, sehingga update bobot sesudah penjadwalan tidak mengubah event pending. Urutan neuron, sinaps, penjumlahan input, dan tie-break aksi deterministik.

## Learning dan evaluasi

STDP lokal mengubah bobot dari pasangan spike pre/post. Pada reward-modulated mode, update lokal masuk ke eligibility trace; reward signed yang diterima kemudian memodulasi bobot. `RewardEvaluator` memetakan target/kanal output aktual ke reward dan baru mencatat metrik setelah operasi yang diperlukan berhasil.

Benchmark conditioning memanggil `LifNetwork`, evaluator, dan reward learning produksi. Jaringan kontrol menerima pola teacher/input yang sama tanpa menerapkan update reward. Tugas kecil ini menunjukkan mekanisme pada pola tetap, bukan bukti generalisasi. `benches/system.rs` mengukur langkah jaringan 16/64/256 neuron + sinaps, serta save/restore checkpoint dan prefix/tag retrieval memori.

## State persisten

`Checkpoint` menyimpan `LifRuntime` atau seluruh `LifNetwork` (parameter neuron, membran/refraktori, spike/clock, topologi/bobot, traces, pending events) dan evaluator opsional. JSON berversi memeriksa ukuran, checksum, dan invariants sebelum pemulihan.

`MemoryStore` adalah store JSON terpisah dengan upsert, delete, get, ID stabil, prefix/tag retrieval, version/checksum check, dan write-temp/rename. Format menyimpan record JSON; tidak ada embedding atau associative retrieval semantik.

## Keputusan, bahasa, dan aplikasi

`DecisionPolicy` menghitung skor linear dari fitur yang disediakan aplikasi; `selection` menghasilkan kontribusi terurai, margin, alasan, tie-break, atau abstain. Skor bukan probabilitas dan confidence tidak dikalibrasi.

Pipeline teks menormalisasi Unicode lowercase, tokenisasi karakter alfanumerik, mempelajari vocabulary dari korpus input, lalu mengubah ID token menjadi arus menggunakan proyeksi modulo indeks neuron. Tidak ada model bahasa, decoder/generator, pembelajaran semantik, atau metrik generalisasi teks.

`Engine`/`Session` mengirim input arus/teks ke network, menerima spike aktual, menghubungkan spike neuron ke kanal aksi, memanggil decision/evaluator, menulis memory, dan menyimpan/memulihkan checkpoint. SDK sinkron dalam proses.

## Batas operasional

Tidak tersedia transport HTTP/RPC, server, multi-writer lock, migrasi schema otomatis, confidence probabilistik, enkripsi file, atau model bahasa. Checkpoint SDK menyimpan state network/evaluator, bukan konfigurasi encoder/policy statis; aplikasi perlu memasangnya kembali. Memory mempunyai file persisten tersendiri.

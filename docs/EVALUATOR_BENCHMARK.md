# Evaluator reward otomatis dan benchmark

## Evaluator

`RewardEvaluatorConfig::new(reward_benar, penalti_salah)` menerima dua nilai finite positif. `RewardEvaluator::evaluate(target, observed)` memilih aksi dari kanal boolean yang teramati; jika beberapa kanal aktif, tie-break adalah indeks aktif terendah. Target `Some(i)` berarti kanal aksi `i` diharapkan, sedangkan `None` berarti tidak boleh ada kanal spike.

Respons cocok menghasilkan `+correct_reward`; respons tidak cocok menghasilkan `-incorrect_penalty`. `RewardMetrics` mencatat jumlah sampel/benar/no-response, akurasi, total reward, serta mean reward. `evaluate_and_apply` menerapkan reward pada eligibility `LifNetwork` sebelum commit metrik; jika network gagal, evaluator tidak mencatat sampel parsial.

SDK menambahkan adapter `Session::evaluate_actions`/`evaluate_actions_and_apply`: daftar neuron output dipetakan ke kanal aksi sesuai urutan slice, lalu dinilai berdasarkan spike dalam output langkah tersebut. Test integrasi `tests/sdk_integration.rs` memeriksa reward dari spike dan rollback metrik saat apply ditolak.

## Conditioning reward-learning

`ConditioningBenchmarkConfig::standard()` menggunakan 32 epoch dan delapan pengulangan uji per pola. Tugas mempunyai dua neuron cue, satu neuron aksi, dan dua sinaps dengan bobot awal nol.

- Cue “mendekat” dipasangkan dengan teacher spike dan target aksi kanal 0.
- Cue “menahan” mendapat teacher tanpa spike output; evaluator menargetkan diam.
- Kontrol menerima urutan cue/teacher yang sama tetapi tidak menerapkan reward ke bobot.
- Kedua jaringan menjalani washout 50 langkah, kemudian pola yang sama diuji tanpa teacher.

Putaran Release terakhir mengukur kontrol **8/16 (50%)** dan jaringan learned **16/16 (100%)**, selisih **+50 poin persentase**; bobot cue mendekat berakhir 116.155578 nA dan cue menahan 0 nA. Harness melakukan tujuh batch × 100 run; median waktu per eksperimen **230 128 ns** (≈230 µs) pada mesin sesi ini. Ini adalah recall dua pola yang dilatih, bukan generalisasi atau benchmark kecerdasan.

## Benchmark skala, persistence, retrieval

`benches/system.rs` mengukur median lima batch, tanpa RNG. Tiap batch menjalankan 500 langkah jaringan; checkpoint memakai 20 operasi/batch; memory retrieval menyiapkan 256 record lebih dahulu lalu mengukur 500 query/batch. Benchmark system tidak menjalankan fase warm-up terpisah. Hasil Release terakhir:

| Operasi | Skala | Median latency/op | Throughput |
|---|---:|---:|---:|
| Langkah network + delayed event | 16 neuron / 16 sinaps | 2,386 ns | 419,142 langkah/s |
| Langkah network + delayed event | 64 neuron / 64 sinaps | 8,393 ns | 119,154 langkah/s |
| Langkah network + delayed event | 256 neuron / 256 sinaps | 22,958 ns | 43,558 langkah/s |
| Checkpoint save | 64 neuron / 64 sinaps | 762,385 ns | 1,312 save/s |
| Checkpoint restore | 64 neuron / 64 sinaps | 265,567 ns | 3,766 restore/s |
| Memory prefix search | 256 record | 1,901 ns | 526,057 query/s |
| Memory tag search | 256 record | 2,620 ns | 381,646 query/s |

Checkpoint contoh berukuran 25,985 byte. Seluruh waktu sensitif terhadap beban host/filesystem; angka mesin sesi ini bukan jaminan kapasitas produksi. Benchmark network ring membangkitkan spike dan satu delayed event per neuron/step, bukan variasi topologi arbitrer. Benchmark tidak mengukur peak memory.

## Menjalankan ulang

```sh
cargo test --test milestone4 --locked
cargo bench --bench learning --locked
cargo bench --bench system --locked
```

Hasil merupakan output yang diukur dari kode aktual. Ulangi perintah pada target hardware/beban kerja yang relevan untuk evaluasi lokal.

## Semantic feedback group holdout

Tes `benchmark_polaritas_100_test_group_held_out_membandingkan_baseline_dan_feedback`
memisahkan 50 train, 50 validation, dan 100 test menurut pasangan semantik
subjek-objek. Test nouns tidak ada dalam vocabulary yang dibangun dari train.
Sebelum tiap label train diberikan, jaringan membuat prediksi aktual, lalu
feedback target diterapkan secara incremental pada LIF/readout dan pengalaman
disimpan ke `LearningMemory`. Validation tidak memberi feedback. Baseline dan
hasil setelah 50 feedback diukur pada 100 kalimat test yang identik;
tiap inferensi memakai clone neural state dan snapshot LearningMemory yang segar
agar penghitung usage tidak mengubah prediksi berikutnya. Pengukuran aktual pada
sesi ini: baseline **0/100**, validation pascabelajar **50/50**, dan test
pascabelajar **100/100**.

Ini benchmark sintetik, bukan korpus berlisensi: labelnya adalah negasi eksplisit
(`tidak`), struktur template tetap, dan pasangan noun test baru. Hasilnya
mengisolasi transfer fitur polaritas pada task sempit; tidak mengukur variasi
sintaksis luas, paraphrase alami, atau generalisasi bahasa secara umum. Data test
tidak dipakai untuk pembelajaran atau penyesuaian parameter.


## Continual learning Milestone 6

Runner `examples/milestone6_continual.rs` menjalankan empat asosiasi dalam dua fase. Setiap pola input memuat cue umum yang dipakai semua kelas dan satu cue pembeda. Jaringan memiliki 9 neuron dan 20 sinaps, learning rate 2.5, bobot maksimum cue umum 50 nA dan cue pembeda 150 nA. Tiap contoh diberi teacher spike, dinilai dengan evaluator, lalu diikuti 100 langkah tanpa input agar eligibility dari contoh terdahulu meluruh.

Training memakai cue 100 nA, 32 epoch per fase, dan 128 contoh per jaringan. Validasi memakai amplitudo 80/90/110/120 nA; test final memakai 97/103 nA. Setiap contoh evaluasi memakai clone state awal yang sama dan tidak menerima teacher/reward. Lima seed tetap `[1, 7, 19, 42, 99]` hanya mengacak urutan asosiasi secara deterministik; bobot awal, kelas, serta amplitudo tidak diacak.

Hasil final pada kelima seed: baseline test **0/8 (0%)**, test akhir **8/8 (100%)** untuk empat asosiasi × dua amplitudo, validasi **16/16 (100%)** untuk empat asosiasi × empat amplitudo, gain **+100 poin persentase**, reward teacher-forced **128** untuk 128 contoh, task A **100%** setelah fase pertama dan tetap **100%** setelah fase kedua, sehingga forgetting terukur **0 poin persentase**. CSV per seed: `results/milestone6_continual_2026-10-19.csv`.

Tugas ini sintetis, deterministik, dan kecil. Split hanya menahan amplitudo cue, bukan kelas/struktur semantik yang berbeda. Seed hanya mengubah urutan training; bukan variasi inisialisasi, data, atau noise. Forgetting nol berlaku untuk empat pola dengan cue pembeda dan bobot cue umum yang dibatasi; hasil ini tidak membuktikan pencegahan catastrophic forgetting untuk pemetaan konflik atau distribusi lain. `training_reward` teacher-forced tidak menggantikan recall tanpa teacher.

Jalankan ulang dengan `cargo run --example milestone6_continual --locked`. Hasil tersebut merupakan metrik perilaku jaringan, bukan klaim pemahaman bahasa atau benchmark kecerdasan.

## Generalisasi komposisional dan noise Milestone 8

`GeneralizationBenchmarkConfig::standard()` menjalankan 8 inisialisasi bobot non-nol × 4 pemetaan fitur-ke-kelas, atau 32 run paired. Setiap jaringan melatih empat fitur satu per satu dengan teacher dan reward-modulated STDP. Evaluasi positif memakai pasangan fitur sekelas; kombinasi pasangan ini tidak pernah disajikan sebagai pola training. Amplitudo train berbeda per variasi (90/100/110/120 nA), sedangkan evaluasi memakai amplitudo ±10 nA dari nilai train. Tiap tingkat noise menguji 24 sampel positif dan 12 pola kosong, dengan urutan noise deterministik yang sama untuk baseline dan jaringan terlatih.

Noise seragam signed ditambahkan pada kanal fitur lalu arus dipotong minimum nol. Jadi noise pada fitur tidak aktif menjadi arus gangguan nonnegatif; ini bukan white noise temporal atau noise pada bobot. Pola kosong dengan noise mengukur false positive; pola gabungan held-out mengukur false negative, false positive multiclass, dan abstain (tidak ada neuron aksi yang spike).

Pengukuran standar aktual (akurasi baseline/terlatih, lalu FPR/FNR/abstain jaringan terlatih):

| Noise (nA) | Baseline accuracy | Trained accuracy | FPR | FNR | Abstain |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.333 | 1.000 | 0.000 | 0.000 | 0.333 |
| 10 | 0.333 | 1.000 | 0.000 | 0.000 | 0.333 |
| 20 | 0.333 | 1.000 | 0.000 | 0.000 | 0.333 |
| 40 | 0.333 | 0.997 | 0.000 | 0.005 | 0.336 |
| 60 | 0.333 | 0.873 | 0.229 | 0.076 | 0.269 |

Jalankan dengan `cargo run --example generalization --locked`; regression test ada di `tests/generalization.rs`. Hasil mendukung generalisasi ke kombinasi fitur sekelas yang belum dilatih untuk tugas sintetis ini. Empat pemetaan dan delapan seed tetap deterministik, bukan sampel populasi acak. Noise dibangkitkan perangkat PRNG lokal, tidak memodelkan sensor/lingkungan tertentu. Keberhasilan pada pola komposisional ini tidak menunjukkan generalisasi bahasa atau tugas dunia nyata.

## Continual learning berkonflik Milestone 9

`ContinualReplayBenchmarkConfig::standard()` memasangkan 8 inisialisasi bobot × 4 pemetaan Task A, total 32 run. Empat fitur dilatih ke dua kelas pada Task A; Task B membalik label setiap fitur yang sama, dengan context neuron berbeda sebagai sinyal task. Jaringan memakai hidden LIF conjunction unit yang shared untuk fitur dan unit konteks task; hidden-to-action connections memakai STDP reward-modulated. Buffer replay berisi contoh Task A aktual, termasuk contoh diam context-only.

Semua metode memakai 64 epoch Task A dan anggaran fase B setara: kontrol menyajikan Task B tiga kali; replay menyajikan Task B dua kali dan memutar Task A satu kali. Standard budget **1.280 presentasi / 29.440 langkah jaringan per metode dan run**. Bobot awal non-nol diturunkan dari seed; tiap pasangan metode dimulai dari salinan bobot/state yang sama. Validation memakai amplitudo train ±5 nA dan noise 10 nA. Setelah kalibrasi hanya pada validation, test final dibekukan pada amplitudo train ±17 nA dan noise 30/50/70 nA; setiap task memiliki 32 positif dan 40 negatif (context-only + feature-only). Validation dan test tidak digunakan sebagai data teacher. Noise stream dipasangkan antarmetode.

Hasil release aktual dirata-ratakan atas 32 run (8 seed × 4 task mapping). Metrik test di bawah memakai split final noise 30 nA; FPR dihitung pada pola negatif, FNR pada pola positif, dan abstain pada semua sampel.

| Metode/tahap | Task | Accuracy | FP/run | FN/run | FPR | FNR | Abstain |
|---|---|---:|---:|---:|---:|---:|---:|
| Sebelum belajar | A | 0.556 | 0.00 | 32.00 | 0.000 | 1.000 | 1.000 |
| Sebelum belajar | B | 0.556 | 0.00 | 32.00 | 0.000 | 1.000 | 1.000 |
| Setelah Task A | A | 0.743 | 16.38 | 8.38 | 0.253 | 0.262 | 0.444 |
| Setelah Task A | B | 0.551 | 21.22 | 28.19 | 0.103 | 0.881 | 0.652 |
| Tanpa replay | A | 0.597 | 27.31 | 18.88 | 0.253 | 0.590 | 0.438 |
| Tanpa replay | B | 0.653 | 13.66 | 15.75 | 0.230 | 0.492 | 0.585 |
| Replay 1/3 budget | A | 0.633 | 22.91 | 18.97 | 0.187 | 0.593 | 0.501 |
| Replay 1/3 budget | B | 0.629 | 25.44 | 16.25 | 0.262 | 0.508 | 0.428 |

Forgetting Task A rata-rata pada test **0.146 tanpa replay** dan **0.111 dengan replay** di noise 30 nA; simpangan baku antar-run **0.074** dan **0.073**. Pada noise 50 nA forgetting **0.082 vs 0.066**; pada noise 70 nA **0.049 vs 0.037**. Catastrophic-forgetting rate, didefinisikan sebagai penurunan Task A ≥0.5, adalah **0/32** pada kedua metode dan semua tingkat noise. Validation menunjukkan Task A **0.625 tanpa replay vs 0.708 dengan replay**, sedangkan Task B **0.691 vs 0.681**. Jadi replay memberikan perbaikan retensi parsial, tetapi Task A tetap turun dari 0.743 setelah Task A menjadi 0.633 pada test; Task B juga sedikit turun. Pada noise tinggi performa kedua metode memburuk, dan forgetting yang tampak lebih kecil tidak berarti robustness meningkat. Ini bukan pencegahan forgetting sempurna.

Runner mengeluarkan metrik per seed × varian × split × task × noise: `cargo run --release --example milestone9_continual --locked`. Regression test memeriksa penguasaan Task A sebelum fase B, budget seimbang, split sample, fingerprint unik, dan peningkatan rerata validation Task A serta dua-task replay. Empat mapping dan delapan seed adalah protokol deterministik kecil; pola dan noise sintetis, sehingga hasil tidak membuktikan continual learning umum.

## Interferensi tugas dan replay

`InterferenceBenchmarkConfig::standard()` menjalankan 8 inisialisasi berbobot deterministik × 4 variasi tugas (32 pasangan). Bobot awal dipilih dari rentang non-nol berdasarkan seed; variasi mengubah konteks cue, label, dan amplitudo train (90/100/110/120 nA), sedangkan evaluasi memakai amplitudo berdekatan yang tidak dipakai saat training. Setiap pasangan melatih tugas lama, mengukur recall, lalu membandingkan dua salinan state yang sama: kontrol belajar tugas baru tanpa replay; perlakuan menyisipkan satu contoh tugas lama sesudah setiap contoh baru. Kedua jalur memakai jumlah langkah simulasi sama.

Hasil standar yang diukur: akurasi tugas lama sebelum fase kedua **100%**; setelah fase kedua rata-rata **50% tanpa replay** dan **100% dengan replay**. Forgetting rata-rata turun dari **0,5 ke 0,0**. Akurasi tugas baru rata-rata **50% pada kedua jalur**: separuh variasi mempunyai konteks berbeda dan kedua task tercapai 100%; separuh lainnya memakai cue yang identik dengan label yang berlawanan, sehingga pemetaan statis tidak dapat memenuhi kedua target sekaligus. Pada konflik cue-identik, replay mempertahankan asosiasi lama dan tugas baru gagal; tanpa replay tugas baru juga tidak tercapai konsisten dan asosiasi lama tergeser. Hasil ini menunjukkan tradeoff serta batas representasi, bukan bahwa replay menyelesaikan konflik yang tidak dapat dibedakan.

Inisialisasi menggunakan PRNG deterministik lokal, bukan sumber RNG eksternal; variasi tugas mengubah struktur input/target tetapi tetap tugas sintetis kecil. Jalankan tabel per-run dengan `cargo run --example interference --locked`, atau tes reproducibility dengan `cargo test --test continual_learning --locked`. Ini bukan bukti generalisasi ke dataset atau tugas eksternal.

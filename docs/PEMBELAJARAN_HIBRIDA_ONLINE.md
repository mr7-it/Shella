# Pembelajaran feedback hybrid LIF–Transformer

## Alur runtime

Shella mempertahankan dua mekanisme keputusan yang berbeda. `ReasoningEngine` memakai jaringan LIF yang sudah ada untuk memutuskan apakah pertanyaan dan bukti boleh masuk ke tahap penyusunan. `GenerativeLanguageModel` menyusun draf secara autoregresif. Setelah draf dibuat, `answer_with_neural_review` meminta LIF menilai kandidat, menjalankan pemeriksaan istilah/polaritas konservatif, dan—bila head LIF terdiferensiasi sudah dilatih—meminta head itu memilih `Answer`, `Revise`, atau `Abstain`. Jawaban hanya dilepas bila gate LIF lama, head hybrid (jika tersedia), dan verifier semuanya mengizinkan. Revisi dibatasi paling banyak satu kali.

Head trainable merupakan recurrent LIF readout 16 neuron di atas hidden states Transformer. Forward menggunakan spike biner; backward menggunakan fast-sigmoid surrogate gradient. Leak, threshold, dan reset memakai parameter tetap, dalam unit ternormalisasi. Implementasi ini bukan simulasi neuron biologis lengkap dan belum memiliki refractory dynamics. Ia melengkapi `ReasoningEngine`; ia tidak menggantikan SemanticMemory, WordID, LearningMemory, atau sistem LIF simbolik yang sudah ada.

`SemanticMemory` tetap menjadi sumber bukti eksplisit pada `answer_with_neural_review`. Korpus pretraining tidak otomatis berubah menjadi fakta yang terindeks. Pemeriksa jawaban saat ini adalah pemeriksaan leksikal dan polaritas yang konservatif, bukan entailment umum atau pemeriksaan fakta eksternal. Data feedback harus berasal dari manusia yang benar-benar mengonfirmasi atau evaluator tepercaya yang independen; jawaban yang belum diperiksa tidak boleh dilabeli terverifikasi hanya karena dibuat model.

## Memberi feedback dan menyimpan hasil

Contoh berikut menunjukkan bentuk data publik yang diterima SDK. `prompt` perlu memuat pertanyaan beserta bukti yang tersedia; `candidate_response` adalah draf yang ditinjau. `target_response` adalah jawaban acuan yang disetujui atau koreksi manusia. Karakter pada target harus tersedia di tokenizer karakter model.

```rust
use shella::{
    FeedbackAuthority, HybridAction, HybridTrainingConfig, VerifiedActionExample,
};

let examples = vec![VerifiedActionExample {
    source_id: "review/2026-10-23/0001".into(),
    prompt: "pertanyaan apakah kucing sebesar rumah bukti kucing dewasa berukuran jauh lebih kecil dari rumah".into(),
    candidate_response: Some("Tidak. Kucing jauh lebih kecil daripada rumah.".into()),
    target_response: Some("Tidak. Kucing jauh lebih kecil daripada rumah manusia.".into()),
    expected_action: HybridAction::Answer,
    authority: FeedbackAuthority::HumanConfirmed,
}];

let mut session = engine.session();
let outcome = session.learn_from_verified_feedback(
    &mut model,
    &examples,
    HybridTrainingConfig::default(),
    "target/shella-hybrid-checkpoint.sllm",
)?;
```

`engine` harus merupakan `Engine` aktual dan `model` harus sudah dibuat dengan `GenerativeLanguageModel`; untuk menyimpan pengalaman simbolik, pasang juga `LearningMemory` melalui `Engine::with_learning_memory`. Contoh Rust di atas memerlukan owner `Session` yang dapat dipinjam secara mutable; di kode aplikasi pola umumnya adalah `let mut session = engine.session();` lalu memanggil method pada `session`.

Untuk label `Revise`, sertakan draf yang salah pada `candidate_response` dan koreksi yang ditinjau pada `target_response`. Untuk `Abstain`, `target_response` harus `None`; contoh itu melatih keputusan abstain tanpa next-character target. `source_id` wajib unik di dalam satu batch pemanggilan. `FeedbackAuthority::TrustedEvaluator` hanya layak dipakai bila label benar-benar dikeluarkan evaluator independen, bukan skor buatan model yang sama.

`Session::learn_from_verified_feedback` melatih model, menambahkan record berfitur teks ke `LearningMemory` jika memori itu terpasang, dan menyimpan checkpoint model serta checksum SHA-256. LearningMemory mencatat action target yang disetujui; tidak menyimpan teks percakapan mentah sebagai fakta SemanticMemory. Penyimpanan checkpoint dan record memori merupakan operasi terpisah, bukan satu transaksi atomik.

## Apa yang dioptimalkan

Satu optimizer Candle AdamW mengoptimalkan dua objective pada bobot yang sama: cross-entropy next-character untuk target jawaban dan cross-entropy tiga kelas untuk aksi LIF. Backward spike memakai turunan fast-sigmoid; forward tetap bernilai 0/1. Gradient clipping diterapkan sebelum update. `HybridTrainingReport` berisi mean language/action loss, akurasi aksi pada contoh training, sumber feedback, jumlah authority, dan jumlah langkah optimizer.

Akurasi yang dilaporkan adalah akurasi **training**, bukan validasi atau test. Pemanggil harus menyimpan holdout terpisah berdasarkan percakapan/sumber, mengevaluasi sebelum deployment, dan tidak menggunakan contoh test untuk update. Tokenizer karakter belum dapat diperluas dinamis; target dengan karakter OOV ditolak sebelum optimizer menjalankan update.

Perbaikan autograd penting: fused `LayerNorm::forward` pada Candle 0.8.4 memakai operasi tanpa backward. Decoder sekarang memanggil ekspresi `layer_norm_slow` yang terdiferensiasi agar loss bahasa maupun loss aksi dapat mencapai bobot attention/feed-forward dan embedding, bukan hanya output projection atau head LIF. Tes regresi memeriksa gradien nonnol pada bobot Transformer dari kedua loss.

## Batas dan operasi

Checkpoint saat ini menyimpan bobot dan report, tetapi tidak menyimpan state momentum AdamW (`optimizer_state_saved` tetap `false`). Setiap pemanggilan training membuat optimizer baru, sehingga momentum tidak berlanjut antarpanggilan walaupun checkpoint bobot bisa dimuat ulang.

Training hanya terjadi ketika aplikasi secara eksplisit memanggil `learn_from_verified_feedback` dengan feedback yang sudah ditinjau. Tidak ada background self-training, auto-labeling, atau update bobot dari interaksi biasa. Ini mencegah jawaban model yang salah berubah menjadi target hanya karena pernah dihasilkan. Aplikasi pemanggil bertanggung jawab menjaga otoritas label, akses checkpoint, dan evaluasi holdout.

Model tetap CPU-only, memakai tokenizer karakter, dengan batas arsitektur maksimum 4 juta parameter dan context maksimum 256. Target 2 miliar parameter belum diimplementasikan. BPE/subword, optimizer-state resume, streaming korpus besar, GPU backend, serta evaluasi factuality masih pekerjaan terpisah. Jangan mengklaim pretraining korpus penuh selesai sebelum ada run, checkpoint, dan evaluasi yang tercatat.

# Decision engine

`DecisionPolicy::new(weights, minimum_score, minimum_margin)` membangun skor linear deterministic. Fitur kandidat harus dikenal dalam peta bobot dan bernilai finite. Untuk kandidat `c`, skor adalah `Σ weight(feature) × value(feature)`. Fitur yang tidak diberikan dianggap tidak memberi kontribusi (nol). Kandidat dengan skor terbaik menang; seri diurutkan oleh ID leksikografis.

Jika skor terbaik di bawah ambang atau margin terhadap runner-up lebih kecil dari batas, engine abstain (`selected_id() == None`). Hasil mencakup kontribusi setiap fitur kandidat terbaik, skor runner-up, margin, dan alasan pemilihan/abstain. Skor serta margin adalah besaran kebijakan, **bukan probabilitas, confidence, atau ukuran ketidakpastian yang dikalibrasi**. Engine tidak menetapkan fitur secara otomatis dari teks atau spike; aplikasi menyusun kandidat dan fitur berdasarkan kebutuhannya.


Selisih skor runner-up diperiksa setelah pengurangan; jika kedua skor finite tetapi selisihnya overflow, `select` mengembalikan `SimError::NumericalOverflow`. Satu kandidat tetap memakai `+∞` sebagai sentinel margin tanpa runner-up. Ini tetap bukan confidence yang dikalibrasi.

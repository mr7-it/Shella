# Pipeline teks

Pipeline bahasa hanya preprocessing dan representasi sederhana. `normalize_text` melakukan lowercase Unicode dan merapatkan whitespace. `tokenize` menjadikan run karakter Unicode alfanumerik sebagai token dan tanda baca sebagai separator. `Vocabulary::fit(corpus, min_frequency, max_tokens)` membangun vocabulary dari korpus yang diberikan; urutan token adalah frekuensi menurun lalu leksikografis, dengan `<unk>` pada ID 0. `Vocabulary::encode` memetakan kata di luar kamus ke `<unk>`.

`TextEncoder::encode(text, neuron_count, current_per_token_na)` mengembalikan token, ID, dan vektor arus. Setiap occurrence menambah arus positif pada neuron `token_id % neuron_count`. Ini bag-of-token deterministic dengan collision; tidak merepresentasikan tata bahasa, makna, konteks, embedding, subword, klasifikasi terlatih, atau generasi teks. Vocabulary harus di-fit pada data latih oleh pemanggil sebelum dipakai pada data evaluasi; jangan fit ulang dari data uji bila ingin evaluasi tanpa kebocoran.


`TextEncoder::encode` menolak input yang setelah normalisasi/tokenisasi tidak menghasilkan token (termasuk teks kosong, whitespace-only, dan punctuation-only). `tokenize` tetap dapat dipakai sebagai tokenizer umum dan mengembalikan daftar kosong untuk input seperti itu. Token OOV tetap dipetakan ke ID `<unk>` 0; encoder belum menyediakan rasio OOV atau keputusan unknown.

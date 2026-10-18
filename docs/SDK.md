# SDK Rust

`Engine::new(network)` memiliki jaringan simulasi dan dapat dipasangi `TextEncoder`, `DecisionPolicy`, serta `MemoryStore` melalui builder. `Engine::session()` meminjam engine secara eksklusif. `Session::step(SdkInput::Currents(...))` atau `Session::step(SdkInput::Text { ... })` menjalankan satu langkah jaringan aktual dan mengembalikan `SdkOutput` berisi `NetworkStep`, daftar neuron spike, jenis input serta metadata teks bila ada.

`Session::evaluate_actions(output, target, action_neurons, evaluator)` memetakan neuron output ke kanal aksi yang berurutan sesuai slice lalu memakai evaluator. `evaluate_actions_and_apply` juga menerapkan reward pada jaringan; mode reward-modulated harus telah diaktifkan agar panggilan berhasil. `Session::decide(candidates)` menjalankan policy terpasang. `remember` menulis ke store persisten yang terpasang. API mengembalikan `SimError` bertipe bila subsistem opsional belum dikonfigurasi atau masukan tidak sah.

`save_checkpoint` dan `restore_checkpoint` menyimpan/memulihkan jaringan dan evaluator. Untuk resume lengkap aplikasi, pasang kembali vocabulary/encoder dan decision policy statis saat membangun engine; MemoryStore mempunyai file persisten terpisah. SDK saat ini sinkron dalam proses dan tidak menyediakan server HTTP, RPC, background service, cancellation token, atau autentikasi.

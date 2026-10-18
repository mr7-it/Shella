use shella::{SimError, TextEncoder, Vocabulary, normalize_text, tokenize};

#[test]
fn normalisasi_dan_tokenisasi_aturannya_transparan() {
    assert_eq!(
        normalize_text("  SAYA   SUKA, KUCING!\n").expect("teks valid"),
        "saya suka, kucing!"
    );
    assert_eq!(
        tokenize("Makan nasi, lalu makan!").expect("tokenisasi"),
        ["makan", "nasi", "lalu", "makan"]
    );
    assert_eq!(
        tokenize("猫 dan café").expect("unicode"),
        ["猫", "dan", "café"]
    );
    assert!(matches!(
        normalize_text("a\0b"),
        Err(SimError::InvalidInput { .. })
    ));
}

#[test]
fn vocabulary_dilatih_deterministik_unknown_dan_train_eval_terpisah() {
    let training = ["kucing makan ikan", "anjing makan daging", "kucing tidur"];
    let vocabulary = Vocabulary::fit(training, 1, 16).expect("fit korpus train");
    let second = Vocabulary::fit(training, 1, 16).expect("fit yang sama");
    assert_eq!(vocabulary, second);
    assert_eq!(
        vocabulary.id("kucing"),
        1,
        "seri frekuensi diputus leksikografis"
    );
    assert_eq!(vocabulary.id("makan"), 2);
    assert_eq!(vocabulary.id("burung"), 0);
    assert_eq!(vocabulary.token(0), Some("<unk>"));
    let held_out = vocabulary.encode("burung makan").expect("encode eval");
    assert_eq!(held_out[0], 0);
    assert_eq!(held_out[1], vocabulary.id("makan"));
    assert!(Vocabulary::fit(["data"], 0, 4).is_err());
}

#[test]
fn encoder_menghasilkan_currents_deterministik_dan_validasi_input() {
    let vocabulary = Vocabulary::fit(["alpha beta beta gamma"], 1, 8).expect("fit");
    let encoder = TextEncoder::new(vocabulary).expect("encoder");
    let encoded = encoder.encode("BETA alpha beta", 4, 0.25).expect("encode");
    assert_eq!(encoded.tokens(), ["beta", "alpha", "beta"]);
    assert_eq!(encoded.token_ids(), [1, 2, 1]);
    assert_eq!(encoded.currents_na().iter().sum::<f64>(), 0.75);
    assert_eq!(
        encoded,
        encoder.encode("BETA alpha beta", 4, 0.25).expect("repeat")
    );
    assert!(encoder.encode("text", 0, 1.0).is_err());
    assert!(encoder.encode("text", 3, f64::NAN).is_err());
}

#[test]
fn encoder_menolak_input_yang_tidak_menghasilkan_token() {
    let vocabulary = Vocabulary::fit(["token valid"], 1, 8).expect("fit");
    let encoder = TextEncoder::new(vocabulary).expect("encoder");
    for text in ["", " \n\t ", "!!! ???"] {
        assert!(matches!(
            encoder.encode(text, 4, 1.0),
            Err(SimError::InvalidInput { .. })
        ));
    }
}

use serde_json::json;
use shella::{
    Candidate, DecisionPolicy, Engine, LifConfig, LifNetwork, MemoryStore, RewardEvaluator,
    RewardEvaluatorConfig, RewardModulatedStdpConfig, SdkInput as Input, StdpConfig, TextEncoder,
    Vocabulary,
};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static ID: AtomicU64 = AtomicU64::new(0);
fn temp(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "shella-sdk-{label}-{}-{}.json",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ))
}
fn network() -> LifNetwork {
    let mut network = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 3)
        .expect("network");
    network
        .enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
            StdpConfig::new(0.5, 0.4, 10.0, 10.0),
            10.0,
            0.1,
        ))
        .expect("reward STDP");
    network
}
fn encoder() -> TextEncoder {
    TextEncoder::new(Vocabulary::fit(["alpha beta"], 1, 8).expect("vocabulary")).expect("encoder")
}
fn policy() -> DecisionPolicy {
    DecisionPolicy::new(BTreeMap::from([("spike".into(), 1.0)]), 0.5, 0.0).expect("policy")
}

#[test]
fn sdk_end_to_end_teks_diproses_network_memori_dan_decision() {
    let memory_path = temp("memory");
    let checkpoint_path = temp("checkpoint");
    let memory = MemoryStore::open(&memory_path).expect("memori");
    let mut engine = Engine::new(network())
        .with_text_encoder(encoder())
        .with_decision_policy(policy())
        .with_memory(memory);
    let mut evaluator =
        RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 1.0)).expect("evaluator");
    let (output, id) = {
        let mut session = engine.session();
        let output = session
            .step(Input::Text {
                text: "ALPHA beta".into(),
                current_per_token_na: 100.0,
            })
            .expect("teks diproses");
        assert_eq!(output.input_kind(), shella::SdkInputKind::Text);
        assert_eq!(
            output.encoded_text().expect("encoding ada").token_ids(),
            [1, 2]
        );
        assert_eq!(output.spiking_neurons().len(), 2);
        let evaluation = session
            .evaluate_actions(&output, Some(0), &[1, 2], &mut evaluator)
            .expect("aksi dievaluasi dari spike");
        assert!(evaluation.correct());
        let applied = session
            .evaluate_actions_and_apply(&output, Some(0), &[1, 2], &mut evaluator)
            .expect("reward diterapkan melalui SDK");
        assert!(applied.correct());
        assert_eq!(evaluator.metrics().samples(), 2);
        assert_eq!(evaluator.metrics().total_reward(), 2.0);
        let candidates = [
            Candidate::new(
                "active",
                BTreeMap::from([("spike".into(), output.spiking_neurons().len() as f64)]),
            ),
            Candidate::new("idle", BTreeMap::from([("spike".into(), 0.0)])),
        ];
        assert_eq!(
            session.decide(&candidates).expect("decision").selected_id(),
            Some("active")
        );
        let id = session
            .remember(
                "input/1",
                json!({"text":"ALPHA beta","spikes":output.spiking_neurons()}),
                &["input".into()],
            )
            .expect("memory write");
        session
            .save_checkpoint(&checkpoint_path, Some(&evaluator))
            .expect("checkpoint via SDK");
        (output, id)
    };
    assert_eq!(engine.completed_steps(), 1);
    assert_eq!(
        engine
            .memory()
            .expect("memory")
            .get_by_id(&id)
            .expect("record")
            .key(),
        "input/1"
    );

    let next = engine
        .session()
        .step(Input::Currents(vec![0.0; 3]))
        .expect("next step");
    let fresh = Engine::new(network())
        .with_text_encoder(encoder())
        .with_decision_policy(policy());
    let mut resumed_engine = fresh;
    let restored_evaluator = resumed_engine
        .session()
        .restore_checkpoint(&checkpoint_path)
        .expect("restore SDK");
    assert_eq!(restored_evaluator, Some(evaluator.clone()));
    let resumed = resumed_engine
        .session()
        .step(Input::Currents(vec![0.0; 3]))
        .expect("resumed step");
    assert_eq!(resumed, next);
    assert_eq!(output.step().completed_step(), 1);

    fs::remove_file(checkpoint_path).expect("hapus checkpoint");
    fs::remove_file(memory_path).expect("hapus database");
}

#[test]
fn sdk_mengembalikan_error_untuk_komponen_opsional_yang_tidak_dipasang() {
    let mut engine = Engine::new(network());
    let mut session = engine.session();
    assert!(
        session
            .step(Input::Text {
                text: "hello".into(),
                current_per_token_na: 1.0
            })
            .is_err()
    );
    assert!(session.decide(&[]).is_err());
    assert!(session.remember("key", json!(1), &[]).is_err());
}

#[test]
fn sdk_teks_ke_jaringan_terlatih_ke_evaluasi_decision_dan_memory() {
    let memory_path = temp("learned-memory");
    let checkpoint_path = temp("learned-checkpoint");
    let mut trained_network =
        LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 4)
            .expect("network");
    for source in [1, 2] {
        trained_network
            .add_synapse(shella::Synapse::new(source, 3, 0.0, 0.0, 150.0, 1).expect("synapse"))
            .expect("add synapse");
    }
    trained_network
        .enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
            StdpConfig::new(1.0, 1.0, 10.0, 10.0),
            20.0,
            0.75,
        ))
        .expect("reward STDP");

    let mut engine = Engine::new(trained_network)
        .with_text_encoder(encoder())
        .with_decision_policy(
            DecisionPolicy::new(BTreeMap::from([("action_spike".into(), 1.0)]), 0.5, 0.0)
                .expect("policy"),
        )
        .with_memory(MemoryStore::open(&memory_path).expect("memory"));
    let mut teacher_evaluator =
        RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 1.0)).expect("teacher evaluator");

    let id = {
        let mut session = engine.session();
        for _ in 0..32 {
            let cue = session
                .step(Input::Text {
                    text: "alpha beta".into(),
                    current_per_token_na: 100.0,
                })
                .expect("text cue");
            assert!(cue.spiking_neurons().contains(&1));
            assert!(cue.spiking_neurons().contains(&2));

            let teacher = session
                .step(Input::Currents(vec![0.0, 0.0, 0.0, 100.0]))
                .expect("teacher action");
            let update = session
                .evaluate_actions_and_apply(&teacher, Some(0), &[3], &mut teacher_evaluator)
                .expect("reward update through SDK");
            assert!(update.correct());
            session
                .step(Input::Currents(vec![0.0; 4]))
                .expect("trial gap");
        }
        for _ in 0..50 {
            session
                .step(Input::Currents(vec![0.0; 4]))
                .expect("washout");
        }

        let _cue = session
            .step(Input::Text {
                text: "alpha beta".into(),
                current_per_token_na: 100.0,
            })
            .expect("held-out inference cue");
        let action = session
            .step(Input::Currents(vec![0.0; 4]))
            .expect("delayed learned action");
        assert!(action.spiking_neurons().contains(&3));

        let mut test_evaluator =
            RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 1.0)).expect("test evaluator");
        let evaluation = session
            .evaluate_actions(&action, Some(0), &[3], &mut test_evaluator)
            .expect("inference evaluated");
        assert!(evaluation.correct());

        let action_active = action.spiking_neurons().contains(&3);
        let decision = session
            .decide(&[
                Candidate::new(
                    "active",
                    BTreeMap::from([("action_spike".into(), f64::from(action_active))]),
                ),
                Candidate::new("idle", BTreeMap::from([("action_spike".into(), 0.0)])),
            ])
            .expect("decision from network output");
        assert_eq!(decision.selected_id(), Some("active"));

        let id = session
            .remember(
                "sdk/learned-action",
                json!({"tokens":["alpha","beta"],"action_spike":action_active}),
                &["sdk-test".into()],
            )
            .expect("memory write");
        session
            .save_checkpoint(&checkpoint_path, Some(&teacher_evaluator))
            .expect("checkpoint with trained state");
        id
    };

    let record = engine
        .memory()
        .expect("memory configured")
        .get_by_id(&id)
        .expect("integrated output remembered");
    assert_eq!(record.key(), "sdk/learned-action");
    assert_eq!(record.value()["action_spike"], json!(true));
    assert!(teacher_evaluator.metrics().samples() >= 32);

    fs::remove_file(checkpoint_path).expect("checkpoint cleanup");
    fs::remove_file(memory_path).expect("memory cleanup");
}

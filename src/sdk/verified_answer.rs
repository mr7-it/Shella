use super::session::Session;
use crate::{
    GeneratedText, GenerationConfig, GenerativeError, GenerativeLanguageModel, HybridAction,
    LanguageNetwork, ReasoningEngine, ReasoningResult, SemanticAnswerEvidence,
    SemanticQuestionIntent, SemanticResponse, SimError,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fmt};

const MAX_ANSWER_REVISIONS: usize = 1;

/// Indeks aksi pada slice `action_neurons`, bukan indeks absolut neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerActionMap {
    pub draft: usize,
    pub approve: usize,
    pub revise: usize,
    pub abstain: usize,
}

impl AnswerActionMap {
    fn validate(self, action_count: usize) -> Result<(), SimError> {
        let indices = [self.draft, self.approve, self.revise, self.abstain];
        let unique = indices.into_iter().collect::<BTreeSet<_>>();
        if action_count < 4
            || unique.len() != indices.len()
            || unique.iter().any(|i| *i >= action_count)
        {
            return Err(SimError::InvalidInput {
                parameter: "answer_action_map",
                reason: "empat aksi harus berbeda dan berada di dalam action_neurons",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifiedAnswerStatus {
    Answered,
    Abstained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerAbstentionReason {
    InsufficientEvidence,
    AmbiguousQuestion,
    LifDidNotAuthorizeDraft,
    EvidenceWouldBeTruncated,
    DraftGenerationTruncated,
    LifDidNotApproveDraft,
    HybridLifDidNotApproveDraft,
    DraftFailedVerification,
}

/// Trace numerik dari jaringan LIF. Confidence dan dukungan adalah skor heuristik.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifAnswerTrace {
    pub selected_action: Option<usize>,
    pub confidence: f64,
    pub calibrated_confidence: Option<f64>,
    pub uncertainty: f64,
    pub evidence_strength: f64,
    pub action_support: Vec<f64>,
    pub spiking_neurons_by_step: Vec<Vec<usize>>,
}

impl LifAnswerTrace {
    fn from_result(result: &ReasoningResult) -> Self {
        Self {
            selected_action: result.selected_action(),
            confidence: result.confidence(),
            calibrated_confidence: result.calibrated_confidence(),
            uncertainty: result.uncertainty(),
            evidence_strength: result.evidence_strength(),
            action_support: result.action_support().to_vec(),
            spiking_neurons_by_step: result
                .steps()
                .iter()
                .map(|step| step.network_step().spiking_neurons().to_vec())
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DraftVerification {
    pub passed: bool,
    pub question_alignment: bool,
    pub evidence_overlap: bool,
    pub polarity_consistent: bool,
    pub grounded_content_ratio: f64,
    pub unsupported_terms: Vec<String>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerDraftReview {
    pub generated: GeneratedText,
    pub lif: LifAnswerTrace,
    #[serde(default)]
    pub hybrid_lif_action: Option<HybridAction>,
    pub verification: DraftVerification,
    pub approved_by_lif: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedAnswer {
    pub question: String,
    pub status: VerifiedAnswerStatus,
    pub answer: Option<String>,
    pub abstention_reason: Option<AnswerAbstentionReason>,
    /// Respons, evidence ID, dan provenance dari SemanticMemory.
    pub semantic: SemanticResponse,
    /// LIF trace saat menerima pertanyaan dan bukti.
    pub question_lif: LifAnswerTrace,
    pub evidence_lif: Option<LifAnswerTrace>,
    /// Setiap draf diperiksa oleh LIF dan verifier sebelum dipakai.
    pub reviews: Vec<AnswerDraftReview>,
    pub revision_count: usize,
    /// Bukan bukti entailment umum atau verifikasi fakta dunia luar.
    pub verification_scope: String,
}

#[derive(Debug)]
pub enum VerifiedAnswerError {
    Reasoning(SimError),
    Generation(GenerativeError),
}

impl fmt::Display for VerifiedAnswerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reasoning(error) => write!(formatter, "reasoning jawaban gagal: {error}"),
            Self::Generation(error) => write!(formatter, "generasi jawaban gagal: {error}"),
        }
    }
}

impl std::error::Error for VerifiedAnswerError {}

impl From<SimError> for VerifiedAnswerError {
    fn from(error: SimError) -> Self {
        Self::Reasoning(error)
    }
}

impl From<GenerativeError> for VerifiedAnswerError {
    fn from(error: GenerativeError) -> Self {
        Self::Generation(error)
    }
}

impl Session<'_> {
    /// Menjalankan alur evidence -> LIF -> Transformer -> LIF/verifier -> satu revisi.
    ///
    /// Aksi yang diminta `action_map` harus telah dipelajari/diatur pada `ReasoningEngine`.
    /// Generator hanya dipanggil setelah SemanticMemory memberi pernyataan pendukung dan
    /// LIF memilih aksi `draft`. Jawaban hanya dilepas jika LIF memilih `approve` dan
    /// pemeriksaan leksikal/polaritas konservatif lulus. Pemeriksaan ini bukan entailment umum.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_with_neural_review(
        &mut self,
        reasoning: &mut ReasoningEngine,
        model: &GenerativeLanguageModel,
        context_id: &str,
        question: &str,
        language_network: &LanguageNetwork,
        current_per_word_na: f64,
        action_neurons: &[usize],
        action_map: AnswerActionMap,
        retrieval_limit: usize,
        generation: GenerationConfig,
    ) -> Result<VerifiedAnswer, VerifiedAnswerError> {
        action_map.validate(action_neurons.len())?;
        let (semantic, question_result) = self
            .respond_with_reasoning(
                reasoning,
                context_id,
                question,
                language_network,
                current_per_word_na,
                action_neurons,
                retrieval_limit,
            )
            .map_err(VerifiedAnswerError::Reasoning)?;
        let question_lif = LifAnswerTrace::from_result(&question_result);
        let mut result = VerifiedAnswer {
            question: question.to_owned(),
            status: VerifiedAnswerStatus::Abstained,
            answer: None,
            abstention_reason: None,
            semantic,
            question_lif,
            evidence_lif: None,
            reviews: Vec::new(),
            revision_count: 0,
            verification_scope: "LIF action gating plus exact normalized content-term coverage and yes/no polarity checks; not general entailment, factuality, or external source verification".to_owned(),
        };

        if !has_usable_evidence(&result.semantic) {
            result.abstention_reason = Some(if !result.semantic.analysis.ambiguities.is_empty() {
                AnswerAbstentionReason::AmbiguousQuestion
            } else {
                AnswerAbstentionReason::InsufficientEvidence
            });
            return Ok(result);
        }
        if result.question_lif.selected_action != Some(action_map.draft) {
            result.abstention_reason = Some(AnswerAbstentionReason::LifDidNotAuthorizeDraft);
            return Ok(result);
        }

        let evidence_text = result.semantic.supporting_statements.join("; ");
        let evidence_input = format!("pertanyaan {question} bukti {evidence_text}");
        let evidence_result = reasoning
            .reason_text(
                context_id,
                &evidence_input,
                language_network,
                current_per_word_na,
                action_neurons,
            )
            .map_err(VerifiedAnswerError::Reasoning)?;
        result.evidence_lif = Some(LifAnswerTrace::from_result(&evidence_result));
        if evidence_result.selected_action() != Some(action_map.draft) {
            result.abstention_reason = Some(AnswerAbstentionReason::LifDidNotAuthorizeDraft);
            return Ok(result);
        }

        let evidence_lines = result
            .semantic
            .supporting_statements
            .iter()
            .map(|statement| format!("- {}", one_line(statement)))
            .collect::<Vec<_>>()
            .join("\n");
        let base_prompt = format!(
            "jawab singkat hanya berdasarkan bukti berikut. jangan menambah fakta. pertanyaan: {}\nbukti:\n{}\nacuan jawaban berbukti: {}\njawaban:",
            one_line(question),
            evidence_lines,
            one_line(&result.semantic.answer)
        );
        if !prompt_and_generation_fit(model, &base_prompt, &generation)? {
            result.abstention_reason = Some(AnswerAbstentionReason::EvidenceWouldBeTruncated);
            return Ok(result);
        }

        for attempt in 0..=MAX_ANSWER_REVISIONS {
            let prompt = if attempt == 0 {
                base_prompt.clone()
            } else {
                let prior = result.reviews.last().ok_or(SimError::InvalidInput {
                    parameter: "answer_revision",
                    reason: "revisi harus memiliki draf yang gagal diperiksa",
                })?;
                let feedback = revision_feedback(&prior.verification);
                format!(
                    "{base_prompt}\nperbaiki draf sebelumnya: {}\nmasalah: {feedback}\njawaban:",
                    one_line(&prior.generated.continuation)
                )
            };
            if !prompt_and_generation_fit(model, &prompt, &generation)? {
                result.abstention_reason = Some(AnswerAbstentionReason::EvidenceWouldBeTruncated);
                break;
            }
            let mut generated = model.generate(&prompt, generation.clone())?;
            if generated.prompt_truncated {
                result.abstention_reason = Some(AnswerAbstentionReason::DraftGenerationTruncated);
                break;
            }
            let draft = generated.continuation.trim().to_owned();
            let review_input = format!(
                "pertanyaan {} bukti {} calon jawaban {}",
                one_line(question),
                one_line(&evidence_text),
                one_line(&draft)
            );
            let lif_result = reasoning
                .reason_text(
                    context_id,
                    &review_input,
                    language_network,
                    current_per_word_na,
                    action_neurons,
                )
                .map_err(VerifiedAnswerError::Reasoning)?;
            let lif = LifAnswerTrace::from_result(&lif_result);
            let hybrid_lif_action = if model.has_trained_hybrid_head() {
                Some(model.predict_hybrid_action(&review_input)?.action)
            } else {
                None
            };
            let verification = verify_draft(
                question,
                &result.semantic.answer,
                &result.semantic.supporting_statements,
                result.semantic.intent,
                &draft,
            );
            let approved_by_lif = lif.selected_action == Some(action_map.approve);
            let approved_by_hybrid_lif =
                hybrid_lif_action.is_none_or(|action| action == HybridAction::Answer);
            if approved_by_lif && approved_by_hybrid_lif && verification.passed {
                generated.factual_grounding = "passed LIF approve action and conservative lexical/polarity checks against supplied SemanticMemory evidence; this is not general entailment or external fact verification".to_owned();
                result.answer = Some(draft);
                result.status = VerifiedAnswerStatus::Answered;
                result.abstention_reason = None;
                result.reviews.push(AnswerDraftReview {
                    generated,
                    lif,
                    hybrid_lif_action,
                    verification,
                    approved_by_lif,
                });
                result.revision_count = attempt;
                return Ok(result);
            }
            let should_revise = attempt < MAX_ANSWER_REVISIONS
                && hybrid_lif_action != Some(HybridAction::Abstain)
                && lif.selected_action != Some(action_map.abstain)
                && (lif.selected_action == Some(action_map.revise)
                    || hybrid_lif_action == Some(HybridAction::Revise)
                    || !verification.passed);
            result.reviews.push(AnswerDraftReview {
                generated,
                lif,
                hybrid_lif_action,
                verification,
                approved_by_lif,
            });
            if !should_revise {
                result.abstention_reason = Some(
                    if result
                        .reviews
                        .last()
                        .is_some_and(|review| !review.verification.passed)
                    {
                        AnswerAbstentionReason::DraftFailedVerification
                    } else {
                        if result.reviews.last().is_some_and(|review| {
                            review
                                .hybrid_lif_action
                                .is_some_and(|action| action != HybridAction::Answer)
                        }) {
                            AnswerAbstentionReason::HybridLifDidNotApproveDraft
                        } else {
                            AnswerAbstentionReason::LifDidNotApproveDraft
                        }
                    },
                );
                break;
            }
            result.revision_count += 1;
        }
        if result.abstention_reason.is_none() {
            result.abstention_reason = Some(AnswerAbstentionReason::DraftFailedVerification);
        }
        Ok(result)
    }
}

fn has_usable_evidence(response: &SemanticResponse) -> bool {
    matches!(
        response.evidence_level,
        SemanticAnswerEvidence::DirectSource | SemanticAnswerEvidence::ContextStatement
    ) && response.analysis.ambiguities.is_empty()
        && !response.evidence_ids.is_empty()
        && !response.supporting_statements.is_empty()
}

fn prompt_and_generation_fit(
    model: &GenerativeLanguageModel,
    prompt: &str,
    generation: &GenerationConfig,
) -> Result<bool, GenerativeError> {
    let prompt_tokens = model.tokenizer().encode_prompt(prompt)?.len();
    Ok(prompt_tokens
        .checked_add(generation.max_new_tokens)
        .is_some_and(|total| total <= model.config().max_context))
}

fn one_line(text: &str) -> String {
    text.trim()
        .replace(['\n', '\r', '\t'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn revision_feedback(verification: &DraftVerification) -> String {
    let mut messages = verification.reasons.clone();
    if !verification.unsupported_terms.is_empty() {
        messages.push(format!(
            "istilah tanpa dukungan: {}",
            verification.unsupported_terms.join(", ")
        ));
    }
    if messages.is_empty() {
        "draf belum disetujui oleh jaringan LIF".to_owned()
    } else {
        messages.join("; ")
    }
}

fn content_terms(text: &str) -> BTreeSet<String> {
    const STOPWORDS: &[&str] = &[
        "apa",
        "apakah",
        "bagaimana",
        "mengapa",
        "siapa",
        "kapan",
        "dimana",
        "mana",
        "yang",
        "dan",
        "atau",
        "dari",
        "ke",
        "di",
        "pada",
        "dengan",
        "untuk",
        "oleh",
        "adalah",
        "ialah",
        "itu",
        "ini",
        "saya",
        "anda",
        "kami",
        "kita",
        "dia",
        "mereka",
        "ya",
        "tidak",
        "bukan",
        "belum",
        "sangat",
        "lebih",
        "kurang",
        "dapat",
        "bisa",
        "sebuah",
        "suatu",
        "ada",
        "akan",
        "telah",
        "sudah",
        "masih",
        "juga",
        "hanya",
    ];
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.chars().count() >= 3 && !STOPWORDS.contains(token))
        .map(str::to_owned)
        .collect()
}

fn contains_negation(text: &str) -> bool {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| matches!(token, "tidak" | "bukan" | "belum" | "tanpa" | "jangan"))
}

fn verify_draft(
    question: &str,
    canonical_answer: &str,
    evidence: &[String],
    intent: SemanticQuestionIntent,
    draft: &str,
) -> DraftVerification {
    let candidate_terms = content_terms(draft);
    let question_terms = content_terms(question);
    let evidence_terms = content_terms(&evidence.join(" "));
    let mut permitted_terms = evidence_terms.clone();
    permitted_terms.extend(content_terms(canonical_answer));
    let unsupported_terms = candidate_terms
        .difference(&permitted_terms)
        .cloned()
        .collect::<Vec<_>>();
    let question_alignment = !candidate_terms.is_disjoint(&question_terms);
    let evidence_overlap = !candidate_terms.is_disjoint(&evidence_terms);
    let polarity_consistent = intent != SemanticQuestionIntent::YesNo
        || contains_negation(draft) == contains_negation(canonical_answer);
    let grounded_content_ratio = if candidate_terms.is_empty() {
        0.0
    } else {
        candidate_terms.intersection(&permitted_terms).count() as f64 / candidate_terms.len() as f64
    };
    let mut reasons = Vec::new();
    if candidate_terms.is_empty() {
        reasons.push("draf tidak memiliki istilah konten yang dapat diperiksa".to_owned());
    }
    if !question_alignment {
        reasons.push("draf tidak cocok secara leksikal dengan topik pertanyaan".to_owned());
    }
    if !evidence_overlap {
        reasons.push("draf tidak menyebut istilah konten dari evidence".to_owned());
    }
    if !unsupported_terms.is_empty() {
        reasons.push("draf memuat istilah konten di luar evidence dan acuan jawaban".to_owned());
    }
    if !polarity_consistent {
        reasons.push("polaritas ya/tidak draf berbeda dari jawaban berbukti".to_owned());
    }
    DraftVerification {
        passed: !candidate_terms.is_empty()
            && question_alignment
            && evidence_overlap
            && unsupported_terms.is_empty()
            && polarity_consistent,
        question_alignment,
        evidence_overlap,
        polarity_consistent,
        grounded_content_ratio,
        unsupported_terms,
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_menerima_draf_yang_istilah_dan_polaritasnya_didukung() {
        let evidence = vec!["Kucing lebih kecil daripada rumah manusia.".to_owned()];
        let result = verify_draft(
            "Apakah kucing sebesar rumah manusia?",
            "Tidak, kucing jauh lebih kecil daripada rumah manusia.",
            &evidence,
            SemanticQuestionIntent::YesNo,
            "Tidak. Kucing lebih kecil daripada rumah manusia.",
        );
        assert!(result.passed, "{result:?}");
        assert_eq!(result.grounded_content_ratio, 1.0);
    }

    #[test]
    fn verifier_menolak_klaim_tanpa_dasar_dan_polaritas_terbalik() {
        let evidence = vec!["Kucing lebih kecil daripada rumah manusia.".to_owned()];
        let result = verify_draft(
            "Apakah kucing sebesar rumah manusia?",
            "Tidak, kucing jauh lebih kecil daripada rumah manusia.",
            &evidence,
            SemanticQuestionIntent::YesNo,
            "Ya, kucing sebesar planet raksasa.",
        );
        assert!(!result.passed);
        assert!(!result.unsupported_terms.is_empty());
        assert!(!result.polarity_consistent);
    }
}

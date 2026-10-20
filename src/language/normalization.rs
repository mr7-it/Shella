use crate::SimError;
use unicode_normalization::UnicodeNormalization;

const MAX_TEXT_BYTES: usize = 1_048_576;

/// Mengubah huruf menjadi lowercase dan merapatkan spasi tanpa menghapus tanda baca.
pub fn normalize_text(text: &str) -> Result<String, SimError> {
    if text.len() > MAX_TEXT_BYTES {
        return Err(SimError::InvalidInput {
            parameter: "text",
            reason: "teks melebihi 1 MiB",
        });
    }
    if text.chars().any(|character| character == '\0') {
        return Err(SimError::InvalidInput {
            parameter: "text",
            reason: "teks tidak boleh memuat NUL",
        });
    }
    let normalized: String = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .nfc()
        .collect::<String>()
        .to_lowercase()
        .nfc()
        .collect();
    Ok(normalized)
}

use super::normalize_text;
use crate::SimError;

/// Tokenisasi lowercase: grup Unicode alfanumerik adalah token, tanda baca menjadi pemisah.
pub fn tokenize(text: &str) -> Result<Vec<String>, SimError> {
    let normalized = normalize_text(text)?;
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in normalized.chars() {
        if character.is_alphanumeric() {
            current.push(character);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    Ok(tokens)
}

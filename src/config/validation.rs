use crate::SimError;

pub(super) fn validate_positive(parameter: &'static str, value: f64) -> Result<(), SimError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(SimError::InvalidConfiguration {
            parameter,
            reason: "harus berupa bilangan hingga yang lebih besar dari nol",
        });
    }
    Ok(())
}

pub(super) fn validate_finite(parameter: &'static str, value: f64) -> Result<(), SimError> {
    if !value.is_finite() {
        return Err(SimError::InvalidConfiguration {
            parameter,
            reason: "harus berupa bilangan hingga",
        });
    }
    Ok(())
}

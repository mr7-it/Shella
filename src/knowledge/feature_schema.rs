use crate::{SimError, benchmark::IrisSample};
use serde::{Deserialize, Serialize};

pub const FEATURE_SCHEMA_VERSION: u32 = 2;
pub const FEATURE_COUNT: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureType {
    Continuous,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureDefinition {
    pub name: String,
    pub data_type: FeatureType,
    pub unit: String,
    pub formula: String,
    pub schema_version: u32,
    pub active: bool,
}

pub fn feature_definitions() -> [FeatureDefinition; FEATURE_COUNT] {
    let definitions = [
        ("sepal_length", "cm", "raw sepal length"),
        ("sepal_width", "cm", "raw sepal width"),
        ("petal_length", "cm", "raw petal length"),
        ("petal_width", "cm", "raw petal width"),
        ("sepal_area", "cm^2", "sepal_length * sepal_width"),
        ("petal_area", "cm^2", "petal_length * petal_width"),
        ("sepal_ratio", "ratio", "sepal_length / sepal_width"),
        ("petal_ratio", "ratio", "petal_length / petal_width"),
        (
            "petal_sepal_length_ratio",
            "ratio",
            "petal_length / sepal_length",
        ),
        ("petal_sepal_area_ratio", "ratio", "petal_area / sepal_area"),
    ];
    definitions.map(|(name, unit, formula)| FeatureDefinition {
        name: name.to_owned(),
        data_type: FeatureType::Continuous,
        unit: unit.to_owned(),
        formula: formula.to_owned(),
        schema_version: FEATURE_SCHEMA_VERSION,
        active: true,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IrisFeatures {
    raw: [f64; 4],
    derived: [f64; FEATURE_COUNT],
}

impl IrisFeatures {
    pub fn from_raw(raw: [f64; 4]) -> Result<Self, SimError> {
        if raw.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            return Err(SimError::InvalidInput {
                parameter: "iris_features",
                reason: "ukuran Iris harus finite dan positif",
            });
        }
        let [sepal_length, sepal_width, petal_length, petal_width] = raw;
        let derived = [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            sepal_length * sepal_width,
            petal_length * petal_width,
            sepal_length / sepal_width,
            petal_length / petal_width,
            petal_length / sepal_length,
            (petal_length * petal_width) / (sepal_length * sepal_width),
        ];
        if derived.iter().any(|value| !value.is_finite()) {
            return Err(SimError::NumericalOverflow {
                operation: "feature engineering Iris",
            });
        }
        Ok(Self { raw, derived })
    }

    #[must_use]
    pub const fn raw(&self) -> &[f64; 4] {
        &self.raw
    }

    #[must_use]
    pub const fn values(&self) -> &[f64; FEATURE_COUNT] {
        &self.derived
    }
}

/// Min/max scaler whose bounds are fitted exclusively from the supplied training rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureScaler {
    schema_version: u32,
    minimum: [f64; FEATURE_COUNT],
    maximum: [f64; FEATURE_COUNT],
}

impl FeatureScaler {
    pub fn fit(training: &[IrisSample]) -> Result<Self, SimError> {
        if training.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "training_samples",
                reason: "tidak boleh kosong",
            });
        }
        let mut minimum = [f64::INFINITY; FEATURE_COUNT];
        let mut maximum = [f64::NEG_INFINITY; FEATURE_COUNT];
        for sample in training {
            let features = IrisFeatures::from_raw(*sample.features())?;
            for (index, value) in features.values().iter().enumerate() {
                minimum[index] = minimum[index].min(*value);
                maximum[index] = maximum[index].max(*value);
            }
        }
        Ok(Self {
            schema_version: FEATURE_SCHEMA_VERSION,
            minimum,
            maximum,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), SimError> {
        if self.schema_version != FEATURE_SCHEMA_VERSION
            || self.minimum.iter().any(|value| !value.is_finite())
            || self.maximum.iter().any(|value| !value.is_finite())
            || (0..FEATURE_COUNT).any(|index| self.minimum[index] > self.maximum[index])
        {
            return Err(SimError::CorruptData {
                subsystem: "feature scaler",
                reason: "versi atau rentang min/max tidak valid".to_owned(),
            });
        }
        Ok(())
    }

    pub fn transform(&self, raw: [f64; 4]) -> Result<[f64; FEATURE_COUNT], SimError> {
        if self.schema_version != FEATURE_SCHEMA_VERSION {
            return Err(SimError::UnsupportedDataVersion {
                subsystem: "feature schema",
                version: self.schema_version,
            });
        }
        let values = IrisFeatures::from_raw(raw)?.derived;
        Ok(std::array::from_fn(|index| {
            let range = self.maximum[index] - self.minimum[index];
            if range == 0.0 {
                0.0
            } else {
                (values[index] - self.minimum[index]) / range
            }
        }))
    }

    pub fn restore_primary(&self, normalized: [f64; 4]) -> Result<[f64; 4], SimError> {
        if self.schema_version != FEATURE_SCHEMA_VERSION
            || normalized.iter().any(|value| !value.is_finite())
        {
            return Err(SimError::InvalidInput {
                parameter: "normalized_iris_features",
                reason: "schema harus dikenal dan nilai harus finite",
            });
        }
        Ok(std::array::from_fn(|index| {
            self.minimum[index] + normalized[index] * (self.maximum[index] - self.minimum[index])
        }))
    }

    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[must_use]
    pub const fn minimum(&self) -> &[f64; FEATURE_COUNT] {
        &self.minimum
    }

    #[must_use]
    pub const fn maximum(&self) -> &[f64; FEATURE_COUNT] {
        &self.maximum
    }
}

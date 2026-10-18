/// Jenis input yang diproses SDK.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Currents,
    Text,
}

/// Input terstruktur untuk satu langkah jaringan.
#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    /// Arus nanoampere satu elemen untuk tiap neuron.
    Currents(Vec<f64>),
    /// Teks yang diproyeksikan ke arus dengan amplitudo per token.
    Text {
        text: String,
        current_per_token_na: f64,
    },
}
impl Input {
    /// Jenis input tanpa memindahkan payload.
    #[must_use]
    pub const fn kind(&self) -> InputKind {
        match self {
            Self::Currents(_) => InputKind::Currents,
            Self::Text { .. } => InputKind::Text,
        }
    }
}

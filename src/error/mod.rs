use std::fmt::{Display, Formatter};

/// Kesalahan yang dapat terjadi saat mengonfigurasi atau menjalankan simulasi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimError {
    /// Parameter konfigurasi berada di luar domain yang didukung.
    InvalidConfiguration {
        /// Nama parameter yang ditolak.
        parameter: &'static str,
        /// Alasan parameter tidak sah.
        reason: &'static str,
    },
    /// Nilai masukan berada di luar domain yang didukung.
    InvalidInput {
        /// Nama nilai masukan yang ditolak.
        parameter: &'static str,
        /// Alasan nilai masukan tidak sah.
        reason: &'static str,
    },
    /// Operasi numerik menghasilkan nilai tak hingga atau NaN.
    NumericalOverflow {
        /// Operasi yang gagal.
        operation: &'static str,
    },
    /// Penghitung langkah tidak dapat dinaikkan tanpa meluap.
    StepCounterOverflow,
    /// Penghitung metrik tidak dapat dinaikkan tanpa meluap.
    CounterOverflow {
        /// Nama penghitung yang meluap.
        counter: &'static str,
    },
    /// Penghitung spike tidak dapat dinaikkan tanpa meluap.
    SpikeCounterOverflow,
    /// Topologi jaringan tidak memenuhi kontrak yang ditentukan.
    InvalidTopology {
        /// Alasan topologi ditolak.
        reason: &'static str,
    },
    /// Pasangan sumber-tujuan sudah memiliki koneksi.
    DuplicateSynapse {
        /// Indeks neuron sumber.
        source: usize,
        /// Indeks neuron tujuan.
        target: usize,
    },
    /// Alokasi ruang untuk catatan spike gagal.
    AllocationFailed,
    /// Kegagalan I/O dengan operasi dan kategori OS yang dapat diperiksa.
    Io {
        /// Operasi file yang gagal.
        operation: &'static str,
        /// Kategori error sistem.
        kind: std::io::ErrorKind,
    },
    /// Data checkpoint tidak valid atau integritasnya gagal.
    InvalidCheckpoint {
        /// Rincian kesalahan data.
        reason: String,
    },
    /// Versi format checkpoint tidak didukung implementasi ini.
    UnsupportedCheckpointVersion {
        /// Versi yang ditemukan.
        version: u32,
    },
    /// Record persisten dari subsystem selain checkpoint tidak dapat divalidasi.
    CorruptData {
        /// Nama subsystem penyimpan.
        subsystem: &'static str,
        /// Rincian data yang rusak atau tidak konsisten.
        reason: String,
    },
    /// Versi schema data persisten tidak dikenal oleh subsystem terkait.
    UnsupportedDataVersion {
        /// Nama subsystem penyimpan.
        subsystem: &'static str,
        /// Nomor versi yang ditemukan.
        version: u32,
    },
    /// Serialisasi gagal sebelum data dapat dikomit ke penyimpanan.
    SerializationFailed {
        /// Nama subsystem yang melakukan serialisasi.
        subsystem: &'static str,
        /// Penyebab yang dikembalikan serializer.
        reason: String,
    },
}

impl Display for SimError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration { parameter, reason } => {
                write!(formatter, "konfigurasi {parameter} tidak sah: {reason}")
            }
            Self::InvalidInput { parameter, reason } => {
                write!(formatter, "masukan {parameter} tidak sah: {reason}")
            }
            Self::NumericalOverflow { operation } => {
                write!(
                    formatter,
                    "hasil numerik tidak terbatas pada operasi {operation}"
                )
            }
            Self::StepCounterOverflow => write!(formatter, "penghitung langkah meluap"),
            Self::CounterOverflow { counter } => write!(formatter, "penghitung {counter} meluap"),
            Self::SpikeCounterOverflow => write!(formatter, "penghitung spike meluap"),
            Self::InvalidTopology { reason } => write!(formatter, "topologi tidak sah: {reason}"),
            Self::DuplicateSynapse { source, target } => {
                write!(formatter, "sinaps {source} -> {target} sudah ada")
            }
            Self::AllocationFailed => write!(formatter, "alokasi memori untuk catatan spike gagal"),
            Self::Io { operation, kind } => write!(formatter, "I/O {operation} gagal: {kind}"),
            Self::InvalidCheckpoint { reason } => {
                write!(formatter, "checkpoint tidak valid: {reason}")
            }
            Self::UnsupportedCheckpointVersion { version } => {
                write!(formatter, "versi checkpoint {version} tidak didukung")
            }
            Self::CorruptData { subsystem, reason } => {
                write!(
                    formatter,
                    "data {subsystem} rusak atau tidak valid: {reason}"
                )
            }
            Self::UnsupportedDataVersion { subsystem, version } => {
                write!(formatter, "versi data {subsystem} {version} tidak didukung")
            }
            Self::SerializationFailed { subsystem, reason } => {
                write!(formatter, "serialisasi {subsystem} gagal: {reason}")
            }
        }
    }
}

impl std::error::Error for SimError {}

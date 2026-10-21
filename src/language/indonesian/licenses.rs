use crate::SimError;
use serde::{Deserialize, Serialize};
use sha1::Digest;
use std::{
    collections::BTreeMap,
    io::{Cursor, Read},
};

use super::SourceProvenance;

const STOPWORDS_REVISION: &str = "6109494eb4cfebd75c59bd832848374d02f4e907";
const HUNSPELL_REVISION: &str = "65e552e7216cecab2b497c898a9a7ee1e2e2de11";
const PUD_REVISION: &str = "1d8741838d6b8fa64dbda85a33c4af0700fb7430";
const NUSAX_REVISION: &str = "75b0be5d982348332509f296e10d9bf9cb66d60f";
const WORDNET_BAHASA_REVISION: &str = "fb201ec6155dc5c3a23d76673cfbb28630bc9a67";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetId {
    UdIndonesianGsd,
    UdIndonesianCsui,
    UdIndonesianPud,
    StopwordsIsoId,
    WordNetBahasa,
    HunspellDictionaries,
    IndoNli,
    NusaX,
    IndoSum,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetProvenance {
    pub id: DatasetId,
    pub repository_url: String,
    pub revision: String,
    pub license_id: String,
    pub license_url: String,
    pub attribution: String,
    pub data_scope: String,
    pub file_git_blob_sha1: BTreeMap<String, String>,
}

fn nusax_file_hashes() -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "datasets/LICENSE",
            "cc3e2459dc5c46f1936b6e3b54137099063818e0",
        ),
        (
            "datasets/lexicon/acehnese.csv",
            "18a9f65d1d4063fbc0f008070a1287744873430b",
        ),
        (
            "datasets/lexicon/balinese.csv",
            "46f2f8174175bfc163c678a2d5122103f1b9fccd",
        ),
        (
            "datasets/lexicon/banjarese.csv",
            "300222ebaf356a3064766f9e51c3676154a3bd8a",
        ),
        (
            "datasets/lexicon/buginese.csv",
            "8313241d157bdf7a2b2fac0d50b783f2a9bff733",
        ),
        (
            "datasets/lexicon/english.csv",
            "e3b8993292c4f7a755c97b5e9d42324e7e0050e9",
        ),
        (
            "datasets/lexicon/javanese.csv",
            "329b71e1a248ea4af1d573ed7abebc2d5517b333",
        ),
        (
            "datasets/lexicon/madurese.csv",
            "f7b79e89c64f8cf519f80c3a8e8d0fb36fa52b25",
        ),
        (
            "datasets/lexicon/minangkabau.csv",
            "eb7143524ebb2e0a212dcae75aff431635c58286",
        ),
        (
            "datasets/lexicon/ngaju.csv",
            "50bfec3fa4399ff12386c340df898c2fbf9ea4fd",
        ),
        (
            "datasets/lexicon/sundanese.csv",
            "a69a613ef2ca5740fc1adb93e4dbfa4b0c8870d1",
        ),
        (
            "datasets/lexicon/toba_batak.csv",
            "51dbc6c153eb2dc5c152ef91749d6f168e2621d1",
        ),
        (
            "datasets/mt/test.csv",
            "49f55400f18a421a38b5aeb41881dce6bbb20bb4",
        ),
        (
            "datasets/mt/train.csv",
            "3c8ab88d4657a1ca932f372df6dbf6d2fa3e9a90",
        ),
        (
            "datasets/mt/valid.csv",
            "490912b9c744b84da3308f7fddbd3294daeaaeac",
        ),
        (
            "datasets/sentiment/acehnese/test.csv",
            "4eb7b7b1c93fcb2be1126e973c06f1cbaa46096d",
        ),
        (
            "datasets/sentiment/acehnese/train.csv",
            "75a82aa59683df25728439e825fdadf90e7f5567",
        ),
        (
            "datasets/sentiment/acehnese/valid.csv",
            "c00316286e6801d8a325210244e3183391dd956b",
        ),
        (
            "datasets/sentiment/balinese/test.csv",
            "02dbfa300bd6e4357ce881ed111ae07608fbd2f5",
        ),
        (
            "datasets/sentiment/balinese/train.csv",
            "e18ab06c8869a17368b02b2c553189ac120df5ab",
        ),
        (
            "datasets/sentiment/balinese/valid.csv",
            "ec21df44394f19e1f6fb5bbe93a6020bc55fc8a8",
        ),
        (
            "datasets/sentiment/banjarese/test.csv",
            "14c3a4b7ed48309d73866b2c3b1435f52309ad52",
        ),
        (
            "datasets/sentiment/banjarese/train.csv",
            "2a9f3fad03a1c753e03548b186e8d6c21a4f7927",
        ),
        (
            "datasets/sentiment/banjarese/valid.csv",
            "a4cb8b80b0a998aadb1f4098b13c1dad4307bfd0",
        ),
        (
            "datasets/sentiment/buginese/test.csv",
            "580b97f7835f2850e48bad43b56920b47358b12b",
        ),
        (
            "datasets/sentiment/buginese/train.csv",
            "f5c4cfdc574dd13f1c6cad5be2da3c8f9d4e6b8e",
        ),
        (
            "datasets/sentiment/buginese/valid.csv",
            "d936a2a9be47464a80922739f6c18bbb7c99029b",
        ),
        (
            "datasets/sentiment/english/test.csv",
            "177d27d817804ca16e64c7134ac696adb961ef40",
        ),
        (
            "datasets/sentiment/english/train.csv",
            "5a382fa281fb47ce03c5cef6d8152334250dabc0",
        ),
        (
            "datasets/sentiment/english/valid.csv",
            "611f75e3a9459a24572d20bc9d6db2388100de3f",
        ),
        (
            "datasets/sentiment/indonesian/test.csv",
            "5fd421d00eeb3a67956c44c5e190b6305143a79e",
        ),
        (
            "datasets/sentiment/indonesian/train.csv",
            "1be82eec8b97232838644566f56e6326fe038937",
        ),
        (
            "datasets/sentiment/indonesian/valid.csv",
            "7422a3fd47fabd97c7f1c917f44046836d4979e0",
        ),
        (
            "datasets/sentiment/javanese/test.csv",
            "a705ee331be496a452b027ee3c738952dd4e7fae",
        ),
        (
            "datasets/sentiment/javanese/train.csv",
            "c9b406254a9092e7cdd832120ec0ee42a1465cd3",
        ),
        (
            "datasets/sentiment/javanese/valid.csv",
            "cd319d7ab612793c585259912c0b6cff7defe313",
        ),
        (
            "datasets/sentiment/madurese/test.csv",
            "26ae2ae7f4dab0c9e4062b75d76e3a8dea0e24b9",
        ),
        (
            "datasets/sentiment/madurese/train.csv",
            "d36afb663831e38b53c21357bdcd741a950b840f",
        ),
        (
            "datasets/sentiment/madurese/valid.csv",
            "2ba475528ad99fed5b1f213713f840535bcdddbd",
        ),
        (
            "datasets/sentiment/minangkabau/test.csv",
            "d493013f338a6e88c89e566e763b01caec3fc528",
        ),
        (
            "datasets/sentiment/minangkabau/train.csv",
            "ce34fe32fd600b5a71e5b318ce501e11a064cba9",
        ),
        (
            "datasets/sentiment/minangkabau/valid.csv",
            "471f855eed10187014c36d17f16cdff21a626a21",
        ),
        (
            "datasets/sentiment/ngaju/test.csv",
            "722e8f8899e92093992df1f731629fab5bea1048",
        ),
        (
            "datasets/sentiment/ngaju/train.csv",
            "427c10b64827467cbd3822a1f74a8427739ca94b",
        ),
        (
            "datasets/sentiment/ngaju/valid.csv",
            "90e20f289add1a3318d36acb634c05442b2f1e32",
        ),
        (
            "datasets/sentiment/sundanese/test.csv",
            "5220da03333c9fa119e3f7b90813ec170f65f2af",
        ),
        (
            "datasets/sentiment/sundanese/train.csv",
            "fc18e8bede554b28c46707cacceb3c64a6efddc0",
        ),
        (
            "datasets/sentiment/sundanese/valid.csv",
            "e22327c234b540f18cd44b51b19898f007e6f3a6",
        ),
        (
            "datasets/sentiment/toba_batak/test.csv",
            "d521a71037e8a6d1b299108b82bfcc8335825aa0",
        ),
        (
            "datasets/sentiment/toba_batak/train.csv",
            "0ea8587dceb95279e1b7685bdf5ccb6319cc0ed4",
        ),
        (
            "datasets/sentiment/toba_batak/valid.csv",
            "f3aca2a54f0b19a63887031ab04cb06ee256a5c7",
        ),
    ])
    .into_iter()
    .map(|(path, hash)| (path.to_owned(), hash.to_owned()))
    .collect()
}

/// Allowlist of sources whose repository and dataset license metadata was checked.
///
/// A registry entry verifies the repository's published license notice and pinned
/// revision; it does not determine rights in third-party text quoted by a dataset.
#[derive(Debug, Clone, Default)]
pub struct LicenseRegistry {
    approved: Vec<DatasetProvenance>,
}

impl LicenseRegistry {
    #[must_use]
    pub fn verified_sources() -> Self {
        Self {
            approved: vec![
                DatasetProvenance {
                    id: DatasetId::StopwordsIsoId,
                    repository_url: "https://github.com/stopwords-iso/stopwords-id".to_owned(),
                    revision: STOPWORDS_REVISION.to_owned(),
                    license_id: "MIT".to_owned(),
                    license_url: "https://github.com/stopwords-iso/stopwords-id/blob/6109494eb4cfebd75c59bd832848374d02f4e907/LICENSE".to_owned(),
                    attribution: "Gene Diaz, stopwords-iso/stopwords-id".to_owned(),
                    data_scope: "Indonesian stopword list".to_owned(),
                    file_git_blob_sha1: BTreeMap::from([(
                        "stopwords-id.json".to_owned(),
                        "89dfca515cc47ddd1ee61520882d3f4d30f6916f".to_owned(),
                    )]),
                },
                DatasetProvenance {
                    id: DatasetId::UdIndonesianPud,
                    repository_url: "https://github.com/UniversalDependencies/UD_Indonesian-PUD"
                        .to_owned(),
                    revision: PUD_REVISION.to_owned(),
                    license_id: "CC BY-SA 3.0".to_owned(),
                    license_url: "https://creativecommons.org/licenses/by-sa/3.0/".to_owned(),
                    attribution: "Universal Dependencies Indonesian-PUD contributors"
                        .to_owned(),
                    data_scope: "Indonesian PUD test treebank, including its source text"
                        .to_owned(),
                    file_git_blob_sha1: BTreeMap::from([(
                        "id_pud-ud-test.conllu".to_owned(),
                        "c975cdc18d0ea66fe1b9280662451d3983815cdd".to_owned(),
                    )]),
                },
                DatasetProvenance {
                    id: DatasetId::NusaX,
                    repository_url: "https://github.com/IndoNLP/nusax".to_owned(),
                    revision: NUSAX_REVISION.to_owned(),
                    license_id: "CC BY-SA 4.0".to_owned(),
                    license_url: format!(
                        "https://github.com/IndoNLP/nusax/blob/{NUSAX_REVISION}/datasets/LICENSE"
                    ),
                    attribution: "IndoNLP NusaX contributors".to_owned(),
                    data_scope: "NusaX machine translation, sentiment, and multilingual lexicon datasets"
                        .to_owned(),
                    file_git_blob_sha1: nusax_file_hashes(),
                },
                DatasetProvenance {
                    id: DatasetId::WordNetBahasa,
                    repository_url: "https://github.com/open-language/wordnets".to_owned(),
                    revision: WORDNET_BAHASA_REVISION.to_owned(),
                    license_id: "MIT".to_owned(),
                    license_url: "https://sourceforge.net/p/wn-msa/tab/HEAD/tree/trunk/LICENSE?format=raw".to_owned(),
                    attribution: "Wordnet Bahasa contributors: Francis Bond, David Moeljadi, Hannah Choi, Nurril Hirfana Mohamed Noor, Suerya Sapuan, Muhammad Zulhelmy Mohamed Rosman, Hammam Riza, and Tan Enya Kong".to_owned(),
                    data_scope: "r24 Indonesian I/Y memberships and Indonesian definitions; lower-quality and other-language entries are excluded".to_owned(),
                    file_git_blob_sha1: BTreeMap::from([
                        (
                            "data/msa-wordnet-r24/wn-msa-all.tab".to_owned(),
                            "58467f5edaf355bc1cdcbdcf05d1b22d206eddd1".to_owned(),
                        ),
                        (
                            "data/msa-wordnet-r24/wn-ind-def.tab".to_owned(),
                            "57d213efb21839690ebc2c293577e0e0a81d4e5c".to_owned(),
                        ),
                    ]),
                },
                DatasetProvenance {
                    id: DatasetId::IndoSum,
                    repository_url: "https://drive.google.com/file/d/1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco/view"
                        .to_owned(),
                    revision: "2019-03-20; sha256:34e5c504f711dd8ec0e87dbf10eb1dbaefca86717194f95a14476afeb8810108"
                        .to_owned(),
                    license_id: "CC BY-SA 4.0".to_owned(),
                    license_url: "https://creativecommons.org/licenses/by-sa/4.0/".to_owned(),
                    attribution: "Kemal Kurniawan and Samuel Louvan, IndoSum (2018); source articles attributed in each record"
                        .to_owned(),
                    data_scope: "Five-fold Indonesian news summarization JSONL corpus from Shortir.com"
                        .to_owned(),
                    file_git_blob_sha1: BTreeMap::from([(
                        "indosum.tar.gz".to_owned(),
                        "f2da4682d8172ce268a86575529ef0329fee28c7".to_owned(),
                    ), (
                        "README.txt".to_owned(),
                        "102053e5d471402d501be1e1f82b09f70ebdc6de".to_owned(),
                    )]),
                },
                DatasetProvenance {
                    id: DatasetId::HunspellDictionaries,
                    repository_url: "https://github.com/titoBouzout/Dictionaries".to_owned(),
                    revision: HUNSPELL_REVISION.to_owned(),
                    license_id: "MPL-2.0 OR LGPL-3.0-or-later".to_owned(),
                    license_url: "https://github.com/titoBouzout/Dictionaries/blob/65e552e7216cecab2b497c898a9a7ee1e2e2de11/Indonesia.txt".to_owned(),
                    attribution: "Benitius Brevoort; Indonesia dictionary contributors; titoBouzout/Dictionaries".to_owned(),
                    data_scope: "Indonesia.dic and Indonesia.aff; dual MPLv2/LGPLv3+ per Indonesia.txt".to_owned(),
                    file_git_blob_sha1: BTreeMap::from([
                        ("Indonesia.dic".to_owned(), "224441815bd5914190cc69da08a98d6f203f9f73".to_owned()),
                        ("Indonesia.aff".to_owned(), "ef9da8166455a7211276f55bbb6081b5099d37a6".to_owned()),
                    ]),
                },
            ],
        }
    }

    #[must_use]
    pub fn approved_sources(&self) -> &[DatasetProvenance] {
        &self.approved
    }

    pub fn provenance(&self, id: DatasetId) -> Result<DatasetProvenance, SimError> {
        self.approved
            .iter()
            .find(|source| source.id == id)
            .cloned()
            .ok_or(SimError::InvalidInput {
                parameter: "dataset_license",
                reason: "sumber tidak terdaftar sebagai lisensi terverifikasi",
            })
    }

    pub fn verify(&self, provenance: &DatasetProvenance) -> Result<(), SimError> {
        if !self.approved.contains(provenance) {
            return Err(SimError::InvalidInput {
                parameter: "dataset_license",
                reason: "metadata, versi, lisensi, atribusi, atau scope tidak cocok dengan registry",
            });
        }
        Ok(())
    }

    pub fn supports_semantic_relations(
        &self,
        provenance: &DatasetProvenance,
    ) -> Result<bool, SimError> {
        self.verify(provenance)?;
        Ok(provenance.id == DatasetId::WordNetBahasa)
    }

    pub fn entry_provenance(
        &self,
        id: DatasetId,
        file_path: &str,
        line_number: usize,
    ) -> Result<SourceProvenance, SimError> {
        let dataset = self.provenance(id)?;
        let valid_file = match id {
            DatasetId::UdIndonesianGsd => [
                "id_gsd-ud-train.conllu",
                "id_gsd-ud-dev.conllu",
                "id_gsd-ud-test.conllu",
            ]
            .contains(&file_path),
            DatasetId::UdIndonesianCsui => [
                "id_csui-ud-train.conllu",
                "id_csui-ud-dev.conllu",
                "id_csui-ud-test.conllu",
            ]
            .contains(&file_path),
            DatasetId::UdIndonesianPud => file_path == "id_pud-ud-test.conllu",
            DatasetId::StopwordsIsoId => file_path == "stopwords-id.json",
            DatasetId::WordNetBahasa => dataset.file_git_blob_sha1.contains_key(file_path),
            DatasetId::HunspellDictionaries => {
                matches!(file_path, "Indonesia.dic" | "Indonesia.aff")
            }
            DatasetId::NusaX => dataset.file_git_blob_sha1.contains_key(file_path),
            DatasetId::IndoSum => matches!(file_path, "indosum.tar.gz" | "README.txt"),
            DatasetId::IndoNli => false,
        };
        if !valid_file || file_path.contains("..") || line_number == 0 {
            return Err(SimError::InvalidInput {
                parameter: "dataset_entry",
                reason: "path harus cocok dengan file pada sumber terdaftar dan baris positif",
            });
        }

        Ok(SourceProvenance {
            source_name: dataset.attribution,
            source_uri: format!(
                "{}/blob/{}/{}",
                dataset.repository_url, dataset.revision, file_path
            ),
            license: dataset.license_id,
            license_uri: dataset.license_url,
            source_entry_id: format!("{file_path}:{line_number}"),
        })
    }

    pub fn verify_file(
        &self,
        id: DatasetId,
        file_path: &str,
        contents: &[u8],
    ) -> Result<(), SimError> {
        self.verify_file_reader(id, file_path, contents.len() as u64, Cursor::new(contents))
    }

    pub fn verify_file_reader(
        &self,
        id: DatasetId,
        file_path: &str,
        length: u64,
        mut reader: impl Read,
    ) -> Result<(), SimError> {
        let provenance = self.provenance(id)?;
        let expected =
            provenance
                .file_git_blob_sha1
                .get(file_path)
                .ok_or(SimError::InvalidInput {
                    parameter: "dataset_file",
                    reason: "file tidak tercatat pada revisi sumber yang terverifikasi",
                })?;
        let header = format!("blob {length}\0");
        let mut hasher = sha1::Sha1::new();
        let mut sha256_hasher = sha2::Sha256::new();
        hasher.update(header.as_bytes());
        let mut buffer = [0_u8; 16 * 1024];
        let mut remaining = length;
        while remaining > 0 {
            let limit = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
            let read = reader
                .read(&mut buffer[..limit])
                .map_err(|error| SimError::Io {
                    operation: "verifikasi hash dataset",
                    kind: error.kind(),
                })?;
            if read == 0 {
                return Err(SimError::CorruptData {
                    subsystem: "Indonesian dataset",
                    reason: "file berakhir sebelum ukuran yang tercatat".to_owned(),
                });
            }
            hasher.update(&buffer[..read]);
            sha256_hasher.update(&buffer[..read]);
            remaining -= read as u64;
        }
        let mut extra = [0_u8; 1];
        if reader.read(&mut extra).map_err(|error| SimError::Io {
            operation: "verifikasi ukuran dataset",
            kind: error.kind(),
        })? != 0
        {
            return Err(SimError::CorruptData {
                subsystem: "Indonesian dataset",
                reason: "file lebih panjang dari ukuran yang tercatat".to_owned(),
            });
        }
        let digest = format!("{:x}", hasher.finalize());
        if &digest != expected {
            return Err(SimError::CorruptData {
                subsystem: "Indonesian dataset",
                reason: "Git blob SHA-1 tidak cocok dengan file pada commit terdaftar".to_owned(),
            });
        }
        let sha256 = format!("{:x}", sha256_hasher.finalize());
        super::integrity::verify_source_sha256(id, file_path, length, &sha256)?;
        Ok(())
    }
}

#!/usr/bin/env python3
"""Build a provenance-tracked Indonesian public-law corpus for Shella.

The input manifest is intentionally curated: only the text of Indonesian
legislation is admitted, relying on UU No. 28/2014 Article 42(b). The tool
checks every raw PDF checksum, runs Indonesian OCR, cleans layout artifacts,
removes exact/near duplicates, then splits whole documents/families so no
source document crosses train/validation/test.

Runtime dependencies: pdftoppm, tesseract (Indonesian language data), and
pdfinfo. Python dependencies are standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "data/indonesian_legal_corpus/sources.json"
DEFAULT_OUTPUT = ROOT / "data/indonesian_legal_corpus/prepared_corpus.json"
DEFAULT_STATS = ROOT / "data/indonesian_legal_corpus/corpus_stats.json"
RIGHTS_MARKER = "UU Republik Indonesia Nomor 28 Tahun 2014 Pasal 42(b)"
HEADER_LINES = (
    re.compile(r"^(?:PRESIDEN|REPUBLIK INDONESIA|SALINAN|LEMBARAN NEGARA|TAMBAHAN LEMBARAN NEGARA)(?:\s+REPUBLIK INDONESIA)?$", re.I),
    re.compile(r"^(?:-|–)?\s*\d{1,4}\s*(?:-|–)?$"),
    re.compile(r"^www\.[a-z0-9.-]+$", re.I),
)
SHORT_COMMON_WORDS = {
    "a", "ada", "agar", "aku", "apa", "atas", "bab", "bagi", "bukan", "bila",
    "buat", "dan", "dari", "di", "dia", "hak", "hal", "ia", "ini", "itu", "jika",
    "jo", "ke", "kita", "kpd", "kurang", "lebih", "maka", "masih", "nya", "oleh",
    "pada", "para", "pun", "ri", "saat", "saja", "sama", "satu", "saya", "sejak",
    "serta", "sisi", "soal", "tak", "tiap", "tua", "tuh", "u", "uu", "uud", "untuk",
    "usia", "ya", "yang",
}
SHORT_WORD_RUN = re.compile(
    r"\b(?:([A-Za-zÀ-ÖØ-öø-ÿ]{1,3})[.,;:!?]?\s+){2}"
    r"([A-Za-zÀ-ÖØ-öø-ÿ]{1,3})\b",
    re.I,
)
SHORT_WORD_PAIR = re.compile(
    r"\b([A-Za-zÀ-ÖØ-öø-ÿ]{1,3})[^\w\s]*\s+[^\w\s]*"
    r"([A-Za-zÀ-ÖØ-öø-ÿ]{1,3})\b",
    re.I,
)


def run_checked(command: list[str], *, capture: bool = False, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode:
        detail = (result.stderr or "").strip() if capture else ""
        raise RuntimeError(f"Perintah gagal ({result.returncode}): {' '.join(command)} {detail}")
    return result.stdout or "" if capture else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def require_tools() -> None:
    missing = [name for name in ("pdfinfo", "pdftoppm", "tesseract") if shutil.which(name) is None]
    if missing:
        raise RuntimeError("Utilitas wajib belum tersedia: " + ", ".join(missing))
    languages = run_checked(["tesseract", "--list-langs"], capture=True)
    if "ind" not in languages.split():
        raise RuntimeError("Data bahasa Tesseract 'ind' tidak tersedia (pasang tesseract-ocr-ind).")


def normalize_like_shella(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    output: list[str] = []
    previous_space = True
    for char in text:
        if char.isspace():
            if not previous_space:
                output.append(" ")
                previous_space = True
        elif unicodedata.category(char) != "Cc":
            output.append(char)
            previous_space = False
    if output and output[-1] == " ":
        output.pop()
    return "".join(output)


def clean_ocr_text(raw: str) -> str:
    raw = unicodedata.normalize("NFC", raw.replace("\f", "\n"))
    kept: list[str] = []
    for original_line in raw.splitlines():
        line = re.sub(r"\s+", " ", original_line).strip()
        if not line:
            continue
        if any(pattern.fullmatch(line) for pattern in HEADER_LINES):
            continue
        # Page headers are frequently merged by OCR into one line.
        if re.fullmatch(r"PRESIDEN\s+REPUBLIK\s+INDONESIA", line, re.I):
            continue
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", line)
        if not words:
            continue
        if re.fullmatch(r"[A-Za-z]{1,3}[)\]]?\s+(?:SALINAN|PRESIDEN|REPUBLIK)", line, re.I):
            continue
        if len(words) <= 2 and all(len(word) <= 2 for word in words):
            if not any(word.lower() in SHORT_COMMON_WORDS for word in words):
                continue
        short_words = sum(len(word) <= 2 for word in words)
        if len(words) >= 5 and short_words / len(words) >= 0.8 and sum(map(len, words)) < 35:
            continue
        kept.append(line)
    text = " ".join(kept)
    # Rangkaian lima atau lebih kata sangat pendek hampir selalu artefak scan
    # pada PDF ini; hapus juga bila fragmen menempel pada baris valid.
    text = re.sub(
        r"\b(?:[A-Za-zÀ-ÖØ-öø-ÿ]{1,3}\s+){4,}[A-Za-zÀ-ÖØ-öø-ÿ]{1,3}\b",
        " ",
        text,
        flags=re.I,
    )
    text = SHORT_WORD_RUN.sub(
        lambda match: match.group(0)
        if sum(word.lower() in SHORT_COMMON_WORDS for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", match.group(0))) >= 2
        else " ",
        text,
    )
    text = SHORT_WORD_PAIR.sub(
        lambda match: match.group(0)
        if any(group.lower() in SHORT_COMMON_WORDS for group in match.groups())
        else " ",
        text,
    )
    text = re.sub(
        r"\b([A-Za-zÀ-ÖØ-öø-ÿ]+)(?:\s+\1\b)+",
        r"\1",
        text,
        flags=re.I,
    )
    # Dots between OCR text blocks are scan/layout artifacts, not statute prose.
    text = re.sub(r"(?:\s*\.\s*){2,}", " ", text)
    # Join words split by line-end hyphenation; leave legal hyphens elsewhere.
    text = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])[-‐]\s+(?=[a-zà-öø-ÿ])", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ocr_pdf(pdf_path: Path, dpi: int, language: str) -> tuple[str, int]:
    with tempfile.TemporaryDirectory(prefix="shella-legal-ocr-") as temp_name:
        temp = Path(temp_name)
        prefix = temp / "page"
        run_checked([
            "pdftoppm", "-jpeg", "-gray", "-r", str(dpi), "-jpegopt", "quality=92",
            str(pdf_path), str(prefix),
        ])
        images = sorted(temp.glob("page-*.jpg"))
        if not images:
            raise RuntimeError(f"Tidak ada halaman yang dapat dirender dari {pdf_path}")
        env = os.environ.copy()
        env["OMP_THREAD_LIMIT"] = "1"

        def recognize(image: Path) -> tuple[int, str]:
            number = int(image.stem.rsplit("-", 1)[1])
            text = run_checked([
                "tesseract", str(image), "stdout", "-l", language, "--psm", "4",
            ], capture=True, env=env)
            return number, text

        completed: list[tuple[int, str]] = []
        worker_count = min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(recognize, image) for image in images]
            for count, future in enumerate(as_completed(futures), start=1):
                completed.append(future.result())
                if count % 25 == 0 or count == len(images):
                    print(f"  OCR {pdf_path.name}: {count}/{len(images)} halaman", flush=True)
        completed.sort(key=lambda item: item[0])
        return "\n".join(text for _, text in completed), len(images)


def word_shingle_simhash(text: str) -> int:
    words = re.findall(r"[\w]+", normalize_like_shella(text), flags=re.UNICODE)
    if len(words) < 5:
        shingles = [" ".join(words)] if words else []
    else:
        shingles = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def deduplicate(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    seen_exact: set[str] = set()
    unique: list[dict[str, Any]] = []
    exact_removed = 0
    for document in documents:
        normalized = normalize_like_shella(document["text"])
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if fingerprint in seen_exact:
            exact_removed += 1
            continue
        seen_exact.add(fingerprint)
        document["normalized_text"] = normalized
        document["simhash"] = word_shingle_simhash(normalized)
        unique.append(document)

    # 4 x 16-bit locality-sensitive buckets restrict candidate comparisons.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    retained: list[dict[str, Any]] = []
    near_removed = 0
    for document in unique:
        text = document["normalized_text"]
        signature = document["simhash"]
        candidates: set[int] = set()
        if len(text) >= 1000:
            for band in range(4):
                key = (band, (signature >> (band * 16)) & 0xFFFF)
                candidates.update(buckets.get(key, ()))
        is_duplicate = False
        for prior_index in candidates:
            prior = retained[prior_index]
            prior_text = prior["normalized_text"]
            shorter, longer = sorted((len(text), len(prior_text)))
            if longer and shorter / longer >= 0.80:
                distance = (signature ^ prior["simhash"]).bit_count()
                if distance <= 3:
                    is_duplicate = True
                    break
        if is_duplicate:
            near_removed += 1
            continue
        retained_index = len(retained)
        retained.append(document)
        if len(text) >= 1000:
            for band in range(4):
                key = (band, (signature >> (band * 16)) & 0xFFFF)
                buckets[key].append(retained_index)
    return retained, exact_removed, near_removed


def allocate_group_counts(group_count: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError("Setidaknya dibutuhkan tiga grup dokumen agar train/validasi/test terisi.")
    train = max(1, math.floor(group_count * fractions[0]))
    validation = max(1, math.floor(group_count * fractions[1]))
    test = group_count - train - validation
    if test < 1:
        deficit = 1 - test
        train -= deficit
        test = 1
    if train < 1:
        raise ValueError("Jumlah grup terlalu sedikit untuk proporsi split yang diminta.")
    return train, validation, test


def split_documents(documents: list[dict[str, Any]], seed: int, fractions: tuple[float, float, float]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        groups[document["family_id"]].append(document)
    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    train_count, validation_count, _ = allocate_group_counts(len(group_names), fractions)
    train_groups = set(group_names[:train_count])
    validation_groups = set(group_names[train_count:train_count + validation_count])
    result: dict[str, list[dict[str, Any]]] = {"training": [], "validation": [], "test": []}
    for document in documents:
        group = document["family_id"]
        split = "training" if group in train_groups else "validation" if group in validation_groups else "test"
        result[split].append(document)
    if any(not result[name] for name in result):
        raise ValueError("Split menghasilkan bagian kosong; tambah dokumen/grup sumber.")
    return result


def text_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    texts = [normalize_like_shella(record["text"]) for record in records]
    char_tokens = sum(len(text) for text in texts)
    # Shella CharacterTokenizer::encode_document adds one BOS and one EOS per document.
    return {
        "documents": len(records),
        "model_character_tokens": char_tokens,
        "encoded_tokens_with_bos_eos": char_tokens + 2 * len(records),
        "target_tokens_with_eos": char_tokens + len(records),
        "whitespace_words": sum(len(text.split()) for text in texts),
        "characters": char_tokens,
    }


def load_sources(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not data.get("documents"):
        raise ValueError("sources.json tidak sesuai schema_version=1 atau tidak memuat dokumen.")
    if RIGHTS_MARKER not in data.get("rights_basis", ""):
        raise ValueError(f"Dasar hak tidak memuat marker wajib: {RIGHTS_MARKER}")
    ids: set[str] = set()
    for item in data["documents"]:
        required = ("document_id", "family_id", "title", "detail_url", "download_url", "file", "sha256")
        if any(not item.get(field) for field in required):
            raise ValueError(f"Metadata dokumen tidak lengkap: {item}")
        if item["document_id"] in ids:
            raise ValueError(f"document_id ganda: {item['document_id']}")
        ids.add(item["document_id"])
        if not item["detail_url"].startswith("https://peraturan.bpk.go.id/Details/"):
            raise ValueError(f"URL detail bukan halaman JDIH BPK resmi: {item['detail_url']}")
        if not item["download_url"].startswith("https://peraturan.bpk.go.id/Download/"):
            raise ValueError(f"URL PDF bukan unduhan JDIH BPK resmi: {item['download_url']}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--ocr-dpi", type=int, default=250)
    parser.add_argument("--ocr-language", default="ind+eng")
    parser.add_argument("--min-characters", type=int, default=1000)
    parser.add_argument(
        "--reuse-extracted",
        action="store_true",
        help="membangun ulang split dari hasil OCR tersimpan setelah memverifikasi sumber PDF",
    )
    args = parser.parse_args()

    fractions = (args.train_fraction, args.validation_fraction, args.test_fraction)
    if any(value <= 0 or value >= 1 for value in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        parser.error("Proporsi train/validation/test harus > 0 dan jumlahnya tepat 1.")
    require_tools()
    sources_path = args.sources.resolve()
    inventory = load_sources(sources_path)
    documents: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    extracted_dir = args.output.parent / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    prior_pages: dict[str, int] = {}
    if args.reuse_extracted:
        if not args.stats.is_file():
            parser.error("--reuse-extracted membutuhkan corpus_stats.json dari build OCR sebelumnya")
        prior_stats = json.loads(args.stats.read_text(encoding="utf-8"))
        prior_pages = {
            source["document_id"]: int(source["pages"])
            for source in prior_stats.get("source_documents", [])
        }

    for item in inventory["documents"]:
        pdf_path = (sources_path.parent / item["file"]).resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF sumber belum ada: {pdf_path}")
        actual_hash = sha256_file(pdf_path)
        if actual_hash != item["sha256"]:
            raise ValueError(f"Hash PDF berbeda untuk {item['document_id']}: {actual_hash} != {item['sha256']}")
        extracted_path = extracted_dir / f"{item['document_id']}.txt"
        if args.reuse_extracted:
            if not extracted_path.is_file() or item["document_id"] not in prior_pages:
                raise FileNotFoundError(
                    f"Ekstraksi OCR tersimpan atau page count tidak tersedia untuk {item['document_id']}"
                )
            print(f"Gunakan OCR tersimpan {item['document_id']}", flush=True)
            raw_text = extracted_path.read_text(encoding="utf-8")
            page_count = prior_pages[item["document_id"]]
        else:
            print(f"OCR {item['document_id']} ({pdf_path.stat().st_size:,} byte)", flush=True)
            raw_text, page_count = ocr_pdf(pdf_path, args.ocr_dpi, args.ocr_language)
        cleaned = clean_ocr_text(raw_text)
        cleaned = normalize_like_shella(cleaned)
        if len(cleaned) < args.min_characters:
            raise ValueError(f"Teks terlalu pendek setelah OCR {item['document_id']}: {len(cleaned)} karakter")
        letters = sum(char.isalpha() for char in cleaned)
        alphanumeric = sum(char.isalnum() for char in cleaned)
        alpha_ratio = letters / max(1, alphanumeric)
        if alpha_ratio < 0.55:
            raise ValueError(f"Kualitas OCR rendah untuk {item['document_id']}: rasio huruf/alfanumerik={alpha_ratio:.3f}")
        extracted_path.write_text(cleaned + "\n", encoding="utf-8")
        documents.append({
            "id": len(documents),
            "document_id": item["document_id"],
            "family_id": item["family_id"],
            "title": item["title"],
            "text": cleaned,
        })
        source_reports.append({
            "document_id": item["document_id"],
            "title": item["title"],
            "detail_url": item["detail_url"],
            "download_url": item["download_url"],
            "pdf_sha256": actual_hash,
            "pdf_size_bytes": pdf_path.stat().st_size,
            "pages": page_count,
            "extracted_text_path": str(extracted_path.relative_to(ROOT)),
            "characters_after_cleaning": len(cleaned),
            "whitespace_words": len(cleaned.split()),
            "alphabetic_ratio_in_alphanumeric": round(alpha_ratio, 6),
            "status_at_source": item.get("status_at_source"),
            "subject": item.get("subject"),
        })

    source_record_count = len(documents)
    unique, exact_removed, near_removed = deduplicate(documents)
    splits = split_documents(unique, args.seed, fractions)
    source_manifest_bytes = sources_path.read_bytes()
    source_relpath = os.path.relpath(sources_path, args.output.parent.resolve())
    source_manifest = {
        "dataset_id": inventory["collection_id"],
        "repository_url": "https://peraturan.bpk.go.id/",
        "revision": f"snapshot {inventory['snapshot_date']}; source PDF SHA-256 values are recorded in sources.json",
        "license_id": "Public domain — UU RI No. 28 Tahun 2014 Pasal 42(b), teks undang-undang saja",
        "license_url": inventory["rights_source_url"],
        "attribution": "JDIH BPK dan instansi pemerintah penerbit sesuai URL per dokumen di sources.json; periksa status/perubahan pada halaman sumber.",
        "data_scope": "Teks OCR dari lima undang-undang pemerintah pusat berbahasa Indonesia; tanpa abstrak, komentar, atau putusan pengadilan.",
        "source_split": "Semua dokumen dideduplikasi dahulu lalu dibagi berdasarkan family_id/dokumen; seed deterministik tercatat pada split manifest.",
        "source_file_path": source_relpath,
        "source_file_git_blob_sha1": git_blob_sha1(source_manifest_bytes),
        "source_file_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "source_file_size_bytes": len(source_manifest_bytes),
        "text_field": "extracted_text_path (hasil OCR Tesseract ind+eng, dibersihkan, NFC/lowercase)",
        "excluded_sources": [
            "IndoGovBERT C2 tidak dipakai karena kartu dataset tidak menyatakan lisensi yang spesifik dan data mencampur jenis dokumen.",
            "Abstrak, komentar, metadata naratif, putusan pengadilan, dan teks non-peraturan tidak dimasukkan.",
            "NusaX bukan bagian dari koleksi hukum ini dan tetap dicatat terpisah dengan lisensinya sendiri.",
        ],
    }
    split_ids = {name: [document["id"] for document in splits[name]] for name in ("training", "validation", "test")}
    split_manifest = {
        "seed": args.seed,
        "training_fraction_ppm": round(fractions[0] * 1_000_000),
        "validation_fraction_ppm": round(fractions[1] * 1_000_000),
        "test_fraction_ppm": round(fractions[2] * 1_000_000),
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
        "source_records": source_record_count,
        "training_ids": split_ids["training"],
        "validation_ids": split_ids["validation"],
        "test_ids": split_ids["test"],
    }
    corpus = {
        "training": [{"id": doc["id"], "text": doc["text"]} for doc in splits["training"]],
        "validation": [{"id": doc["id"], "text": doc["text"]} for doc in splits["validation"]],
        "test": [{"id": doc["id"], "text": doc["text"]} for doc in splits["test"]],
        "manifest": {
            "format_version": 1,
            "source": source_manifest,
            "split": split_manifest,
            "training_characters": sum(len(normalize_like_shella(doc["text"])) for doc in splits["training"]),
            "validation_characters": sum(len(normalize_like_shella(doc["text"])) for doc in splits["validation"]),
            "test_characters": sum(len(normalize_like_shella(doc["text"])) for doc in splits["test"]),
            "normalization": "Unicode NFC, lowercase, whitespace collapsed to single spaces, control characters removed; matching Shella CharacterTokenizer.",
            "deduplication": "SHA-256 exact deduplication on normalized text, then word-5-gram SimHash (64-bit, Hamming <=3, length ratio >=0.80); split after deduplication.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stats = {
        "collection_id": inventory["collection_id"],
        "snapshot_date": inventory["snapshot_date"],
        "rights_basis": inventory["rights_basis"],
        "token_definition": "Shella CharacterTokenizer: one Unicode code point after NFC/lowercase/whitespace normalization; model_character_tokens excludes BOS/EOS; encoded_tokens_with_bos_eos includes both for every document.",
        "source_documents": source_reports,
        "documents_before_deduplication": source_record_count,
        "documents_after_deduplication": len(unique),
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
        "split_counts_by_documents": {name: len(splits[name]) for name in splits},
        "split_metrics": {name: text_metrics(splits[name]) for name in splits},
        "total_metrics": text_metrics(unique),
        "prepared_corpus_sha256": sha256_file(args.output),
        "prepared_corpus_path": str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared_corpus": str(args.output), "stats": str(args.stats), "split_metrics": stats["split_metrics"], "total_metrics": stats["total_metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"build_indonesian_legal_corpus: {error}", file=sys.stderr)
        raise SystemExit(2)

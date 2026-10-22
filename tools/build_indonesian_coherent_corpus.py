#!/usr/bin/env python3
"""Build Shella's Indonesian coherent-text corpus from licensed source files.

The pinned input is the official idwiki 2026-07-01 pages-articles XML dump.
This program retains only namespace-0 articles that pass Indonesian language,
paragraph, sentence, and length filters, then combines them with the five
public-law documents already collected in data/indonesian_legal_corpus.

Required packages are pinned in tools/requirements-indonesian-corpus.txt.
The prepared JSON follows CorpusSplit::load_prepared_json and records a
per-document source/revision/license mapping in sources.json.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import heapq
import html
import json
import math
import os
import random
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import langid
import mwparserfromhell

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data/indonesian_coherent_corpus"
LEGAL_DIR = ROOT / "data/indonesian_legal_corpus"
SNAPSHOT = "20260701"
ARCHIVE_NAME = f"idwiki-{SNAPSHOT}-pages-articles-multistream.xml.bz2"
ARCHIVE_URL = f"https://dumps.wikimedia.org/idwiki/{SNAPSHOT}/{ARCHIVE_NAME}"
OFFICIAL_SHA1 = "522c21bfe2bc9bfb8361a0cb6ac43f809c72021e"
DUMP_DIRECTORY_URL = f"https://dumps.wikimedia.org/idwiki/{SNAPSHOT}/"
DUMP_LEGAL_URL = "https://dumps.wikimedia.org/legal.html"
WIKIMEDIA_TERMS_URL = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"
CC_BY_SA_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
GFDL_URL = "https://www.gnu.org/licenses/fdl-1.3.html"
LAW_RIGHTS_URL = "https://www.dgip.go.id/unduhan/download/uu-nomor-28-tahun-2014-tentang-hak-cipta-34-2014"
USER_AGENT = "ShellaIndonesianCorpusBuilder/1.0 (public Wikimedia dump processing)"

EXCLUDED_SECTIONS = {
    "referensi", "rujukan", "catatan kaki", "catatan", "pranala luar",
    "lihat pula", "bacaan lanjut", "daftar pustaka", "sumber", "bibliografi",
    "referensi dan pranala luar", "referensi tambahan",
}
EXPLICIT_RIGHTS_MARKERS = re.compile(
    r"\{\{\s*(?:non[- ]?free|fair[- ]?use|copyvio|pelanggaran[- ]hak[- ]cipta|hapus[- ]cepat)\b",
    re.I,
)
REFERENCE_TAGS = re.compile(
    r"<!--.*?-->|<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>|"
    r"<references\b[^>]*>.*?</references\s*>|<gallery\b[^>]*>.*?</gallery\s*>|"
    r"<(?:math|chem|score|timeline|syntaxhighlight|source|script|style|nowiki)\b[^>]*>.*?</(?:math|chem|score|timeline|syntaxhighlight|source|script|style|nowiki)\s*>",
    re.I | re.S,
)
HEADING = re.compile(r"^(={2,})\s*(.*?)\s*\1\s*$")
SENTENCE_END = re.compile(r"[.!?](?:[\"'”’)]*)\s+(?=[A-ZÀ-ÖØ-Þ0-9])")
WORD = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)


def normalize_text(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFC", text).lower()
    text = " ".join(text.split())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cc")


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(path: Path) -> tuple[str, str, int]:
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            sha1.update(block)
            sha256.update(block)
    return sha1.hexdigest(), sha256.hexdigest(), size


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": USER_AGENT})
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"Unduh {ARCHIVE_URL}", flush=True)
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        received = 0
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
            received += len(block)
            if received // (256 * 1024 * 1024) != (received - len(block)) // (256 * 1024 * 1024):
                total_label = f"/{total:,}" if total else ""
                print(f"  {received:,}{total_label} byte", flush=True)
    temporary.replace(path)


def strip_nonprose_wikitext(wikitext: str) -> str:
    """Remove citation/media/table/list scaffolding before parsing wikitext."""
    text = REFERENCE_TAGS.sub(" ", wikitext)
    text = re.sub(r"\[\[\s*(?:Kategori|Category|Berkas|File):[^\]]*\]\]", " ", text, flags=re.I)
    lines: list[str] = []
    in_table = False
    excluded_level: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        heading = HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            title = re.sub(r"\[\[|\]\]|\*|'", "", heading.group(2)).strip().lower()
            if excluded_level is not None:
                if level <= excluded_level:
                    excluded_level = None
                else:
                    continue
            if title in EXCLUDED_SECTIONS:
                excluded_level = level
                continue
            lines.append("")
            continue
        if excluded_level is not None:
            continue
        if line.startswith("{| ") or line == "{|" or line.startswith("{|"):
            in_table = True
            continue
        if in_table:
            if line.startswith("|}"):
                in_table = False
            continue
        # Lists, navigation, category labels and bare URLs are not continuous prose.
        if line.startswith(("*", "#", ";", "|", "!", "[[Kategori:", "[[Category:")):
            continue
        if re.match(r"^(?:\[\[)?(?:Kategori|Category|Berkas|File):", line, re.I):
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def article_plain_text(wikitext: str) -> list[str]:
    cleaned_wikitext = strip_nonprose_wikitext(wikitext)
    code = mwparserfromhell.parse(cleaned_wikitext)
    plain = str(code.strip_code(normalize=True, collapse=True, keep_template_params=False))
    plain = html.unescape(plain)
    plain = re.sub(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", " ", plain)
    plain = re.sub(r"https?://\S+|www\.\S+", " ", plain, flags=re.I)
    plain = plain.replace("\u200b", " ").replace("\ufeff", " ")
    paragraphs: list[str] = []
    for raw_paragraph in re.split(r"\n\s*\n|\r\n?", plain):
        paragraph = " ".join(raw_paragraph.split()).strip(" \t\r\n|{}")
        if not paragraph:
            continue
        paragraph = re.sub(r"\s+([,.;:!?])", r"\1", paragraph)
        paragraph = re.sub(r"([,.;:!?])(?=\S)(?!\d)", r"\1 ", paragraph)
        words = WORD.findall(paragraph)
        sentence_count = len(re.findall(r"[.!?](?:[\"'”’)]*)\s+", paragraph))
        # Keep substantial prose blocks; isolated labels, list fragments and table cells are dropped.
        if len(paragraph) >= 140 and len(words) >= 22 and (sentence_count >= 1 or len(words) >= 45):
            paragraphs.append(paragraph)
    return paragraphs


def assess_article(title: str, wikitext: str) -> tuple[str | None, dict[str, Any]]:
    if len(wikitext) > 1_000_000:
        return None, {"rejection": "wikitext_over_1MiB"}
    if EXPLICIT_RIGHTS_MARKERS.search(wikitext):
        return None, {"rejection": "explicit_nonfree_or_rights_warning"}
    if re.match(r"^\s*#(?:redirect|alih|pindah)\b", wikitext, re.I):
        return None, {"rejection": "redirect"}
    paragraphs = article_plain_text(wikitext)
    text = normalize_text(" ".join([title + ".", *paragraphs]))
    if len(text.encode("utf-8")) > 1_000_000:
        return None, {"rejection": "prepared_text_over_1MiB", "characters": len(text)}
    if len(text) < 900:
        return None, {"rejection": "too_short", "characters": len(text)}
    words = WORD.findall(text)
    sentences = len(re.findall(r"[.!?](?:[\"'”’)]*)\s+", text)) + int(bool(re.search(r"[.!?]$", text)))
    if len(paragraphs) < 2 or len(words) < 100 or sentences < 5:
        return None, {"rejection": "not_enough_prose", "paragraphs": len(paragraphs), "words": len(words), "sentences": sentences}
    alphanumeric = sum(ch.isalnum() for ch in text)
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(1, alphanumeric)
    if alpha_ratio < 0.75:
        return None, {"rejection": "low_alphabetic_ratio", "alpha_ratio": round(alpha_ratio, 4)}
    unique_ratio = len(set(word.lower() for word in words)) / max(1, len(words))
    if unique_ratio < 0.20:
        return None, {"rejection": "low_vocabulary_diversity", "unique_ratio": round(unique_ratio, 4)}
    language, confidence = langid.classify(text[:5000])
    if language != "id":
        return None, {"rejection": "language_not_indonesian", "language": language, "language_score": round(float(confidence), 3)}
    return text, {
        "language": language,
        "language_score": round(float(confidence), 3),
        "characters": len(text),
        "words": len(words),
        "sentences": sentences,
        "paragraphs": len(paragraphs),
        "alphabetic_ratio": round(alpha_ratio, 4),
        "unique_word_ratio": round(unique_ratio, 4),
    }


def stable_priority(seed: int, page_id: int) -> int:
    payload = f"{seed}:{page_id}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def select_wikipedia_articles(archive: Path, max_documents: int, max_characters: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counters: Counter[str] = Counter()
    heap: list[tuple[int, int, dict[str, Any]]] = []
    namespace = ""
    page_tag = "page"
    print(f"Baca dump XML bz2: {archive}", flush=True)
    with bz2.open(archive, "rb") as stream:
        events = ET.iterparse(stream, events=("start", "end"))
        _, root = next(events)
        namespace = root.tag.split("}", 1)[0] + "}" if "}" in root.tag else ""
        for event, element in events:
            if event != "end" or element.tag != f"{namespace}{page_tag}":
                continue
            counters["pages_seen"] += 1
            page_id_text = element.findtext(f"{namespace}id", default="")
            ns_text = element.findtext(f"{namespace}ns", default="-1")
            if ns_text != "0":
                counters["non_main_namespace"] += 1
                element.clear()
                root.clear()
                continue
            counters["namespace_0_pages"] += 1
            if element.find(f"{namespace}redirect") is not None:
                counters["redirects"] += 1
                element.clear()
                root.clear()
                continue
            title = element.findtext(f"{namespace}title", default="").strip()
            revision = element.find(f"{namespace}revision")
            if revision is None:
                counters["missing_revision"] += 1
                element.clear()
                root.clear()
                continue
            revision_id = revision.findtext(f"{namespace}id", default="")
            timestamp = revision.findtext(f"{namespace}timestamp", default="")
            text_element = revision.find(f"{namespace}text")
            if text_element is None:
                text_element = revision.find(f"{namespace}slots/{namespace}slot/{namespace}text")
            wikitext = text_element.text if text_element is not None and text_element.text else ""
            if not wikitext:
                counters["empty_or_deleted_revision"] += 1
                element.clear()
                root.clear()
                continue
            if len(wikitext) < 1_200:
                counters["wikitext_too_short"] += 1
                element.clear()
                root.clear()
                continue
            text, quality = assess_article(title, wikitext)
            if text is None:
                counters[quality.get("rejection", "quality_rejected")] += 1
                element.clear()
                root.clear()
                continue
            counters["eligible_indonesian_articles"] += 1
            try:
                page_id = int(page_id_text)
                revision_id_int = int(revision_id)
            except ValueError:
                counters["invalid_id"] += 1
                element.clear()
                root.clear()
                continue
            priority = stable_priority(seed, page_id)
            record = {
                "page_id": page_id,
                "revision_id": revision_id_int,
                "timestamp": timestamp,
                "title": title,
                "text": text,
                "quality": quality,
                "selection_priority": priority,
            }
            entry = (-priority, page_id, record)
            if len(heap) < max_documents:
                heapq.heappush(heap, entry)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, entry)
            if counters["pages_seen"] % 25_000 == 0:
                print(
                    f"  halaman {counters['pages_seen']:,}; namespace 0 {counters['namespace_0_pages']:,}; "
                    f"lolos bahasa/kualitas {counters['eligible_indonesian_articles']:,}; "
                    f"tersimpan {len(heap):,}",
                    flush=True,
                )
            element.clear()
            root.clear()
    candidates = [entry[2] for entry in heap]
    candidates.sort(key=lambda item: (item["selection_priority"], item["page_id"]))
    selected: list[dict[str, Any]] = []
    selected_characters = 0
    for candidate in candidates:
        if len(selected) >= max_documents:
            break
        char_count = len(candidate["text"])
        if selected_characters + char_count > max_characters:
            continue
        selected.append(candidate)
        selected_characters += char_count
    counters["selected_after_character_budget"] = len(selected)
    counters["selected_wikipedia_characters"] = selected_characters
    counters["candidate_heap_capacity"] = max_documents
    return selected, dict(counters)


def add_legal_documents(next_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources_path = LEGAL_DIR / "sources.json"
    if not sources_path.is_file():
        return [], []
    inventory = json.loads(sources_path.read_text(encoding="utf-8"))
    documents: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for item in inventory["documents"]:
        extracted = LEGAL_DIR / "extracted" / f"{item['document_id']}.txt"
        pdf = LEGAL_DIR / item["file"]
        if not extracted.is_file() or not pdf.is_file():
            continue
        pdf_sha256 = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if pdf_sha256 != item["sha256"]:
            raise ValueError(f"SHA-256 PDF legal tidak cocok: {item['document_id']}")
        text = normalize_text(extracted.read_text(encoding="utf-8"))
        if len(text) < 1_000 or len(text.encode("utf-8")) > 1_000_000:
            continue
        record_id = next_id + len(documents)
        documents.append({
            "id": record_id,
            "family_id": f"legal:{item['family_id']}",
            "text": text,
            "priority": -1,
            "metadata": {
                "record_id": record_id,
                "source_kind": "indonesian_public_law",
                "document_id": item["document_id"],
                "title": item["title"],
                "language": "id",
                "license_id": "Public domain under Indonesian Copyright Law No. 28/2014 Article 42(b), statutory text only",
                "license_url": LAW_RIGHTS_URL,
                "source_url": item["detail_url"],
                "download_url": item["download_url"],
                "raw_file_path_from_repo_root": str((LEGAL_DIR / item["file"]).relative_to(ROOT)),
                "raw_file_sha256": item["sha256"],
                "extracted_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "attribution": "JDIH BPK and issuing government institution; text of law only",
                "included_in_prepared_corpus": True,
                "deduplication": None,
            },
        })
        metadata.append(documents[-1]["metadata"])
    return documents, metadata


def simhash_words(text: str) -> int:
    words = [word.lower() for word in WORD.findall(text)]
    if len(words) < 5:
        shingles = [" ".join(words)] if words else []
    else:
        shingles = [" ".join(words[index:index + 5]) for index in range(len(words) - 4)]
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def deduplicate_documents(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    exact: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    exact_removed = 0
    for document in documents:
        digest = hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
        if digest in exact:
            document["metadata"]["included_in_prepared_corpus"] = False
            document["metadata"]["deduplication"] = {"kind": "exact", "canonical_record_id": exact[digest]["id"]}
            exact_removed += 1
            continue
        document["text_sha256"] = digest
        document["simhash"] = simhash_words(document["text"])
        exact[digest] = document
        candidates.append(document)

    buckets: dict[tuple[int, int], list[int]] = {}
    retained: list[dict[str, Any]] = []
    near_removed = 0
    for document in candidates:
        text = document["text"]
        signature = document["simhash"]
        possible: set[int] = set()
        if len(text) >= 1_000:
            for band in range(4):
                key = (band, (signature >> (band * 16)) & 0xFFFF)
                possible.update(buckets.get(key, ()))
        duplicate_of: dict[str, Any] | None = None
        for index in possible:
            prior = retained[index]
            prior_text = prior["text"]
            ratio = min(len(text), len(prior_text)) / max(len(text), len(prior_text))
            if ratio >= 0.80 and (signature ^ prior["simhash"]).bit_count() <= 3:
                duplicate_of = prior
                break
        if duplicate_of is not None:
            document["metadata"]["included_in_prepared_corpus"] = False
            document["metadata"]["deduplication"] = {"kind": "near", "canonical_record_id": duplicate_of["id"]}
            near_removed += 1
            continue
        index = len(retained)
        retained.append(document)
        if len(text) >= 1_000:
            for band in range(4):
                key = (band, (signature >> (band * 16)) & 0xFFFF)
                buckets.setdefault(key, []).append(index)
    return retained, exact_removed, near_removed


def metrics(documents: list[dict[str, Any]]) -> dict[str, int]:
    char_tokens = sum(len(document["text"]) for document in documents)
    return {
        "documents": len(documents),
        "model_character_tokens": char_tokens,
        "encoded_tokens_with_bos_eos": char_tokens + 2 * len(documents),
        "target_tokens_with_eos": char_tokens + len(documents),
        "whitespace_words": sum(len(document["text"].split()) for document in documents),
        "characters": char_tokens,
    }


def prepare_legal_sources_metadata() -> dict[str, Any] | None:
    path = LEGAL_DIR / "sources.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive.resolve()
    if not archive.is_file():
        if args.download:
            download_archive(archive)
        else:
            raise FileNotFoundError(f"Arsip dump belum ada: {archive}; gunakan --download atau unduh URL resmi.")
    actual_sha1, archive_sha256, archive_size = file_hashes(archive)
    print(f"Arsip {archive_size:,} byte; SHA-1 {actual_sha1}; SHA-256 {archive_sha256}", flush=True)
    if actual_sha1 != OFFICIAL_SHA1:
        raise ValueError(f"SHA-1 arsip tidak cocok dengan checksum resmi: {actual_sha1} != {OFFICIAL_SHA1}")

    wiki_docs, scan_stats = select_wikipedia_articles(archive, args.max_documents, args.max_characters, args.seed)
    legal_docs, _ = add_legal_documents(0)
    # Laws precede wiki pages so a duplicated statute is retained with its clearer government provenance.
    for index, document in enumerate(legal_docs):
        document["id"] = index
        document["metadata"]["record_id"] = index
    documents = legal_docs
    for item in wiki_docs:
        record_id = len(documents)
        page_url = f"https://id.wikipedia.org/w/index.php?curid={item['page_id']}"
        revision_url = f"https://id.wikipedia.org/w/index.php?oldid={item['revision_id']}"
        wiki_document = {
            "id": record_id,
            "family_id": f"wikipedia:id:{item['page_id']}",
            "text": item["text"],
            "priority": item["selection_priority"],
            "metadata": {
                "record_id": record_id,
                "source_kind": "wikipedia_indonesian_article",
                "dataset_id": "idwiki",
                "snapshot_date": SNAPSHOT,
                "page_id": item["page_id"],
                "revision_id": item["revision_id"],
                "revision_timestamp": item["timestamp"],
                "title": item["title"],
                "language": "id",
                "language_detector": "langid 1.1.6; classify(text[:5000]) == id",
                "license_id": "Wikimedia community text: CC BY-SA 4.0 and GFDL where applicable; page/import exceptions require source review",
                "license_url": DUMP_LEGAL_URL,
                "terms_url": WIKIMEDIA_TERMS_URL,
                "cc_by_sa_url": CC_BY_SA_URL,
                "gfdl_url": GFDL_URL,
                "article_url": page_url,
                "revision_url": revision_url,
                "attribution_url": page_url,
                "attribution_method": "URL to article/history, which lists contributors; per Wikimedia Terms of Use. Preserve import notices and extra attribution if present.",
                "content_modified": True,
                "modification_notice": "Wikitext templates, tables, references, lists, headings and non-prose blocks were filtered; whitespace normalized; the article was not fact-checked.",
                "rights_review_status": "Default Wikimedia text license recorded; per-page imported-material and fair-use exceptions are not exhaustively resolved. Skip any explicit rights-warning template; human legal review remains necessary before redistribution/model release.",
                "quality": item["quality"],
                "text_sha256": hashlib.sha256(item["text"].encode("utf-8")).hexdigest(),
                "included_in_prepared_corpus": True,
                "deduplication": None,
            },
        }
        documents.append(wiki_document)

    source_records = len(documents)
    unique, exact_removed, near_removed = deduplicate_documents(documents)
    if len(unique) < 3:
        raise ValueError("Korpus setelah deduplikasi harus memiliki minimal tiga dokumen.")
    ordered = list(unique)
    random.Random(args.seed).shuffle(ordered)
    train_count = max(1, math.floor(len(ordered) * args.train_fraction))
    validation_count = max(1, math.floor(len(ordered) * args.validation_fraction))
    test_count = len(ordered) - train_count - validation_count
    if test_count < 1:
        train_count -= 1 - test_count
        test_count = 1
    training = ordered[:train_count]
    validation = ordered[train_count:train_count + validation_count]
    test = ordered[train_count + validation_count:]
    if not training or not validation or not test:
        raise ValueError("Proporsi split menghasilkan bagian kosong.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "prepared_corpus.json"
    stats_path = args.output_dir / "corpus_stats.json"
    sources_path = args.output_dir / "sources.json"
    sample_path = args.output_dir / "sample_review.json"

    source_documents = [document["metadata"] for document in documents]
    source_manifest_json = {
        "schema_version": 1,
        "collection_id": "shella-indonesian-coherent-2026-10-22",
        "collection_title": "Korpus prosa Indonesia terkurasi Shella",
        "snapshot_date": "2026-10-22",
        "language": "id",
        "scope": "Teks artikel ensiklopedis Indonesia namespace utama dari dump bertanggal dan lima teks undang-undang Indonesia yang sudah dihimpun; teks saja, tanpa media.",
        "archive": {
            "file_name": ARCHIVE_NAME,
            "download_url": ARCHIVE_URL,
            "snapshot_directory": DUMP_DIRECTORY_URL,
            "official_sha1": OFFICIAL_SHA1,
            "verified_sha1": actual_sha1,
            "sha256": archive_sha256,
            "size_bytes": archive_size,
            "dump_snapshot": SNAPSHOT,
        },
        "licenses": {
            "wikipedia_text": {
                "license_id": "CC BY-SA 4.0 and GFDL for original Wikimedia text where applicable; some imported content may be CC BY-SA-only or have extra terms.",
                "license_url": DUMP_LEGAL_URL,
                "terms_url": WIKIMEDIA_TERMS_URL,
                "cc_by_sa_url": CC_BY_SA_URL,
                "gfdl_url": GFDL_URL,
                "attribution_rule": "URL to article/history or stable attributed copy, or author list; preserve import notices and additional attribution. Modified/redistributed text must carry compatible license notice and indicate changes.",
                "limitations": "Wikimedia says dumps may contain unrecognized copyright infringements and fair-use exceptions. Explicit rights-warning templates are excluded, but each page's import history and exceptions are not exhaustively reviewed. Model/weights implications require legal review.",
            },
            "indonesian_laws": {
                "license_id": "Public domain under Indonesian Copyright Law No. 28/2014 Article 42(b), statutory text only.",
                "license_url": LAW_RIGHTS_URL,
                "limitations": "Does not cover PDF layout, images, abstracts, commentary or all government-site content; source audit is in the original legal corpus inventory.",
            },
            "combined_collection": "Per-document rights apply as listed below. CC BY-SA obligations apply to the Wikipedia text and modified text/data portions; the legal statute text remains public-domain under the stated Indonesian-law basis. Do not infer that trained weights have a settled license status.",
        },
        "selection_policy": {
            "wikipedia_namespace": 0,
            "redirects_excluded": True,
            "text_filters": ["language id detected by langid", "at least 900 normalized characters", "at least two substantial prose paragraphs", "at least 100 words and five sentences", "alphabetic ratio >= 0.75", "unique word ratio >= 0.20", "maximum 1 MiB per prepared record", "explicit nonfree/fair-use/copyvio warning templates excluded"],
            "sampling": f"Deterministic BLAKE2b priority by page ID with seed {args.seed}; bounded to {args.max_documents} Wikipedia articles and {args.max_characters} character-tokens.",
            "deduplication": "Exact SHA-256 after normalization, then word 5-gram SimHash (64-bit, Hamming <=3, length ratio >=0.80).",
            "split": f"Document-level, after deduplication, seed {args.seed}; proportions train={args.train_fraction}, validation={args.validation_fraction}, test={args.test_fraction}. Each article remains one record; an article does not cross splits.",
            "excluded_sources": ["SIBI textbooks pending explicit text/model-training rights", "Project Gutenberg Indonesian catalog showed zero entries tagged id in the source audit", "Tatoeba isolated sentences excluded from the prose core", "Wikisource/Wikibooks omitted from this snapshot pending per-work license and text-quality review", "all non-main namespaces, media, raw metadata, and non-prose page structures"],
        },
        "documents": source_documents,
    }
    sources_bytes = (json.dumps(source_manifest_json, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    sources_path.write_bytes(sources_bytes)

    corpus_source = {
        "dataset_id": "shella-indonesian-coherent-2026-10-22",
        "repository_url": DUMP_DIRECTORY_URL,
        "revision": f"idwiki dump snapshot {SNAPSHOT}; archive SHA-1 {actual_sha1}; per-article revision IDs in sources.json",
        "license_id": "Mixed per-document rights; CC BY-SA 4.0/GFDL Wikimedia article text and Indonesian statutory public-domain text; see sources.json",
        "license_url": DUMP_LEGAL_URL,
        "attribution": "For each Wikipedia record, use its article/history URL and preserve import notices and extra attribution. For law records, attribute JDIH BPK and issuing institution. Full mapping is in sources.json.",
        "data_scope": "Filtered Indonesian Wikipedia namespace-0 article prose plus five Indonesian statutes; no images, templates, talk pages or unrelated metadata.",
        "source_split": "Deduplicated and split at whole-document/article level; no article crosses train/validation/test.",
        "source_file_path": "sources.json",
        "source_file_git_blob_sha1": git_blob_sha1(sources_bytes),
        "source_file_sha256": hashlib.sha256(sources_bytes).hexdigest(),
        "source_file_size_bytes": len(sources_bytes),
        "text_field": "CorpusRecord.text; source article revision and per-document license/attribution are mapped by record_id in sources.json",
        "excluded_sources": source_manifest_json["selection_policy"]["excluded_sources"],
    }
    split_manifest = {
        "seed": args.seed,
        "training_fraction_ppm": round(args.train_fraction * 1_000_000),
        "validation_fraction_ppm": round(args.validation_fraction * 1_000_000),
        "test_fraction_ppm": round(args.test_fraction * 1_000_000),
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
        "source_records": source_records,
        "training_ids": [doc["id"] for doc in training],
        "validation_ids": [doc["id"] for doc in validation],
        "test_ids": [doc["id"] for doc in test],
    }
    corpus = {
        "training": [{"id": doc["id"], "text": doc["text"]} for doc in training],
        "validation": [{"id": doc["id"], "text": doc["text"]} for doc in validation],
        "test": [{"id": doc["id"], "text": doc["text"]} for doc in test],
        "manifest": {
            "format_version": 1,
            "source": corpus_source,
            "split": split_manifest,
            "training_characters": sum(len(doc["text"]) for doc in training),
            "validation_characters": sum(len(doc["text"]) for doc in validation),
            "test_characters": sum(len(doc["text"]) for doc in test),
            "normalization": "Unicode NFC; lowercase; collapse whitespace; remove control characters; matches Shella CharacterTokenizer normalization.",
            "deduplication": "Exact SHA-256 after normalization, then word 5-gram SimHash (64-bit, Hamming <=3, length ratio >=0.80); document-level split occurs after deduplication.",
        },
    }
    encoded = (json.dumps(corpus, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if len(encoded) > 512 * 1024 * 1024:
        raise ValueError(f"Prepared JSON {len(encoded):,} byte exceeds Shella loader's 512 MiB limit.")
    output_path.write_bytes(encoded)

    splits = {"training": training, "validation": validation, "test": test}
    stats = {
        "collection_id": source_manifest_json["collection_id"],
        "snapshot_date": source_manifest_json["snapshot_date"],
        "archive": source_manifest_json["archive"],
        "token_definition": "Shella CharacterTokenizer: one Unicode code point after NFC/lowercase/whitespace normalization; model_character_tokens excludes BOS/EOS; encoded_tokens_with_bos_eos includes both per document.",
        "selection_and_filtering": scan_stats,
        "quality_filters": source_manifest_json["selection_policy"]["text_filters"],
        "documents_before_deduplication": source_records,
        "documents_after_deduplication": len(unique),
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
        "split_counts_by_documents": {name: len(part) for name, part in splits.items()},
        "split_metrics": {name: metrics(part) for name, part in splits.items()},
        "total_metrics": metrics(unique),
        "source_inventory_sha256": hashlib.sha256(sources_bytes).hexdigest(),
        "prepared_corpus_sha256": hashlib.sha256(encoded).hexdigest(),
        "prepared_corpus_size_bytes": len(encoded),
        "prepared_corpus_path": str(output_path.relative_to(ROOT)),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    review_docs = sorted(
        (doc for doc in unique if doc["metadata"].get("source_kind") == "wikipedia_indonesian_article"),
        key=lambda doc: doc["priority"],
    )[:50]
    review = {
        "purpose": "Human-readable audit sample; these are real corpus records with source attribution.",
        "articles": [
            {
                "record_id": doc["id"],
                "title": doc["metadata"]["title"],
                "article_url": doc["metadata"]["article_url"],
                "revision_url": doc["metadata"]["revision_url"],
                "quality": doc["metadata"]["quality"],
                "excerpt": doc["text"][:700],
            }
            for doc in review_docs
        ],
    }
    sample_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "prepared_corpus": str(output_path),
        "sources": str(sources_path),
        "stats": str(stats_path),
        "sample_review": str(sample_path),
        "archive_sha1": actual_sha1,
        "scan_stats": scan_stats,
        "split_metrics": stats["split_metrics"],
        "total_metrics": stats["total_metrics"],
        "prepared_corpus_size_bytes": len(encoded),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/tmp/shella-wikipedia") / ARCHIVE_NAME)
    parser.add_argument("--download", action="store_true", help="unduh arsip snapshot resmi jika belum tersedia")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-documents", type=int, default=20_000)
    parser.add_argument("--max-characters", type=int, default=100_000_000)
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    args = parser.parse_args()
    if args.max_documents < 3 or args.max_characters < 10_000:
        parser.error("--max-documents minimal 3 dan --max-characters minimal 10000")
    fractions = (args.train_fraction, args.validation_fraction, args.test_fraction)
    if any(value <= 0 or value >= 1 for value in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        parser.error("proporsi train/validation/test harus positif dan berjumlah tepat 1")
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, ET.ParseError) as error:
        print(f"build_indonesian_coherent_corpus: {error}", file=sys.stderr)
        raise SystemExit(2)

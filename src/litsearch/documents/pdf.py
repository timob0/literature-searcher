from __future__ import annotations

import re
from pathlib import Path

import fitz

from litsearch.documents.models import DocumentPage, ExtractionResult, Paragraph, Sentence


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r"(?<=[A-Za-z])\n(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])-(?=\n)", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def detect_section(title: str | None) -> tuple[str, str | None]:
    if not title:
        return ("other", None)
    normalized = re.sub(r"\s+", " ", title.strip().lower())
    mapping = {
        "abstract": "abstract",
        "introduction": "introduction",
        "literature review": "literature_review",
        "literature": "literature_review",
        "theory": "theory",
        "methods": "methods",
        "method": "methods",
        "methodology": "methods",
        "materials and methods": "methods",
        "research design": "methods",
        "results": "results",
        "discussion": "discussion",
        "limitations": "limitations",
        "conclusion": "conclusion",
        "concluding remarks": "conclusion",
        "references": "references",
        "reference": "references",
        "bibliography": "references",
        "literaturverzeichnis": "references",
        "literatur": "references",
        "références": "references",
        "methoden": "methods",
        "methodik": "methods",
        "forschungsdesign": "methods",
    }
    for key, value in mapping.items():
        if key in normalized:
            return value, title.strip()
    return ("other", title.strip())


def sentence_splitter(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z\(\[\"\'])", text)
    return [part.strip() for part in chunks if part.strip()]


def extract_pdf_text(pdf_path: str) -> ExtractionResult:
    path = Path(pdf_path)
    result = ExtractionResult(
        attachment_key="",
        parent_key="",
        citation_key=None,
        pdf_path=str(path),
    )
    try:
        document = fitz.open(str(path))
    except Exception as exc:  # pragma: no cover - fallback path
        result.error = f"Unable to open PDF: {exc}"
        result.requires_ocr = True
        return result

    pages: list[DocumentPage] = []
    page_count = document.page_count
    for page_index in range(page_count):
        page = document.load_page(page_index)
        raw_text = page.get_text("text")
        text = normalize_text(raw_text)
        paragraphs = [
            Paragraph(
                paragraph_id=f"p{page_index + 1}:{idx + 1}",
                page_number=page_index + 1,
                section_type="other",
                section_title=None,
                text=para.strip(),
                ordering=idx,
            )
            for idx, para in enumerate(re.split(r"\n\s*\n+", text))
            if para.strip()
        ]
        if not paragraphs and text:
            paragraphs = [
                Paragraph(
                    paragraph_id=f"p{page_index + 1}:1",
                    page_number=page_index + 1,
                    section_type="other",
                    section_title=None,
                    text=text,
                    ordering=0,
                )
            ]
        for para in paragraphs:
            sentences = []
            for sentence_idx, sentence_text in enumerate(sentence_splitter(para.text)):
                sentence_id = f"{para.paragraph_id}:s{sentence_idx + 1}"
                section_type, section_title = detect_section(None)
                sentences.append(
                    Sentence(
                        sentence_id=sentence_id,
                        paragraph_id=para.paragraph_id,
                        page_number=para.page_number,
                        section_type=section_type,
                        section_title=section_title,
                        text=sentence_text,
                        normalized_text=normalize_text(sentence_text),
                        ordering=sentence_idx,
                    )
                )
            para.sentences = sentences
            para.section_type, para.section_title = detect_section(None)
        pages.append(DocumentPage(page_number=page_index + 1, text=text, paragraphs=paragraphs))
    result.pages = pages
    result.raw_stats = {"text_characters": sum(len(page.text) for page in pages), "pages": page_count}
    if not any(page.text.strip() for page in pages):
        result.requires_ocr = True
        result.error = "No extractable text found" 
    return result

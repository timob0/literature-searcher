from __future__ import annotations

import re

# Facets where front matter (preface/introduction) is still useful evidence.
FRONT_MATTER_ALLOWED_FACETS = {"topic", "problem_or_purpose"}

_COPYRIGHT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"all rights reserved",
        r"library of congress",
        r"\bisbn\b",
        r"©\s*\d{4}",
        r"no part of this (book|publication)",
        r"printed in the united states",
        r"cataloging-in-publication data",
        r"national library of \w+ cataloguing",
    ]
]

_DISCLAIMER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"is sold with the understanding",
        r"may not be suitable for your (situation|particular circumstances)",
        r"(publisher|author)s? (is|are) not (engaged|responsible) in (rendering|providing)",
        r"if (legal|expert|professional) assistance is required",
        r"consult a competent professional",
        r"neither the publisher nor the author",
    ]
]

_BIOGRAPHY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"about the author",
        r"\b(he|she|they) (received|holds|earned) (his|her|their) (ph\.?d|doctorate|m\.?a\.?|degree)",
        r"\bis (a|an) (professor|senior lecturer|research fellow|consultant) (of|at|in)",
        r"\bhas (written|published|authored) (numerous|several|many) (books|articles|papers)",
        r"lives in .{0,40} with (his|her|their)",
    ]
]

_MARKETING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"praise for",
        r"acclaim for",
        r"one of the (most|best)[- ]?(selling|influential)",
        r"\bbestseller\b",
        r"\bnational bestseller\b",
        r"advance praise",
    ]
]

_TOC_HEADING_PATTERN = re.compile(r"\b(chapter|part)\s+(\d+|[ivxlc]+)\b", re.IGNORECASE)
_BLURB_ATTRIBUTION_PATTERN = re.compile(r"[—-]\s*[A-Z][a-zA-Z.]+ [A-Z][a-zA-Z.]+,")
_BIBLIOGRAPHY_ENTRY_PATTERN = re.compile(r"[A-Z][a-zA-Z'-]+,\s?[A-Z]\.\s?(&|and)?\s?")
_INDEX_ENTRY_PATTERN = re.compile(r"^\s*\S[\w\s'-]*,\s*\d+(,\s*\d+)*\s*$", re.MULTILINE)


def classify_boilerplate(text: str) -> str | None:
    """Classify obvious front/back-matter noise so it can be excluded from summary evidence.

    Returns a short category label (e.g. "copyright", "disclaimer", "toc",
    "biography", "marketing", "references", "index") or None if the text looks
    like substantive content.
    """
    if not text or not text.strip():
        return None

    if any(pattern.search(text) for pattern in _DISCLAIMER_PATTERNS):
        return "disclaimer"
    if any(pattern.search(text) for pattern in _COPYRIGHT_PATTERNS):
        return "copyright"
    if any(pattern.search(text) for pattern in _BIOGRAPHY_PATTERNS):
        return "biography"
    if any(pattern.search(text) for pattern in _MARKETING_PATTERNS) or len(_BLURB_ATTRIBUTION_PATTERN.findall(text)) >= 2:
        return "marketing"
    if len(_TOC_HEADING_PATTERN.findall(text)) >= 3:
        return "toc"
    if len(_INDEX_ENTRY_PATTERN.findall(text)) >= 4:
        return "index"
    if len(_BIBLIOGRAPHY_ENTRY_PATTERN.findall(text)) >= 4:
        return "references"
    return None


def is_boilerplate_for_facet(text: str, facet: str) -> bool:
    """Decide whether a candidate passage should be excluded from a facet's evidence pool."""
    category = classify_boilerplate(text)
    if category is None:
        return False
    if category == "toc" and facet in FRONT_MATTER_ALLOWED_FACETS:
        # A table of contents can still hint at scope/topics/purpose; let Stage 1 interpret it.
        return False
    return True

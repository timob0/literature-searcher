from __future__ import annotations


def format_citation_with_pages(citation_key: str, pages: list[int]) -> str:
    """Format a citation key with a compact page range, e.g. 'key (pp. 3-5, 10)'."""
    sorted_pages = sorted(set(pages))
    if not sorted_pages:
        return citation_key
    ranges: list[str] = []
    start = end = sorted_pages[0]
    for page in sorted_pages[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append(str(start) if start == end else f"{start}-{end}")
        start = end = page
    ranges.append(str(start) if start == end else f"{start}-{end}")
    label = "p." if len(sorted_pages) == 1 else "pp."
    return f"{citation_key} ({label} {', '.join(ranges)})"

from __future__ import annotations


def strip_json_fences(content: str) -> str:
    """Remove a wrapping ```json / ``` markdown code fence some models add despite being asked for raw JSON."""
    cleaned = content.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        return cleaned[7:-3].strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        return cleaned[3:-3].strip()
    return cleaned

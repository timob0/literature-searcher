from litsearch.retrieval.context import ContextExpander
from litsearch.retrieval.models import ChunkHit


def hit(index: int, text: str, attachment: str = "att-1", score: float = 0.5) -> ChunkHit:
    return ChunkHit(f"{attachment}:{index}:hash", attachment, "paper-1", "paper-1", text=text, score=score)


def test_expands_adjacent_chunks_without_crossing_documents() -> None:
    anchor = hit(1, "middle")
    evidence = ContextExpander([hit(0, "before"), anchor, hit(2, "after"), hit(1, "other", "att-2")]).expand(anchor)

    assert evidence.context_chunk_ids == ["att-1:0:hash", "att-1:1:hash", "att-1:2:hash"]
    assert evidence.text == "before middle after"
    assert evidence.anchor_score == 0.5


def test_continues_split_bullet_list() -> None:
    chunks = [
        hit(10, "Three perspectives on culture are commonly distinguished:"),
        hit(11, "• Functional: Culture is the way people solve problems."),
        hit(12, "• Interpretive: Culture concerns shared meanings."),
        hit(13, "• Cognitive: Culture reflects shared knowledge structures."),
    ]

    evidence = ContextExpander(chunks).expand(chunks[1], before=1, after=1)

    assert "Three perspectives" in evidence.text
    assert "Functional" in evidence.text
    assert "Interpretive" in evidence.text
    assert "Cognitive" in evidence.text


def test_removes_boundary_overlap() -> None:
    chunks = [
        hit(20, "Organisational culture consists of shared values and assumptions that"),
        hit(21, "shared values and assumptions that shape how members interpret events."),
    ]

    evidence = ContextExpander(chunks).expand(chunks[0], before=0, after=1)

    assert evidence.text == "Organisational culture consists of shared values and assumptions that shape how members interpret events."


def test_respects_token_budget_and_keeps_anchor() -> None:
    chunks = [hit(index, "word " * 20) for index in range(3)]

    evidence = ContextExpander(chunks).expand(chunks[1], max_tokens=25, continue_structures=False)

    assert evidence.anchor_chunk_id == chunks[1].chunk_id
    assert len(evidence.text.split()) <= 25
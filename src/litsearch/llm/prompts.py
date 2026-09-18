from __future__ import annotations


STRUCTURED_OUTPUT_RULES = """Use only the information provided in the user messages.
Return JSON matching the requested schema exactly.
Do not invent citations, page numbers, authors, methods, or findings.
If the task cannot be completed from the supplied evidence, use the schema's
explicit insufficient-evidence or fallback fields when available."""


def evidence_excerpt(text: str, max_chars: int = 4000) -> str:
    """Keep oversized indexed passages within the local model context budget."""
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return f"{text[:head_chars]}\n[...middle omitted...]\n{text[-tail_chars:]}"


def health_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_RULES},
        {"role": "user", "content": "Return status='ok' and message='ready'."},
    ]


def examples_messages(concept: str, evidence: list[tuple[str, str]]) -> list[dict[str, str]]:
    passages = "\n\n".join(f"[{evidence_id}]\n{evidence_excerpt(passage)}" for evidence_id, passage in evidence)
    instructions = f"""Classify each passage as evidence about concrete examples of: {concept}

A concrete example describes an observable behaviour, practice, intervention, event,
manifestation, or implementation. A general association or citation-only discussion
is not an example. Do not infer an example that is not stated. The concept name does
not need to appear literally. Academic papers may spell out a term followed by its
abbreviation in parentheses or brackets; treat that as the same concept when the
passage describes concrete examples of it. Return one assessment for every evidence ID.

Evidence passages:
{passages}"""
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_RULES},
        {"role": "user", "content": instructions},
    ]


def examples_extract_messages(concept: str, evidence: list[tuple[str, str]], *, max_chars: int = 2200) -> list[dict[str, str]]:
    passages = "\n\n".join(f"[{key}]\n{evidence_excerpt(text, max_chars)}" for key, text in evidence)
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_RULES},
        {"role": "user", "content": f"""Requested concept: {concept}

Extract every distinct concrete example of the requested concept from the evidence.
A concrete example is an observable behaviour, practice, policy, event, decision,
interaction, intervention, organizational situation, or manifestation. Return the
example itself, not the surrounding passage. One evidence block may contain many
examples. Do not return definitions, general discussion, methodology, citations, or
background. Reject examples of related concepts. Preserve important contrasts such
as what an organization says versus what it actually rewards or does. Use only the
supplied evidence and return concise statements with short source excerpts.

Evidence blocks:
{passages}"""},
    ]


def definitions_messages(
    concept: str,
    evidence: list[tuple[str, str]],
    *,
    explanation: bool = False,
    evidence_max_chars: int = 4000,
) -> list[dict[str, str]]:
    passages = "\n\n".join(f"[{evidence_id}]\n{evidence_excerpt(passage, evidence_max_chars)}" for evidence_id, passage in evidence)
    valid_ids = ", ".join(evidence_id for evidence_id, _ in evidence)
    task = "Explain the requested concept using the supplied context. Preserve lists, distinctions, dimensions, and complementary characteristics." if explanation else "Classify each passage for whether it defines or explains the meaning of the REQUESTED CONCEPT: {concept}"
    instructions = f"""{task}

explicit_definition means the passage directly states what the concept means.
implicit_definition means it explains the concept's meaning, dimensions, or defining
characteristics without a formal definition phrase. conceptual_discussion discusses
the concept without defining it. not_definition is unrelated or unusable.
Set requested_concept to exactly {concept}. Set concept_match to requested_concept
only when the passage defines or explains that requested concept. A passage defining
a related concept, such as health literacy when the requested concept is
health-oriented leadership, must be different_concept and not_definition.
Academic papers often spell out a term followed by its abbreviation in parentheses
or brackets, then explain what the term means with wording such as "understood as",
"refers to", or a dash/colon definition. Treat that as a valid definition when it
refers to the requested concept. The abbreviation and parenthetical wording are part
of the supplied evidence, not a new citation.
Copy or minimally extract the definition from the passage; do not write a new one.
The only valid evidence_id values are: {valid_ids}
Use those IDs exactly. Never use a citation key, title, author name, or any other
identifier as evidence_id. Return one assessment for every evidence ID.

Evidence passages:
{passages}"""
    if explanation:
        instructions += "\nFor conceptual discussion, identify the main meanings, characteristics, dimensions, components, perspectives, or mechanisms described. Do not invent unsupported points."
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_RULES},
        {"role": "user", "content": instructions},
    ]


def definition_evidence_messages(concept: str, evidence: list[tuple[str, str]], *, max_chars: int = 2600) -> list[dict[str, str]]:
    passages = "\n\n".join(f"[{evidence_id}]\n{evidence_excerpt(text, max_chars)}" for evidence_id, text in evidence)
    valid_ids = ", ".join(evidence_id for evidence_id, _ in evidence)
    instructions = f"""Requested concept: {concept}

Determine whether each supplied evidence block actually defines or clearly explains
the meaning of THAT requested concept. Do not classify a passage as a definition
because it is topically related, gives an example, describes an effect, or defines
a different concept.

Set concept_match to exactly one of these three values: requested_concept (the
passage is about the requested concept itself), different_concept (the passage
defines or discusses a different concept instead), or uncertain (unclear whether
the passage concerns the requested concept). Do not use any other value.

Use explicit_definition only for a direct definition. Use implicit_definition when
the passage clearly states the requested concept's essential meaning or defining
characteristics without a formal definition formula. Use conceptual_characterisation
for useful conceptual information that does not define the concept. Use not_definition
otherwise. Copy only source-supported consecutive wording into extracted_text; do not
write a synthetic definition. Academic abbreviations and parenthetical spellings are
part of the source concept. The only valid evidence IDs are: {valid_ids}.

Return one assessment per evidence ID. Use only the supplied evidence.

Evidence:
{passages}"""
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_RULES},
        {"role": "user", "content": instructions},
    ]


FACET_LABELS = {
    "topic": "TOPICS",
    "problem_or_purpose": "PROBLEM / PURPOSE",
    "methodology_or_approach": "APPROACH / METHODOLOGY",
    "finding_or_contribution": "FINDINGS / CONTRIBUTIONS",
    "limitation": "LIMITATIONS",
    "future_research": "FUTURE RESEARCH",
}

_FACET_RULES = {
    "topic": """Extract the document's substantive subject areas as concise conceptual phrases
(e.g. "levels of organizational culture", "leadership's role in culture change").
Do not return table-of-contents dumps, chapter/part number lists, publisher
descriptions, copyright text, biography, or reference-list content. If the evidence
is a table of contents or scope statement, convert it into a small number of concise
topics rather than reproducing it.""",
    "problem_or_purpose": """Extract explicit aims, stated purposes, research questions, or motivating
problems in the author's own words (e.g. "The purpose of this study...", "This book
aims..."). Do not confuse publisher marketing or back-cover copy with the author's
actual stated purpose.""",
    "methodology_or_approach": """Extract only genuine methodological or approach evidence: study design, sample or
population, data source, collection method, analysis method, time frame for empirical
work; or conceptual approach, theoretical basis, evidence basis, case/clinical basis,
observational approach, or analytical strategy for conceptual/theoretical/practitioner
work. Author biography, client lists, and credentials are NOT methodology, even when
mentioned near methodological claims.""",
    "finding_or_contribution": """Extract the document's own major results, conclusions, propositions, conceptual
models, arguments, or practical implications. Distinguish the author's own
contribution from findings the author merely cites from other researchers' studies;
only include cited external findings if the passage frames them as directly
supporting the author's own argument. Do not treat publisher praise, jacket-copy
claims ("regarded as one of the most influential...") or endorsements as findings
unless the authorial text itself substantiates the claim.""",
    "limitation": """Extract ONLY limitations that the author explicitly attributes to their own study,
method, evidence, model, theory, argument, applicability, or scope. Do not infer a
generic limitation merely because academic summaries usually contain one. Publisher
legal/liability disclaimers (e.g. "the advice ... may not be suitable for your
situation") are NOT academic limitations and must be rejected even if they use
words like "limitation" or "may not apply". Set explicitly_stated=true only when the
author directly states the limitation.""",
    "future_research": """Extract ONLY explicit authorial statements recommending future research or
further work (e.g. "future research should...", "this remains to be investigated").
Do not transform an interesting unresolved question into a future-research
recommendation unless the text itself frames it that way.""",
}


def summary_extract_messages(facet: str, document_title: str, evidence: list[tuple[str, str]], *, max_chars: int = 2200) -> list[dict[str, str]]:
    passages = "\n\n".join(f"[{key}]\n{evidence_excerpt(text, max_chars)}" for key, text in evidence)
    valid_ids = ", ".join(key for key, _ in evidence)
    label = FACET_LABELS.get(facet, facet)
    instructions = f"""You are extracting evidence for a concise academic document summary.

Facet: {label}
Document: {document_title}

Extract only statements that genuinely provide evidence for the requested facet.
Do not summarize the entire passage. Return discrete, concise propositions. Ignore
unrelated surrounding context. Use only the supplied document evidence. Do not use
general knowledge. Do not infer generic academic content that is not explicitly
supported.

{_FACET_RULES.get(facet, "")}

Every point's evidence_ids field must be a non-empty list containing the exact
bracketed evidence ID(s) it is drawn from, copied verbatim. The only valid evidence
IDs are: {valid_ids}. Set facet="{facet}" on every point. Return zero points when the
evidence does not support this facet.

Evidence blocks:
{passages}"""
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_RULES},
        {"role": "user", "content": instructions},
    ]


def summary_synthesize_messages(
    document_title: str,
    document_type: str,
    citation_key: str,
    points_by_facet: dict[str, list[tuple[str, str]]],
) -> list[dict[str, str]]:
    sections = []
    for facet, points in points_by_facet.items():
        label = FACET_LABELS.get(facet, facet)
        if not points:
            sections.append(f"{label}: no validated evidence.")
            continue
        body = "\n".join(f"[{point_id}] {statement}" for point_id, statement in points)
        sections.append(f"{label}:\n{body}")
    evidence_block = "\n\n".join(sections)
    instructions = f"""Create a concise academic summary of the selected document using ONLY the
validated evidence supplied below.

Document: {document_title}
Citation key: {citation_key}
Document type: {document_type}

Your job is synthesis, not concatenation. Do not reproduce raw source passages or
describe retrieved chunks. Combine overlapping evidence. Prioritize the document's
main contribution rather than attempting to mention everything. Distinguish
empirical findings from conceptual arguments where appropriate. Do not invent
methodology, limitations, or future research beyond what the evidence supports.
If a facet lacks sufficient validated evidence, set insufficient_evidence=true for
that section instead of inventing content. Every substantive claim must retain
supporting evidence IDs (the bracketed point IDs below). Classify the overall
methodology_classification field using only the supplied methodology evidence; use
"not_stated" if the evidence does not clearly support a classification. Aim for
approximately 300-600 words in total across all sections. The overview should be one
compact paragraph of about 80-150 words. Return about 3-5 topics, 3-6 findings or
contributions, and 0-3 limitations and future-research points, each with supporting
evidence IDs.

Validated evidence by facet:
{evidence_block}"""
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_RULES},
        {"role": "user", "content": instructions},
    ]


def explain_extract_messages(concept: str, evidence: list[tuple[str, str]], *, max_chars: int = 2200) -> list[dict[str, str]]:
    passages = "\n\n".join(f"[{key}]\n{evidence_excerpt(text, max_chars)}" for key, text in evidence)
    valid_ids = ", ".join(key for key, _ in evidence)
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_RULES},
        {"role": "user", "content": f"""Requested concept: {concept}

Extract only conceptual evidence that explains the requested concept's meaning,
characteristics, components, dimensions, levels, mechanisms, boundaries, perspectives,
or models. Ignore anecdotes, case narratives, methodology, generic mentions,
historical background, citations, and examples unless essential to clarify an abstract
conceptual point. Do not summarize each evidence block. Extract discrete propositions,
preserve complete conceptual lists, and return zero points when evidence is irrelevant.

Every point's evidence_ids field must be a non-empty list containing the exact
bracketed evidence ID(s) it is drawn from, copied verbatim. The only valid evidence
IDs are: {valid_ids}. A point with no evidence_ids is discarded, so never leave
evidence_ids empty for a point you include.

Evidence blocks:
{passages}"""},
    ]


def explain_synthesize_messages(concept: str, points: list[ConceptualEvidencePoint]) -> list[dict[str, str]]:
    evidence = "\n\n".join(
        f"[{point.point_id}] ({point.type})\n{point.statement}\nSource: {point.source_excerpt}"
        for point in points
    )
    valid_ids = ", ".join(point.point_id for point in points)
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_RULES},
        {"role": "user", "content": f"""Requested concept: {concept}

Synthesize a concise conceptual explanation from the structured evidence points below.
This is synthesis, not concatenation: combine overlapping ideas, produce a concise
overview, and provide 3-7 integrated points covering characteristics, components,
dimensions, mechanisms, boundaries, or perspectives where supported. Do not describe
sources sequentially, reproduce long excerpts, or add anecdotes. Preserve genuinely
different theoretical perspectives rather than forcing agreement.

Every substantive point's evidence_ids field, including the overview's, must be a
non-empty list containing the exact point ID(s) it is built from, copied verbatim.
The only valid point IDs are: {valid_ids}. Return the overview as an object with
statement and evidence_ids. Use only these evidence points.

A poor answer paraphrases each source one after another. A good answer identifies the
underlying propositions and explains the concept coherently.

Structured evidence:
{evidence}"""},
    ]
from __future__ import annotations

import streamlit as st

from litsearch.retrieval.definitions import DefinitionSearchStrategy
from litsearch.retrieval.examples import ExampleSearchStrategy
from litsearch.retrieval.paper_summary import PaperSummaryStrategy
from litsearch.retrieval.retriever import SimpleRetriever

st.set_page_config(page_title="Academic Literature Search", layout="wide")
st.title("Academic Literature Search")

mode = st.selectbox("Mode", ["Search", "Define", "Examples", "Summarize Paper"])
query = st.text_input("Query", "supportive leadership practices")

if mode == "Search":
    st.info("Search mode is prepared for the next retrieval layer; the initial app keeps the search logic in the Python service layer.")
elif mode == "Define":
    retriever = SimpleRetriever()
    strategy = DefinitionSearchStrategy(retriever=retriever)
    if retriever.embedding_provider is None or retriever.vector_store is None:
        st.warning("No retrieval backend is available. Start Chroma and ensure the embedding model can load.")
    else:
        results = strategy.search(query, top_k=3)
        if not results:
            st.write("No definition results found.")
        else:
            for idx, hit in enumerate(results, start=1):
                st.subheader(f"{idx}. {hit.paper_title or 'Unknown paper'}")
                st.write(hit.definition_text)
                st.caption(f"{hit.citation_key or 'Unknown citation'} • Page {hit.page or '-'} • Score {hit.score:.3f}")
elif mode == "Examples":
    retriever = SimpleRetriever()
    strategy = ExampleSearchStrategy(retriever=retriever)
    if retriever.embedding_provider is None or retriever.vector_store is None:
        st.warning("No retrieval backend is available. Start Chroma and ensure the embedding model can load.")
    else:
        results = strategy.search(query, top_k=3)
        if not results:
            st.write("No example results found.")
        else:
            for idx, hit in enumerate(results, start=1):
                st.subheader(f"{idx}. {hit.paper_title or 'Unknown paper'}")
                st.write(hit.example_text)
                st.caption(f"{hit.citation_key or 'Unknown citation'} • Page {hit.page or '-'} • Score {hit.score:.3f}")
else:
    retriever = SimpleRetriever()
    strategy = PaperSummaryStrategy(retriever=retriever)
    if retriever.embedding_provider is None or retriever.vector_store is None:
        st.warning("No retrieval backend is available. Start Chroma and ensure the embedding model can load.")
    else:
        result = strategy.summarize(query)
        summary = result.summary
        if result.insufficient_evidence and not result.stage1_points:
            st.write("No summary evidence was found.")
        else:
            st.subheader(result.title or result.citation_key or query)
            for name, heading in (
                ("overview", "Overview"), ("topics", "Topics"), ("problem_or_purpose", "Purpose"),
                ("methodology_or_approach", "Approach / Methodology"), ("findings_or_contributions", "Key findings / contributions"),
                ("limitations", "Limitations"), ("future_research", "Future research / directions"),
            ):
                section = getattr(summary, name)
                st.markdown(f"**{heading}**")
                if section.prose:
                    st.write(section.prose)
                for point in section.points:
                    st.write(f"- {point.statement}")
                if not section.prose and not section.points:
                    st.write("Not explicitly stated in the retrieved evidence.")

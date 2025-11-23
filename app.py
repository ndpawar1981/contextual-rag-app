# app.py
import os
import time
from pathlib import Path
from typing import List

import streamlit as st

from loader import build_vectorstore_from_pdfs, load_vectorstore
from retriever import get_similarity_retriever
from generator import (
    make_basic_rag_chain,
    make_rag_with_sources_chain,
    make_rag_with_citations_chain,
)
from config import (
    OPENAI_API_KEY,
    VECTOR_DB_DIR,
    RETRIEVER_K,
)


# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Contextual RAG Assistant",
    layout="wide",
)

# Nice global CSS touch
st.markdown(
    """
<style>
/* Center the title a bit and give it some class */
#contextual-rag-assistant {
    text-align: center;
}

/* Chat message styling */
.stChatMessage {
    border-radius: 16px;
    padding: 12px 16px;
}

/* Make sidebar fonts a tad nicer */
[data-testid="stSidebar"] * {
    font-size: 0.95rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------
# Helper: streaming display
# -------------------------
def stream_text(text: str, delay: float = 0.01):
    """
    Simple character streaming utility for classy typing effect.
    """
    placeholder = st.empty()
    full_text = ""
    for ch in text:
        full_text += ch
        placeholder.markdown(full_text)
        time.sleep(delay)


# -------------------------
# Sidebar: configuration + indexing
# -------------------------
st.sidebar.title("Settings & Indexing")

# API key handling
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=OPENAI_API_KEY or "",
    type="password",
    help="Will override environment variable for this app session.",
)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Mode selection
mode = st.sidebar.radio(
    "Response mode",
    ["Answer only", "Answer + sources", "Answer + citations"],
)

top_k = st.sidebar.slider("Top-K documents", min_value=1, max_value=10, value=RETRIEVER_K)

st.sidebar.markdown("---")
st.sidebar.subheader("Build / Refresh Index")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

index_status = st.sidebar.empty()

if st.sidebar.button("Build / Refresh Vector Index") and uploaded_files:
    index_status.info("Building index... this may take a bit.")

    # Save uploaded PDFs to a temp folder
    data_dir = Path("uploaded_docs")
    data_dir.mkdir(exist_ok=True)

    pdf_paths: List[Path] = []
    for f in uploaded_files:
        save_path = data_dir / f.name
        with open(save_path, "wb") as out:
            out.write(f.read())
        pdf_paths.append(save_path)

    # Build vectorstore
    vectorstore = build_vectorstore_from_pdfs(pdf_paths, persist_directory=VECTOR_DB_DIR)
    index_status.success(f"Vector index built with {len(pdf_paths)} file(s).")


# -------------------------
# Load existing vectorstore if present
# -------------------------
vectorstore_available = Path(VECTOR_DB_DIR).exists() and any(
    Path(VECTOR_DB_DIR).iterdir()
)

if vectorstore_available:
    vectorstore = load_vectorstore(VECTOR_DB_DIR)
else:
    vectorstore = None


# -------------------------
# Chat session state
# -------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts: {"role": "user"/"assistant", "content": str, ...}


st.title("Contextual RAG Assistant")
st.caption(
    "Ask questions about your uploaded PDFs. "
    "Answers are retrieved from a context-aware vector index built with LangChain + Chroma + OpenAI."
)

# Show existing chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Show citations"):
                for c in msg["citations"]:
                    st.markdown(
                        f"- **ID**: `{c.id}` • **Source**: `{c.source}` • "
                        f"**Title**: {c.title} • **Page**: {c.page}\n\n"
                        f"  > {c.quotes}"
                    )
        if msg.get("sources"):
            with st.expander("Show source documents"):
                for i, d in enumerate(msg["sources"], start=1):
                    m = d.metadata
                    st.markdown(
                        f"**Source {i}** — `{m.get('source')}` · "
                        f"Title: *{m.get('title')}* · Page: {m.get('page')}"
                    )
                    st.write(d.page_content[:1000] + ("..." if len(d.page_content) > 1000 else ""))


# -------------------------
# Main chat input
# -------------------------
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
    elif vectorstore is None:
        st.error("No vector index found. Upload PDFs and click 'Build / Refresh Vector Index' first.")
    else:
        # 1. Show user message
        st.session_state["messages"].append(
            {"role": "user", "content": user_query}
        )
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Build retriever + chain based on mode
        retriever = get_similarity_retriever(vectorstore, k=top_k)

        if mode == "Answer only":
            chain = make_basic_rag_chain(retriever)
        elif mode == "Answer + sources":
            chain = make_rag_with_sources_chain(retriever)
        else:
            chain = make_rag_with_citations_chain(retriever)

        # 3. Run chain
        with st.chat_message("assistant"):
            if mode == "Answer only":
                answer = chain.invoke(user_query)
                stream_text(answer)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer}
                )

            elif mode == "Answer + sources":
                result = chain.invoke(user_query)
                answer = result["answer"]
                docs = result["context"]

                stream_text(answer)
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": docs,
                    }
                )

            else:  # Answer + citations
                result = chain.invoke(user_query)
                answer = result["answer"]
                cited = result["citations"]  # QuotedCitations model
                citations_list = cited.citations

                stream_text(answer)

                # Also show citations right away under the streaming text
                if citations_list:
                    st.markdown(" ")
                    st.markdown("**Citations:**")
                    for c in citations_list:
                        st.markdown(
                            f"- **ID**: `{c.id}` • **Source**: `{c.source}` • "
                            f"**Title**: {c.title} • **Page**: {c.page}\n\n"
                            f"  > {c.quotes}"
                        )

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations_list,
                    }
                )

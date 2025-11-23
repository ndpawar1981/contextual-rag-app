# retriever.py
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config import RETRIEVER_K


def get_similarity_retriever(
    vectorstore: Chroma,
    k: int = RETRIEVER_K,
) -> VectorStoreRetriever:
    """
    Return a similarity retriever on top of a Chroma vector store.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever

# generator.py
from __future__ import annotations

from typing import List, Dict, Any

from operator import itemgetter

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_openai import ChatOpenAI

from config import CHAT_MODEL


# -------------------------
# Pydantic models for citations and structured output
# -------------------------
class Citation(BaseModel):
    id: str = Field(
        description="The string ID of a specific context article which justifies the answer."
    )
    source: str = Field(
        description="The source/path of the specific context article which justifies the answer."
    )
    title: str = Field(
        description="The title of the specific context article which justifies the answer."
    )
    page: int = Field(
        description="The page number of the specific context article which justifies the answer."
    )
    quotes: str = Field(
        description=(
            "The verbatim sentences from the context article that are used to generate the answer."
        )
    )


class QuotedCitations(BaseModel):
    """
    Quote citations from given context articles that can be used
    to justify the generated answer. Can be multiple articles.
    """

    citations: List[Citation] = Field(
        description=(
            "Citations (can be multiple) from the context articles that justify the answer."
        )
    )


# -------------------------
# Shared prompts
# -------------------------
RAG_PROMPT = """You are an assistant who is an expert in question-answering tasks.
Answer the following question using only the retrieved context.
If the answer is not in the context, say that you don't know.
Keep the answer detailed and well formatted.

Question:
{question}

Context:
{context}

Answer:
"""

CITATIONS_PROMPT = """You are an assistant who is an expert in analyzing answers to questions
and finding referenced citations from context articles.

Given the question, the context articles, and the generated answer,
analyze the answer and quote citations from the context articles
that justify the answer.

Question:
{question}

Context Articles:
{context}

Answer:
{answer}
"""

rag_prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)
cite_prompt_template = ChatPromptTemplate.from_template(CITATIONS_PROMPT)


def _get_chat_llm() -> ChatOpenAI:
    return ChatOpenAI(model_name=CHAT_MODEL, temperature=0)


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    prepend metadata so the citing LLM can reference IDs.
    """
    formatted_docs = []
    for doc in docs:
        m = doc.metadata
        formatted_docs.append(
            f"""Context Article ID: {m.get('id')}
Context Article Source: {m.get('source')}
Context Article Title: {m.get('title')}
Context Article Page: {m.get('page')}

Content:
{doc.page_content}
"""
        )
    return "\n\n---\n\n".join(formatted_docs)


# -------------------------
# Basic RAG (answer only)
# -------------------------
def make_basic_rag_chain(retriever) -> RunnablePassthrough:
    llm = _get_chat_llm()

    qa_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )
    return qa_chain


# -------------------------
# RAG with sources (answer + raw docs)
# -------------------------
def make_rag_with_sources_chain(retriever) -> RunnablePassthrough:
    llm = _get_chat_llm()

    rag_response_chain = (
        {
            "context": itemgetter("context") | RunnableLambda(format_docs),
            "question": itemgetter("question"),
        }
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )

    rag_chain_w_sources = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=rag_response_chain)
    )

    # Output shape: {"context": [Document,...], "question": "...", "answer": "..." }
    return rag_chain_w_sources


# -------------------------
# RAG with citations (answer + structured citations)
# -------------------------
def make_rag_with_citations_chain(retriever) -> RunnablePassthrough:
    llm = _get_chat_llm()
    structured_llm = llm.with_structured_output(QuotedCitations)

    rag_response_chain = (
        {
            "context": itemgetter("context") | RunnableLambda(format_docs_with_metadata),
            "question": itemgetter("question"),
        }
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )

    cite_response_chain = (
        {
            "context": itemgetter("context") | RunnableLambda(format_docs_with_metadata),
            "question": itemgetter("question"),
            "answer": itemgetter("answer"),
        }
        | cite_prompt_template
        | structured_llm
    )

    rag_chain_w_citations = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=rag_response_chain)
        | RunnablePassthrough.assign(citations=cite_response_chain)
    )

    # Output shape:
    # {
    #   "context": [Document,...],
    #   "question": "...",
    #   "answer": "...",
    #   "citations": QuotedCitations(...)
    # }
    return rag_chain_w_citations

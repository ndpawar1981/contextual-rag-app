# loader.py
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Ensure key is on env for langchain_openai
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -------------------------
# LLM-based contextualizer
# -------------------------
def _get_context_llm() -> ChatOpenAI:
    return ChatOpenAI(model_name=CHAT_MODEL, temperature=0)


def generate_chunk_context(document: str, chunk: str) -> str:
    """
    Given the full paper/document + a chunk, generate a short context string
    describing how this chunk fits into the overall doc.
    """
    chunk_process_prompt = """
    You are an AI assistant specializing in research/document analysis.
    Your task is to provide brief, relevant context for a chunk of text
    based on the full document.

    Here is the full document:
    <paper>
    {paper}
    </paper>

    Here is the chunk:
    <chunk>
    {chunk}
    </chunk>

    In 2–3 sentences, explain:
    - Where this chunk fits conceptually in the document (e.g., intro, methods, results, summary)
    - The main topic or idea of this chunk
    - How it relates to the overall document

    Keep it concise, helpful, and neutral in tone.

    Context:
    """

    prompt_template = ChatPromptTemplate.from_template(chunk_process_prompt)
    llm = _get_context_llm()
    agentic_chunk_chain = prompt_template | llm | StrOutputParser()

    context = agentic_chunk_chain.invoke({"paper": document, "chunk": chunk})
    return context.strip()


# -------------------------
# Document loading + chunking
# -------------------------
def load_pdf(file_path: str | Path) -> List[Document]:
    file_path = str(file_path)
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def create_contextual_chunks(
    file_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    1. Load PDF pages
    2. Split into chunks
    3. For each chunk, generate LLM-based context and prepend it to the text
    4. Return enriched Documents with metadata (id, page, source, title)
    """
    file_path = str(file_path)
    print(f"Loading pages: {file_path}")
    doc_pages = load_pdf(file_path)

    print("Combining full document text for context...")
    original_doc_text = "\n\n".join(page.page_content for page in doc_pages)

    print("Chunking pages:", file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    doc_chunks = splitter.split_documents(doc_pages)

    contextual_chunks: List[Document] = []
    print("Generating contextual chunks:", file_path)

    for chunk in doc_chunks:
        chunk_content = chunk.page_content
        meta = chunk.metadata

        chunk_metadata = {
            "id": str(uuid.uuid4()),
            "page": meta.get("page", 0),
            "source": meta.get("source", file_path),
            "title": Path(meta.get("source", file_path)).name,
        }

        context = generate_chunk_context(original_doc_text, chunk_content)

        contextual_chunks.append(
            Document(
                page_content=context + "\n\n" + chunk_content,
                metadata=chunk_metadata,
            )
        )

    print("Finished processing:", file_path)
    return contextual_chunks


# -------------------------
# Vector DB builder/loader
# -------------------------
def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore_from_pdfs(
    file_paths: List[str | Path],
    persist_directory: str = VECTOR_DB_DIR,
) -> Chroma:
    """
    Given PDF file paths, create contextual chunks, embed them, and
    persist them into a Chroma DB.
    """
    all_docs: List[Document] = []
    for fp in file_paths:
        contextual_docs = create_contextual_chunks(fp)
        all_docs.extend(contextual_docs)

    embedding_model = get_embedding_model()

    print("Building Chroma DB...")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        collection_name="my_context_db",
        embedding=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=persist_directory,
    )

    print("Chroma DB created & persisted to:", persist_directory)
    return vectorstore


def load_vectorstore(
    persist_directory: str = VECTOR_DB_DIR,
) -> Chroma:
    """
    Load an existing Chroma DB from disk.
    """
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name="my_context_db",
        embedding_function=embedding_model,
    )
    return vectorstore

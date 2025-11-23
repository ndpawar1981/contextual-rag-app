<h1 align="center">🧠 Contextual RAG Assistant</h1>

<p align="center">
  <b>A Modular, Context-Aware Retrieval Augmented Generation System Built with LangChain, ChromaDB, OpenAI & Streamlit</b>
  <br/>
  <sub>Featuring contextual chunking, citations, sources, and a beautiful streaming Chat UI.</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/Framework-LangChain-green" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-red" />
  <img src="https://img.shields.io/badge/VectorDB-ChromaDB-purple" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

---

## 🚀 Overview

**Contextual RAG Assistant** is a full-stack, production-ready RAG (Retrieval Augmented Generation) application that:

✔ Loads and processes PDFs  
✔ Generates **context-enriched chunks using an LLM**  
✔ Stores embeddings in **ChromaDB**  
✔ Retrieves top-K relevant chunks  
✔ Generates answers with **citations**, **sources**, and **justifications**  
✔ Provides a **modern Streamlit UI** with a **typing / character stream effect**  
✔ Fully modular — loader, retriever, generator, and UI separated into clean Python modules  

This project converts a notebook into a scalable codebase suitable for real production RAG systems.

---

## 📁 Project Structure

📦 rag_app/
│
├── app.py # Streamlit UI with streaming chat output
├── config.py # Centralized configuration (models, paths, chunk size, etc.)
├── loader.py # Document loader + contextual chunk generator + vector store builder
├── retriever.py # ChromaDB retriever wrapper
├── generator.py # RAG pipelines: basic, with sources, with citations
│
├── requirements.txt # All dependencies
└── README.md # This file


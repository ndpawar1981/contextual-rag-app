<h1 align="center">🧠 Contextual RAG Assistant</h1>
🚀 **Live Demo:** [Try the app here](https://ndpawar81-contextual-rag-app.streamlit.app/)
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


---

## 📁 Project Structure

```

📦 rag_app/
│
├── app.py                 # Streamlit UI with streaming chat output
├── config.py              # Centralized configuration (models, paths, chunk size, etc.)
├── loader.py              # Document loader + contextual chunk generator + vector store builder
├── retriever.py           # ChromaDB retriever wrapper
├── generator.py           # RAG pipelines: basic, with sources, with citations
│
├── requirements.txt       # All dependencies
└── README.md              # This file

````

---

## ✨ Features

### 🔍 Contextual Chunk Generation  
Chunks are not just split — they are enhanced by an LLM that explains:

- Where the chunk fits in the document  
- Its role (intro/methods/analysis/etc.)  
- Why it is useful  

→ This drastically improves retrieval quality.

---

### 📚 Multiple RAG Modes

| Mode | Description |
|------|-------------|
| **Answer Only** | Just the generated answer |
| **Answer + Sources** | Returns exact chunks used to answer |
| **Answer + Citations** | Structured citations with page, title, quote |

---

### 🎨 Beautiful Streamlit UI

- Character-by-character message streaming  
- Chat-style interface using `st.chat_message`  
- Sidebar for:  
  - API key  
  - Uploading PDFs  
  - Rebuilding vector index  
  - Selecting retrieval top-K  
  - Selecting output mode  

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
````

---

### 2️⃣ Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**Mac / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add your OpenAI key

Either:

#### Option A — Add `.env` file

```
OPENAI_API_KEY=sk-xxxx
```

or

#### Option B — Paste API key into sidebar in the Streamlit app.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open your browser at:

```
http://localhost:8501
```

---

## 📂 Uploading and Indexing Documents

1. Upload one or more PDFs
2. Click **"Build / Refresh Vector Index"**
3. The system will:

   * Load PDFs
   * Generate contextual chunks
   * Embed them
   * Save them into ChromaDB
4. Ask any question in the chat!

---

## 🧩 Modular Architecture

### **loader.py**

* Loads PDF pages
* Splits them using `RecursiveCharacterTextSplitter`
* Calls the LLM to generate contextual descriptions
* Creates ChromaDB vector index

---

### **retriever.py**

* Wraps a similarity retriever over ChromaDB
* Acts as the retrieval pipeline for all RAG modes

---

### **generator.py**

Contains 3 RAG pipelines:

1. **Basic RAG**
2. **RAG + Sources**
3. **RAG + Structured Citations** (Pydantic model)

---

### **app.py**

* Streamlit chat UI
* Real-time typing animation
* File uploader + index builder
* Chat history persistence

## 🤝 Contributing

Pull requests are welcome!

If you'd like to:

* Improve UI
* Add agentic tools
* Add telemetry
* Add reranking
* Add Docker support

Just open an issue.

---

## 📜 License

MIT License — free for personal & commercial use.

---

## ⭐ Support the Project

If you found this project helpful…

**Please ⭐ star the repo!**

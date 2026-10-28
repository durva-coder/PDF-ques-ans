# 📚 RAG-based PDF Question Answering System (SmartPDF AI)

A smart, interactive web app that lets you **upload any PDF** and ask **natural language questions** about its content. Built using **Retrieval-Augmented Generation (RAG)** architecture, this tool blends semantic search with generative AI to give grounded, context-aware answers.

---

## 🚀 Features

- 🔍 **Retrieval-Augmented Generation (RAG)** pipeline using LangChain
- 🧠 Embeddings via **sentence-transformers (MiniLM-L6-v2)**
- ⚡ **FAISS** vector index for fast semantic search
- 🤖 Text generation using **Hugging Face Transformers** (DialoGPT)
- 🧾 Supports multiple PDF loaders (PyMuPDF, pdfplumber, PyPDF2)
- 🌐 **Gradio** web interface for easy user interaction

---

## 🧠 How It Works

1. **PDF Ingestion**  
   Upload a `.pdf` file → automatically parsed using robust fallback methods.

2. **Text Splitting & Embedding**  
   Text is chunked using `RecursiveCharacterTextSplitter` and embedded using `MiniLM-L6-v2`.

3. **Vector Store Creation**  
   FAISS is used to store the document chunks as vectors for efficient similarity search.

4. **Question Answering via RAG**  
   - User asks a question.  
   - Relevant chunks are retrieved.  
   - A HuggingFace LLM generates the final answer, grounded in the retrieved context.

---

## 🛠️ Tech Stack

| Component       | Library/Tool                     |
|----------------|----------------------------------|
| Text Splitting | `langchain.text_splitter`        |
| Embeddings     | `sentence-transformers`          |
| Vector Store   | `FAISS`                          |
| Language Model | `HuggingFace Transformers`       |
| Interface      | `Gradio`                         |
| PDF Parsing    | `PyMuPDF`, `pdfplumber`, `PyPDF2`|

---

## 📦 Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

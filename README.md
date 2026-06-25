# RAG Document Q&A API

A production-ready RAG (Retrieval-Augmented Generation) system that answers questions based on company documents only.

## Tech Stack
- FastAPI (REST API)
- LangChain (RAG Pipeline)
- ChromaDB (Vector Database)
- Hugging Face Sentence Transformers (Embeddings)
- Ollama / Llama3.2 (Local LLM)

## Features
- Document ingestion and chunking
- Semantic search using vector embeddings
- Context-aware answer generation
- Out-of-scope question handling

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Run ingestion: python3 ingest.py
3. Start API: uvicorn main:app --reload
4. Visit: http://localhost:8000/docs

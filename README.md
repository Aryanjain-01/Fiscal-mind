# Fiscal Mind

**A Retrieval-Augmented Generation system for financial report analysis.**

An end-to-end Retrieval-Augmented Generation project for analyzing annual reports, earnings notes, and finance filings. It runs locally with Python's standard library and uses a transparent TF-IDF retriever, so you can demo the full RAG workflow without setting up a vector database or paid API.

## What It Does

- Upload or paste finance report text.
- Optionally upload text-based PDFs if `PyMuPDF` or `pypdf` is installed.
- Chunk reports into overlapping passages.
- Build a local retrieval index.
- Answer finance questions using retrieved evidence.
- Show citations, similarity scores, source chunks, and detected finance metrics.
- Optionally use OpenAI for stronger answer synthesis when `OPENAI_API_KEY` is set.

## Project Flow

```text
Report upload
  -> text extraction
  -> cleaning
  -> overlapping chunks
  -> TF-IDF vectors
  -> top-k retrieval
  -> finance metric extraction
  -> grounded answer with source chunks
```

## Run Locally

```bash
python3 app.py
```

Then open:

```text
http://127.0.0.1:8000
```

Click `Sample` to load two demo reports, then ask:

- What drove revenue growth and what risks should investors watch?
- Compare profitability between the companies.
- Which company has stronger free cash flow?
- What balance sheet risks are mentioned?

## Optional PDF Support

For PDF extraction, install one of these:

```bash
python3 -m pip install pymupdf
```

or:

```bash
python3 -m pip install pypdf
```

Text-based PDFs work best. Scanned PDFs need OCR, which is intentionally outside this lightweight starter version.

## Optional OpenAI Answering

The app works without an LLM by producing extractive answers from retrieved chunks. To use OpenAI for more natural synthesis:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4o-mini"
python3 app.py
```

The prompt instructs the model to answer only from retrieved context and include chunk citations.

## File Map

- `app.py` - local HTTP API and static file server.
- `rag_engine.py` - chunking, retrieval, metric extraction, answer generation.
- `storage.py` - JSON document persistence.
- `static/` - browser UI.
- `sample_reports/` - demo finance reports.
- `data/documents.json` - local indexed report store, created at runtime.

## Upgrade Ideas

- Replace TF-IDF with sentence-transformer or OpenAI embeddings.
- Store chunks in ChromaDB, Qdrant, Pinecone, or Supabase pgvector.
- Add table extraction for income statement, balance sheet, and cash flow tables.
- Add evaluation questions with faithfulness and citation coverage metrics.
- Add multi-company comparison reports.
- Add authentication and per-user document libraries.

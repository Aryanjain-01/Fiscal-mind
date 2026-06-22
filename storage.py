from __future__ import annotations

import json
import uuid
from pathlib import Path

from rag_engine import Document


DATA_DIR = Path("data")
DOCUMENTS_FILE = DATA_DIR / "documents.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def load_documents() -> list[Document]:
    ensure_data_dir()
    if not DOCUMENTS_FILE.exists():
        return []

    raw_documents = json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8"))
    return [Document(**item) for item in raw_documents]


def save_documents(documents: list[Document]) -> None:
    ensure_data_dir()
    DOCUMENTS_FILE.write_text(
        json.dumps([document.__dict__ for document in documents], indent=2),
        encoding="utf-8",
    )


def add_document(name: str, text: str) -> Document:
    documents = load_documents()
    document = Document(id=str(uuid.uuid4()), name=name, text=text)
    documents.append(document)
    save_documents(documents)
    return document


def delete_document(document_id: str) -> bool:
    documents = load_documents()
    kept = [document for document in documents if document.id != document_id]
    if len(kept) == len(documents):
        return False
    save_documents(kept)
    return True


def clear_documents() -> None:
    save_documents([])

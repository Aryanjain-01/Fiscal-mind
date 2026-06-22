from __future__ import annotations

import base64
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rag_engine import FinanceRAG
from storage import add_document, clear_documents, delete_document, load_documents


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
SAMPLE_DIR = ROOT / "sample_reports"


def json_response(handler: BaseHTTPRequestHandler, payload: dict | list, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    body = handler.rfile.read(length).decode("utf-8")
    return json.loads(body) if body else {}


def clean_report_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        import fitz  # type: ignore

        document = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text("text") for page in document)
    except Exception:
        pass

    try:
        from pypdf import PdfReader  # type: ignore
        import io

        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def decode_upload(payload: dict) -> tuple[str, str]:
    name = payload.get("name", "Untitled report")
    text = payload.get("text", "")
    encoded_file = payload.get("fileBase64")

    if encoded_file:
        file_bytes = base64.b64decode(encoded_file)
        if name.lower().endswith(".pdf"):
            text = extract_pdf_text(file_bytes)
            if not text.strip():
                raise ValueError("PDF text extraction needs PyMuPDF or pypdf installed, or a text-based PDF.")
        else:
            text = file_bytes.decode("utf-8", errors="ignore")

    text = clean_report_text(text)
    if len(text) < 80:
        raise ValueError("Please provide a longer finance report text or upload a readable text/PDF file.")
    return name, text


class FinanceRAGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.serve_static("index.html")
            return
        if path == "/api/documents":
            documents = load_documents()
            json_response(
                self,
                [
                    {
                        "id": document.id,
                        "name": document.name,
                        "characters": len(document.text),
                    }
                    for document in documents
                ],
            )
            return
        if path == "/api/stats":
            documents = load_documents()
            rag = FinanceRAG(documents)
            json_response(self, {"documents": len(documents), "chunks": len(rag.chunks)})
            return
        if path.startswith("/static/"):
            self.serve_static(path.replace("/static/", "", 1))
            return

        json_response(self, {"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/api/documents":
                payload = read_json(self)
                name, text = decode_upload(payload)
                document = add_document(name, text)
                json_response(self, {"id": document.id, "name": document.name, "characters": len(document.text)}, status=201)
                return

            if parsed.path == "/api/query":
                payload = read_json(self)
                query = payload.get("query", "").strip()
                if not query:
                    json_response(self, {"error": "Question is required."}, status=400)
                    return
                rag = FinanceRAG(load_documents())
                json_response(self, rag.answer(query, top_k=int(payload.get("topK", 5))))
                return

            if parsed.path == "/api/load-sample":
                clear_documents()
                for file_path in sorted(SAMPLE_DIR.glob("*.txt")):
                    add_document(file_path.name, file_path.read_text(encoding="utf-8"))
                json_response(self, {"loaded": len(load_documents())})
                return

            if parsed.path == "/api/clear":
                clear_documents()
                json_response(self, {"ok": True})
                return
        except ValueError as error:
            json_response(self, {"error": str(error)}, status=400)
            return
        except json.JSONDecodeError:
            json_response(self, {"error": "Invalid JSON body."}, status=400)
            return

        json_response(self, {"error": "Not found"}, status=404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/documents/"):
            document_id = parsed.path.rsplit("/", 1)[-1]
            if delete_document(document_id):
                json_response(self, {"ok": True})
            else:
                json_response(self, {"error": "Document not found."}, status=404)
            return
        json_response(self, {"error": "Not found"}, status=404)

    def serve_static(self, filename: str) -> None:
        file_path = (STATIC_DIR / filename).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.exists():
            json_response(self, {"error": "Not found"}, status=404)
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), FinanceRAGHandler)
    print(f"Fiscal Mind running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

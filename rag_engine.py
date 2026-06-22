from __future__ import annotations

import math
import os
import re
import urllib.error
import urllib.request
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "with",
}


@dataclass
class Document:
    id: str
    name: str
    text: str


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    text: str
    chunk_index: int


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9&.-]*", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]


def chunk_text(text: str, max_words: int = 180, overlap: int = 35) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []

    words = clean_text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)

    return chunks


def extract_financial_metrics(text: str) -> list[str]:
    patterns = [
        r"(revenue|net sales|sales|income|profit|ebitda|gross margin|operating margin|cash flow|free cash flow|assets|liabilities|debt|eps|earnings per share)[^.:\n]{0,90}?(\$?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|m|bn|%)?)",
        r"(\$?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|m|bn|%)?)\s+(revenue|net sales|sales|income|profit|ebitda|gross margin|operating margin|cash flow|free cash flow|assets|liabilities|debt|eps|earnings per share)",
    ]

    findings: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = " ".join(part.strip() for part in match.groups())
            phrase = re.sub(r"\s+", " ", phrase)
            key = phrase.lower()
            if key not in seen:
                findings.append(phrase)
                seen.add(key)
    return findings[:10]


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


class FinanceRAG:
    def __init__(self, documents: Iterable[Document]):
        self.documents = list(documents)
        self.chunks = self._build_chunks(self.documents)
        self.chunk_term_counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.idf = self._build_idf(self.chunk_term_counts)
        self.chunk_vectors = [self._weight_vector(counter) for counter in self.chunk_term_counts]

    def _build_chunks(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            for index, text in enumerate(chunk_text(document.text)):
                chunks.append(
                    Chunk(
                        id=f"{document.id}::chunk-{index + 1}",
                        document_id=document.id,
                        document_name=document.name,
                        text=text,
                        chunk_index=index + 1,
                    )
                )
        return chunks

    def _build_idf(self, counters: list[Counter[str]]) -> dict[str, float]:
        doc_frequency: defaultdict[str, int] = defaultdict(int)
        for counter in counters:
            for term in counter:
                doc_frequency[term] += 1

        total = max(len(counters), 1)
        return {term: math.log((1 + total) / (1 + freq)) + 1 for term, freq in doc_frequency.items()}

    def _weight_vector(self, counter: Counter[str]) -> dict[str, float]:
        total_terms = sum(counter.values()) or 1
        return {term: (count / total_terms) * self.idf.get(term, 0.0) for term, count in counter.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0

        numerator = sum(left.get(term, 0.0) * right.get(term, 0.0) for term in left.keys() & right.keys())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_counter = Counter(tokenize(query))
        query_vector = self._weight_vector(query_counter)
        ranked: list[RetrievalResult] = []

        for chunk, chunk_vector in zip(self.chunks, self.chunk_vectors):
            score = self._cosine(query_vector, chunk_vector)
            if score > 0:
                ranked.append(RetrievalResult(chunk=chunk, score=score))

        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[:top_k]

    def answer(self, query: str, top_k: int = 5) -> dict:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return {
                "answer": "I could not find relevant evidence in the uploaded finance reports.",
                "metrics": [],
                "sources": [],
            }

        openai_answer = self._try_openai_answer(query, results)
        if openai_answer:
            answer_text = openai_answer
        else:
            answer_text = self._extractive_answer(query, results)

        combined_text = " ".join(result.chunk.text for result in results)
        return {
            "answer": answer_text,
            "metrics": extract_financial_metrics(combined_text),
            "sources": [
                {
                    **asdict(result.chunk),
                    "score": round(result.score, 4),
                    "preview": result.chunk.text[:420] + ("..." if len(result.chunk.text) > 420 else ""),
                }
                for result in results
            ],
        }

    def _extractive_answer(self, query: str, results: list[RetrievalResult]) -> str:
        query_terms = set(tokenize(query))
        candidates: list[tuple[int, str, Chunk]] = []

        for result in results:
            for sentence in split_sentences(result.chunk.text):
                overlap = len(query_terms & set(tokenize(sentence)))
                if overlap:
                    candidates.append((overlap, sentence, result.chunk))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            first_sentences = split_sentences(results[0].chunk.text)
            first_sentence = first_sentences[0] if first_sentences else results[0].chunk.text
            candidates = [(0, first_sentence, results[0].chunk)]

        lines = []
        used_sentences: set[str] = set()
        for _, sentence, chunk in candidates:
            key = sentence.lower()
            if key in used_sentences:
                continue
            used_sentences.add(key)
            lines.append(f"{sentence} [{chunk.document_name}, chunk {chunk.chunk_index}]")
            if len(lines) == 4:
                break

        return " ".join(lines)

    def _try_openai_answer(self, query: str, results: list[RetrievalResult]) -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        context = "\n\n".join(
            f"Source: {result.chunk.document_name}, chunk {result.chunk.chunk_index}\n{result.chunk.text}"
            for result in results
        )
        prompt = (
            "You are a finance report analysis assistant. Answer only from the supplied context. "
            "Include citations in square brackets using the provided source and chunk labels. "
            "If the context is insufficient, say so.\n\n"
            f"Question: {query}\n\nContext:\n{context}"
        )
        body = json.dumps(
            {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
            return None

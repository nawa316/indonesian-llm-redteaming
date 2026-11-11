#!/usr/bin/env python3
"""ChromaDB persistence for the Indonesian RAG corpus.

Embeddings come from Ollama (``nomic-embed-text``) when the model is available,
and fall back to a **deterministic** scikit-learn character n-gram TF-IDF
embedding fitted on the seed corpus otherwise. The fallback keeps ChromaDB
usable offline, during smoke tests, and without spending VRAM on a second model.

Both paths produce 768-dimensional, L2-normalised vectors, so collections stay
compatible across environments.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EMBED_DIM = 768
DEFAULT_COLLECTION = os.getenv("RAG_COLLECTION", "owasp_id_docs")
DEFAULT_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")


def default_chroma_dir() -> Path:
    if os.getenv("CHROMA_DIR"):
        return Path(os.environ["CHROMA_DIR"])
    workspace = Path(os.getenv("WORKSPACE", "/workspace"))
    if workspace.exists():
        return workspace / "chromadb"
    return ROOT / ".runtime" / "chromadb"


def _fallback_corpus() -> list[str]:
    """Seed texts used to fit the offline TF-IDF vectorizer deterministically."""
    path = ROOT / "dataset" / "rag_documents_id.json"
    if path.exists():
        try:
            docs = json.loads(path.read_text(encoding="utf-8"))
            texts = [str(d["text"]) for d in docs if d.get("text")]
            if texts:
                return texts
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return [
        "kebijakan keamanan informasi dan kata sandi",
        "konfigurasi database internal dan kredensial layanan",
        "prosedur respons insiden dan isolasi tenant",
    ]


@dataclass
class RAGChunk:
    doc_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None

    @property
    def access_level(self) -> str:
        return str(self.metadata.get("access_level", "public"))

    @property
    def is_poisoned(self) -> bool:
        value = self.metadata.get("is_poisoned", False)
        return value is True or str(value).strip().casefold() in {"true", "1", "yes"}


class OllamaEmbeddingFunction:
    """Chroma embedding function: Ollama first, deterministic hash fallback.

    The class is duck-typed against chromadb's ``EmbeddingFunction`` protocol and
    intentionally declares ``name()`` for Chroma >= 1.0.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or DEFAULT_EMBED_MODEL
        self.timeout = timeout
        self._ollama_disabled = False
        self._fallback = None

    # -- protocol ----------------------------------------------------------
    def name(self) -> str:
        return f"ollama-or-hash-{self.model}"

    def __call__(self, input: Sequence[str] | str) -> list[list[float]]:  # noqa: A002
        texts = [input] if isinstance(input, str) else list(input)
        if not texts:
            return []

        vectors = self._embed_ollama(texts)
        if vectors is not None:
            return vectors
        return self._embed_fallback(texts)

    def embed_documents(self, input: Sequence[str] | str) -> list[list[float]]:  # noqa: A002
        return self(input)

    def embed_query(self, input: str | Sequence[str]) -> list[float]:  # noqa: A002
        # Chroma 1.x passes the query as a list even for embed_query.
        return self(input)[0]

    # -- backends ----------------------------------------------------------
    def _embed_ollama(self, texts: Sequence[str]) -> list[list[float]] | None:
        if self._ollama_disabled:
            return None
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": list(texts)},
                timeout=self.timeout,
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
            if not embeddings or len(embeddings) != len(texts):
                raise ValueError("unexpected /api/embed payload")
            return [[float(x) for x in vector] for vector in embeddings]
        except (httpx.HTTPError, ValueError, KeyError):
            self._ollama_disabled = True
            return None

    def _embed_fallback(self, texts: Sequence[str]) -> list[list[float]]:
        if self._fallback is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "scikit-learn is required for the offline embedding fallback; "
                    "install it with `pip install scikit-learn>=1.5`"
                ) from exc
            corpus = _fallback_corpus()
            self._fallback = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                max_features=EMBED_DIM,
                norm="l2",
                lowercase=True,
                sublinear_tf=True,
            )
            self._fallback.fit(corpus)

        matrix = self._fallback.transform(texts)
        vectors: list[list[float]] = []
        for row in matrix:
            values = row.toarray().ravel().astype("float32").tolist()
            # Keep a fixed 768-d contract even if the fitted vocabulary is
            # smaller, so collections remain compatible with Ollama vectors.
            if len(values) < EMBED_DIM:
                values.extend([0.0] * (EMBED_DIM - len(values)))
            vectors.append(values[:EMBED_DIM])
        return vectors


class ChromaStore:
    """Thin, typed wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        path: str | Path | None = None,
        collection: str | None = None,
        embedding_function: OllamaEmbeddingFunction | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "chromadb is required for RAG; install it with "
                "`pip install 'chromadb>=0.5'`"
            ) from exc

        self.path = Path(path) if path else default_chroma_dir()
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection or DEFAULT_COLLECTION
        self.embedding_function = embedding_function or OllamaEmbeddingFunction()

        # Keep Chroma quiet and offline-friendly.
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    # -- writes ------------------------------------------------------------
    def add_documents(self, documents: Iterable[dict[str, Any]]) -> int:
        docs = list(documents)
        if not docs:
            return 0
        texts = [d["text"] for d in docs]
        # Compute embeddings explicitly: Chroma's EF protocol changed across
        # 0.5 -> 1.x, explicit vectors keep citations reproducible everywhere.
        embeddings = self.embedding_function(texts)
        self.collection.upsert(
            ids=[d["doc_id"] for d in docs],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "doc_id": d["doc_id"],
                    "owasp_category": d.get("owasp_category", ""),
                    "access_level": d.get("access_level", "public"),
                    "is_poisoned": bool(d.get("is_poisoned", False)),
                }
                for d in docs
            ],
        )
        return len(docs)

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # collection may not exist
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    # -- reads -------------------------------------------------------------
    def count(self) -> int:
        return int(self.collection.count())

    def query(
        self,
        text: str,
        *,
        top_k: int = 4,
        public_only: bool = False,
    ) -> list[RAGChunk]:
        if not text.strip() or self.count() == 0:
            return []
        where = {"access_level": "public"} if public_only else None
        query_vector = self.embedding_function.embed_query(text)
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=max(1, min(top_k, self.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        chunks: list[RAGChunk] = []
        for index, document in enumerate(documents):
            metadata = dict(metadatas[index] or {})
            chunks.append(
                RAGChunk(
                    doc_id=str(metadata.get("doc_id") or f"doc-{index}"),
                    text=document,
                    metadata=metadata,
                    distance=float(distances[index]) if index < len(distances) else None,
                )
            )
        return chunks


if __name__ == "__main__":  # smoke test
    store = ChromaStore()
    print(f"path={store.path} collection={store.collection_name} count={store.count()}")
    for chunk in store.query("kebijakan kata sandi", top_k=2):
        print(chunk.doc_id, chunk.access_level, chunk.is_poisoned, chunk.distance)

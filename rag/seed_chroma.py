#!/usr/bin/env python3
"""Seed ChromaDB with the Indonesian RAG corpus.

Usage:
    python rag/seed_chroma.py            # upsert (idempotent)
    python rag/seed_chroma.py --reset    # wipe collection, then seed
    python rag/seed_chroma.py --verify   # print counts and sample retrieval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.chroma_store import ChromaStore  # noqa: E402

DEFAULT_JSON = ROOT / "dataset" / "rag_documents_id.json"
REQUIRED = {"doc_id", "text", "owasp_category", "access_level", "is_poisoned"}
ALLOWED_ACCESS = {"public", "confidential"}


def load_documents(path: Path) -> list[dict]:
    docs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"{path}: expected a non-empty JSON array")

    seen: set[str] = set()
    for index, doc in enumerate(docs):
        missing = REQUIRED - set(doc)
        if missing:
            raise ValueError(f"{path}: document #{index} missing {sorted(missing)}")
        if doc["doc_id"] in seen:
            raise ValueError(f"{path}: duplicate doc_id {doc['doc_id']!r}")
        seen.add(doc["doc_id"])
        if doc["access_level"] not in ALLOWED_ACCESS:
            raise ValueError(
                f"{doc['doc_id']}: access_level must be one of {sorted(ALLOWED_ACCESS)}"
            )
        if not isinstance(doc["is_poisoned"], bool):
            raise ValueError(f"{doc['doc_id']}: is_poisoned must be boolean")
        if not str(doc["text"]).strip():
            raise ValueError(f"{doc['doc_id']}: empty text")
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--path", default=None, help="Chroma persistence directory")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    docs = load_documents(Path(args.json))
    store = ChromaStore(path=args.path, collection=args.collection)
    if args.reset:
        print(f"resetting collection {store.collection_name}")
        store.reset()

    written = store.add_documents(docs)
    public = sum(1 for d in docs if d["access_level"] == "public")
    confidential = len(docs) - public
    poisoned = sum(1 for d in docs if d["is_poisoned"])
    print(
        f"seeded {written} documents into {store.path} "
        f"(collection={store.collection_name}, public={public}, "
        f"confidential={confidential}, poisoned={poisoned})"
    )

    if store.count() != len(docs):
        print("WARNING: collection count differs from seed file", file=sys.stderr)

    if args.verify:
        for query in ("konfigurasi database", "kebijakan kata sandi", "kredensial layanan"):
            print(f"\nquery={query!r}")
            for chunk in store.query(query, top_k=3):
                print(
                    f"  {chunk.doc_id:<14} access={chunk.access_level:<12} "
                    f"poisoned={chunk.is_poisoned!s:<5} distance={chunk.distance:.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

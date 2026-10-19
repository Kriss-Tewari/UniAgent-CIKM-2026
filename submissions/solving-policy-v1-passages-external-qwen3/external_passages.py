"""Exact external-source excerpts with corpus and character-offset provenance."""
from __future__ import annotations
import re

def make_external_passages(
    documents,
    window=200,
    stride=160,
    character_limit=200_000,
):
    if not 1 <= stride <= window:
        raise ValueError("Require 1 <= stride <= window.")
    if character_limit < 1:
        raise ValueError("character_limit must be positive.")

    passages = []
    seen = set()

    for document in documents:
        corpus = str(document["corpus"])
        doc_id = str(document["doc_id"])
        identity = (corpus, doc_id)

        if identity in seen:
            raise ValueError(f"Duplicate source document: {identity}")
        seen.add(identity)

        original = document["text"]
        if not isinstance(original, str):
            raise TypeError("Document text must be a string.")

        searchable = original[:character_limit]
        words = list(re.finditer(r"\S+", searchable))

        for offset in range(0, len(words), stride):
            end = min(offset + window, len(words))
            start_char = words[offset].start()
            end_char = words[end - 1].end()

            passages.append({
                "id": f"X{len(passages) + 1:05d}",
                "document": f"{corpus}#{doc_id}",
                "corpus": corpus,
                "doc_id": doc_id,
                "title": document.get("title"),
                "url": document.get("url"),
                "word_start": offset,
                "word_end": end,
                "char_start": start_char,
                "char_end": end_char,
                "source_characters": len(original),
                "source_truncated": len(original) > character_limit,
                "text": original[start_char:end_char],
            })

            if end == len(words):
                break

    return passages

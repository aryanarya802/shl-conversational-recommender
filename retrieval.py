"""
Keyword-based retrieval over the SHL catalog.
Scores catalog items by matching terms from the conversation against
item name, description, keys, and job_levels fields.
"""
from __future__ import annotations
import json
import math
import os
import re
from collections import Counter
from typing import Any

_CATALOG: list[dict[str, Any]] = []


def load_catalog(path: str | None = None) -> None:
    global _CATALOG
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "catalog.json")
    with open(path, encoding="utf-8") as f:
        _CATALOG = json.load(f)


def get_all_items() -> list[dict[str, Any]]:
    return _CATALOG


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def _build_item_text(item: dict[str, Any]) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
    ]
    return " ".join(p for p in parts if p)


def retrieve(query: str, top_k: int = 30) -> list[dict[str, Any]]:
    """Return up to top_k catalog items most relevant to query."""
    if not _CATALOG:
        return []

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return _CATALOG[:top_k]

    scores: list[tuple[float, dict[str, Any]]] = []
    for item in _CATALOG:
        item_text = _build_item_text(item)
        item_tokens = _tokenize(item_text)
        item_set = set(item_tokens)
        token_counter = Counter(item_tokens)

        # TF-IDF-like score
        score = 0.0
        for qt in query_tokens:
            if qt in item_set:
                tf = token_counter[qt] / max(len(item_tokens), 1)
                score += 1 + math.log(1 + tf)

        # Boost: name contains query token (exact substring match)
        name_lower = item.get("name", "").lower()
        for qt in query_tokens:
            if qt in name_lower:
                score += 2.0

        # Boost: key categories
        keys_lower = " ".join(item.get("keys", [])).lower()
        for qt in query_tokens:
            if qt in keys_lower:
                score += 1.0

        if score > 0:
            scores.append((score, item))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scores[:top_k]]


def format_catalog_for_prompt(items: list[dict[str, Any]]) -> str:
    """Render catalog items as a compact table for the LLM prompt."""
    lines = []
    for item in items:
        line = (
            f"- name: {item['name']} | url: {item['url']} | "
            f"test_type: {item['test_type']} | "
            f"description: {item.get('description', '')[:180]} | "
            f"job_levels: {', '.join(item.get('job_levels', []))} | "
            f"duration: {item.get('duration', 'N/A')} | "
            f"languages: {', '.join(item.get('languages', []))}"
        )
        lines.append(line)
    return "\n".join(lines)

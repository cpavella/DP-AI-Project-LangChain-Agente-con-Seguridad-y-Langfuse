"""
Pipeline de ingesta RAG · LOADER PDF → Markdown (PyMuPDF4LLM)
Ruta: rag/pipeline/loaders.py

Módulo glue que provee el paso "load" del orquestador:
  _load_pymupdf4llm(raw, filename) -> list[{"text", "metadata"}]  (una por página)

Se usa PyMuPDF4LLM porque extrae **Markdown** (headers #, tablas |, listas), que
es justo lo que aprovecha el chunker estructural (e_chunking · MarkdownNodeParser).

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

from __future__ import annotations

from typing import Any

import pymupdf
import pymupdf4llm


def _load_pymupdf4llm(raw: bytes, filename: str = "") -> list[dict[str, Any]]:
    """
    Extrae un PDF (bytes) a Markdown, una entrada por página.

    Args:
        raw:      bytes del PDF.
        filename: nombre del archivo (va a la metadata como `source`).

    Returns:
        Lista de dicts {"text": <markdown>, "metadata": {source, page, ...}}.
    """
    doc = pymupdf.open(stream=raw, filetype="pdf")
    try:
        # page_chunks=True → una entrada (dict) por página con su markdown + metadata.
        page_data = pymupdf4llm.to_markdown(doc, page_chunks=True)
    finally:
        doc.close()

    pages: list[dict[str, Any]] = []
    for i, pg in enumerate(page_data):
        if isinstance(pg, dict):
            text = pg.get("text", "") or ""
            meta = dict(pg.get("metadata", {}) or {})
        else:  # compatibilidad si una versión devuelve strings
            text = str(pg)
            meta = {}
        meta.setdefault("source", filename)
        meta.setdefault("page", i)
        pages.append({"text": text, "metadata": meta})
    return pages

"""
Pipeline de ingesta RAG · B · OCR CONTROLADO (español)
Ruta: Backend-RAGs/pipeline/b_ocr.py

OCR de las páginas escaneadas que marca extraction_validator.py, EN ESPAÑOL
(por defecto `spa+eng` — Tesseract combina idiomas, robusto para docs mixtos),
en vez del auto-OCR de pymupdf4llm que corre en INGLÉS y destroza tildes/ñ
(probado: "daños" -> "danos", que es otra palabra).

Requiere Tesseract + los traineddata de idiomas:
  macOS:  brew install tesseract tesseract-lang
  Linux:  apt-get install tesseract-ocr tesseract-ocr-spa

La carpeta `tessdata` se autodetecta (o se respeta TESSDATA_PREFIX).
"""

from __future__ import annotations

import os
from typing import Any

import pymupdf

# `spa+eng`: español primero, con inglés de respaldo para PDFs mixtos.
DEFAULT_LANGUAGE = "spa+eng"
DEFAULT_DPI = 200  # 200-300 dpi da buen OCR sin disparar el costo/tiempo

# Ubicaciones típicas de la carpeta tessdata (idiomas).
_TESSDATA_CANDIDATES = (
    "/opt/homebrew/share/tessdata",              # macOS Apple Silicon (Homebrew)
    "/usr/local/share/tessdata",                 # macOS Intel / Linux manual
    "/usr/share/tessdata",
    "/usr/share/tesseract-ocr/5/tessdata",       # Debian/Ubuntu (tesseract 5)
    "/usr/share/tesseract-ocr/4.00/tessdata",    # Debian/Ubuntu (tesseract 4)
)


def resolve_tessdata() -> str | None:
    """Ubica la carpeta `tessdata`. Respeta TESSDATA_PREFIX si está definido."""
    env = os.getenv("TESSDATA_PREFIX")
    if env and os.path.isdir(env):
        return env
    for cand in _TESSDATA_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    return None


def ocr_diagnostics() -> dict[str, Any]:
    """Estado del OCR: dónde está tessdata y si están los idiomas necesarios.
    Útil para un health-check antes de procesar (fallar temprano y claro)."""
    tessdata = resolve_tessdata()
    langs: list[str] = []
    if tessdata:
        try:
            langs = sorted(
                f[: -len(".traineddata")]
                for f in os.listdir(tessdata)
                if f.endswith(".traineddata")
            )
        except OSError:
            langs = []
    return {
        "tessdata": tessdata,
        "available": tessdata is not None and "spa" in langs,
        "num_languages": len(langs),
        "has_spanish": "spa" in langs,
        "has_english": "eng" in langs,
    }


def ocr_pages(
    raw: bytes,
    pages: list[int] | None = None,
    language: str = DEFAULT_LANGUAGE,
    dpi: int = DEFAULT_DPI,
) -> dict[int, str]:
    """OCR de las páginas indicadas (índices 0-based) de un PDF en bytes.

    Args:
      raw:      bytes del PDF.
      pages:    índices 0-based a OCRear (típicamente el `ocr_pages` del validador).
                None → OCR de TODAS las páginas.
      language: idioma(s) Tesseract, p.ej. "spa", "eng", "spa+eng".
      dpi:      resolución de render para el OCR.

    Devuelve {indice_pagina: texto_ocr}. Lanza RuntimeError si falta tessdata.
    """
    tessdata = resolve_tessdata()
    if tessdata is None:
        raise RuntimeError(
            "No se encontró la carpeta 'tessdata' de Tesseract. Instálalo con: "
            "brew install tesseract tesseract-lang (macOS) o define TESSDATA_PREFIX."
        )

    doc = pymupdf.open(stream=raw, filetype="pdf")
    try:
        targets = pages if pages is not None else list(range(doc.page_count))
        out: dict[int, str] = {}
        for i in targets:
            if i < 0 or i >= doc.page_count:
                continue
            page = doc[i]
            tp = page.get_textpage_ocr(
                language=language, dpi=dpi, full=True, tessdata=tessdata
            )
            out[i] = page.get_text(textpage=tp)
        return out
    finally:
        doc.close()


def ocr_page_text(
    raw: bytes,
    page_index: int,
    language: str = DEFAULT_LANGUAGE,
    dpi: int = DEFAULT_DPI,
) -> str:
    """Atajo: OCR de una sola página. Devuelve su texto (o '' si no aplica)."""
    return ocr_pages(raw, [page_index], language=language, dpi=dpi).get(page_index, "")

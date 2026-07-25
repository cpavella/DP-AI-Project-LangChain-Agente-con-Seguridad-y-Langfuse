"""
Pipeline de ingesta RAG · A · VALIDAR EXTRACCIÓN
Ruta: Backend-RAGs/pipeline/a_extraction_validator.py

Paso 1 del pipeline (la "puerta de robustez"): antes de normalizar/chunkear,
inspecciona la CAPA DE TEXTO NATIVA del PDF (sin OCR) para clasificar cada página
y decidir qué hacer con el documento — sobre todo, QUÉ PÁGINAS necesitan OCR.

⚠️ Clave: se analiza el texto NATIVO (`page.get_text()`), NO la salida de
pymupdf4llm — porque pymupdf4llm ya OCRea (en inglés) las páginas escaneadas, lo
que OCULTARÍA que son escaneadas. Aquí queremos DETECTARLAS para OCRearlas en
español nosotros después (ver pipeline/ocr.py).

Veredictos por página:
  ok       -> texto nativo sano                         → chunkear normal
  scanned  -> poco/nada de texto + imagen grande        → OCR (español)
  empty    -> poco/nada de texto + sin imagen           → saltar (página en blanco)
  garbage  -> hay texto pero es gibberish (encoding roto) → revisar / OCR

Veredicto del documento: ok · needs_ocr · partial_ocr · garbage · empty
"""

from __future__ import annotations

import re
from typing import Any

import pymupdf

# --------------------------------------------------------------------------- #
# Umbrales · calibrar con PDFs reales (por eso cada página devuelve `signals`)
# --------------------------------------------------------------------------- #
MIN_CHARS_PER_PAGE = 50      # menos que esto = página "flaca" (sin capa de texto útil)
IMG_COVER_SCANNED = 0.60     # imagen cubre >60% de la página = escaneado (no vacío)
MAX_REPLACEMENT_RATIO = 0.05  # >5% de '�' (U+FFFD) = encoding roto
MIN_ALPHA_RATIO = 0.55       # <55% de letras = sospechoso
MIN_WORDLIKE_RATIO = 0.35    # <35% de tokens "palabra" = gibberish

_WORD = re.compile(r"[A-Za-zÀ-ÿ]{2,}")


# --------------------------------------------------------------------------- #
# Señales por página
# --------------------------------------------------------------------------- #
def _image_coverage(page) -> float:
    """Fracción del área de la página cubierta por imágenes.

    Es la señal que distingue `scanned` (imagen grande + sin texto) de `empty`
    (sin texto y sin imagen)."""
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    covered = 0.0
    try:
        infos = page.get_image_info()
    except Exception:  # noqa: BLE001
        infos = []
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        r = pymupdf.Rect(bbox)
        covered += abs(r.width * r.height)
    return min(covered / page_area, 1.0)


def classify_page(text: str, image_coverage: float) -> tuple[str, dict[str, Any]]:
    """Clasifica una página a partir de su texto nativo + cobertura de imagen.
    Devuelve (veredicto, señales) — las señales sirven para auditar y calibrar."""
    t = (text or "").strip()
    n = len(t)
    signals: dict[str, Any] = {"chars": n, "image_coverage": round(image_coverage, 2)}

    # Página con poco/nada de texto → escaneada (si hay imagen) o vacía.
    if n < MIN_CHARS_PER_PAGE:
        verdict = "scanned" if image_coverage >= IMG_COVER_SCANNED else "empty"
        return verdict, signals

    # Hay texto: ¿es sano o gibberish (encoding roto)?
    letters = sum(c.isalpha() for c in t)
    replacement = t.count("�") / n
    alpha_ratio = letters / n
    tokens = t.split()
    wordlike = sum(bool(_WORD.search(tok)) for tok in tokens) / max(1, len(tokens))
    signals.update(
        replacement=round(replacement, 3),
        alpha=round(alpha_ratio, 2),
        wordlike=round(wordlike, 2),
    )
    if (
        replacement > MAX_REPLACEMENT_RATIO
        or alpha_ratio < MIN_ALPHA_RATIO
        or wordlike < MIN_WORDLIKE_RATIO
    ):
        return "garbage", signals
    return "ok", signals


# --------------------------------------------------------------------------- #
# Veredicto del documento
# --------------------------------------------------------------------------- #
def _summarize(per_page: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for pp in per_page:
        counts[pp["verdict"]] = counts.get(pp["verdict"], 0) + 1
    total = len(per_page) or 1
    ok = counts.get("ok", 0)
    scanned = counts.get("scanned", 0)
    garbage = counts.get("garbage", 0)

    if garbage / total > 0.3:
        doc = "garbage"              # demasiado texto roto → revisar / reintentar
    elif ok == 0 and scanned > 0:
        doc = "needs_ocr"           # nada de texto nativo, todo escaneado → OCR completo
    elif scanned > 0 or garbage > 0:
        doc = "partial_ocr"         # mezcla → OCR solo las páginas malas
    elif ok / total >= 0.9:
        doc = "ok"                  # extracción sana
    elif ok == 0:
        doc = "empty"               # nada aprovechable → NO indexar, reportar
    else:
        doc = "ok"

    # Índices 0-based de las páginas a mandar a OCR (consumido por pipeline/ocr.py).
    ocr_pages = [pp["index"] for pp in per_page if pp["verdict"] in ("scanned", "garbage")]
    return {
        "doc_verdict": doc,
        "num_pages": len(per_page),
        "counts": counts,
        "ocr_pages": ocr_pages,
        "pages": per_page,
    }


def validate_extraction(raw: bytes) -> dict[str, Any]:
    """Analiza un PDF (bytes) usando su CAPA DE TEXTO NATIVA (sin OCR) y clasifica
    cada página + el documento.

    Devuelve un reporte:
      {
        "doc_verdict": "ok|needs_ocr|partial_ocr|garbage|empty",
        "num_pages": int,
        "counts": {verdict: n},
        "ocr_pages": [índices 0-based a OCRear en español],
        "pages": [{page, index, verdict, signals}],
      }
    """
    doc = pymupdf.open(stream=raw, filetype="pdf")
    try:
        per_page: list[dict[str, Any]] = []
        for i in range(doc.page_count):
            page = doc[i]
            text = page.get_text()  # ← capa NATIVA · sin OCR (a propósito)
            cov = _image_coverage(page)
            verdict, signals = classify_page(text, cov)
            per_page.append({"page": i + 1, "index": i, "verdict": verdict, "signals": signals})
        return _summarize(per_page)
    finally:
        doc.close()

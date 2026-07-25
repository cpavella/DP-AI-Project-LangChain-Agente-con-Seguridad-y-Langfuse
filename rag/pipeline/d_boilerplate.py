"""
Pipeline de ingesta RAG · D · QUITAR BOILERPLATE
Ruta: Backend-RAGs/pipeline/d_boilerplate.py

Paso 3 del pipeline: elimina encabezados / pies de pagina / numeros de pagina /
watermarks que se repiten en muchas paginas y contaminan los chunks y el retrieval.

Tecnica robusta:
  1. Mira solo las N lineas del BORDE (top/bottom) de cada pagina — ahi vive el
     boilerplate; el cuerpo no se toca (evita falsos positivos).
  2. Normaliza cada linea candidata colapsando digitos ('Pagina 3' == 'Pagina 4')
     para detectar repeticion aunque cambie el numero.
  3. Las lineas que aparecen en >= `min_ratio` de las paginas se marcan boilerplate
     y se quitan SOLO de los bordes.

Necesita varias paginas para funcionar (un doc de 1 pagina no tiene repeticion →
no se quita nada, que es lo correcto).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# Colapsa numeros a '#' para que "pagina 3" y "pagina 4" cuenten como la misma.
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def _norm_line(line: str) -> str:
    return _WS.sub(" ", _DIGITS.sub("#", line.strip().lower())).strip()


def _edge_indices(lines: list[str], edge_lines: int) -> set[int]:
    """Indices de las primeras/ultimas `edge_lines` lineas NO vacias."""
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    return set(nonempty[:edge_lines] + nonempty[-edge_lines:])


def detect_boilerplate(
    pages: list[str], *, edge_lines: int = 3, min_ratio: float = 0.5
) -> set[str]:
    """Devuelve el conjunto de lineas-borde normalizadas consideradas boilerplate."""
    n = len(pages)
    if n < 2:
        return set()
    counter: Counter[str] = Counter()
    for text in pages:
        lines = text.split("\n")
        seen = set()
        for i in _edge_indices(lines, edge_lines):
            nl = _norm_line(lines[i])
            if nl:
                seen.add(nl)
        counter.update(seen)
    threshold = max(2, int(min_ratio * n))
    return {line for line, c in counter.items() if c >= threshold}


def remove_boilerplate(
    pages: list[str], *, edge_lines: int = 3, min_ratio: float = 0.5
) -> dict[str, Any]:
    """Quita el boilerplate detectado SOLO de los bordes de cada pagina.

    Devuelve {"pages": [...limpias...], "boilerplate": [...], "removed_lines": int}.
    """
    boiler = detect_boilerplate(pages, edge_lines=edge_lines, min_ratio=min_ratio)
    cleaned: list[str] = []
    removed = 0
    for text in pages:
        lines = text.split("\n")
        edges = _edge_indices(lines, edge_lines)
        kept = []
        for i, ln in enumerate(lines):
            if i in edges and _norm_line(ln) in boiler:
                removed += 1
                continue
            kept.append(ln)
        cleaned.append("\n".join(kept).strip())
    return {"pages": cleaned, "boilerplate": sorted(boiler), "removed_lines": removed}

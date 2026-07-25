"""
Runner de ingesta LOCAL del pipeline RAG.
Ruta: rag/pipeline/run_ingesta.py

Lee los PDFs de una carpeta local (por defecto rag/documents/) y los ingesta a la
colección del tenant en Qdrant, imprimiendo el progreso paso a paso.

Es el punto de entrada ejecutable del pipeline: envuelve
`ingest_documents(files, tenant_input)` de g_orchestrator (que recibe bytes y
emite eventos) con la lectura de archivos del disco.

Uso:
    python -m rag.pipeline.run_ingesta
    python -m rag.pipeline.run_ingesta --tenant "Municipio de Girardota"
    python -m rag.pipeline.run_ingesta --dir rag/documents --tenant "Girardota"

Requisitos (libs del pipeline, ver requirements.txt):
    pip install pymupdf4llm PyMuPDF llama-index-core
    # OCR español (opcional): brew install tesseract tesseract-lang

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

from __future__ import annotations

import argparse
import os
import sys

# Bootstrap: permite ejecutar tanto `python -m rag.pipeline.run_ingesta` como
# `python rag/pipeline/run_ingesta.py` (este último no trae el proyecto en sys.path).
if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _ROOT)

from rag.pipeline.g_orchestrator import ingest_documents

# Tenant por defecto (override con --tenant o la env RAG_TENANT).
DEFAULT_TENANT = os.getenv("RAG_TENANT", "Municipio de Girardota")
# Carpeta de documentos por defecto: rag/documents/
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")


def _cargar_pdfs(carpeta: str) -> list[tuple[str, bytes]]:
    """Lee todos los .pdf de `carpeta` como (nombre, bytes)."""
    if not os.path.isdir(carpeta):
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")
    files: list[tuple[str, bytes]] = []
    for nombre in sorted(os.listdir(carpeta)):
        if nombre.lower().endswith(".pdf"):
            with open(os.path.join(carpeta, nombre), "rb") as f:
                files.append((nombre, f.read()))
    return files


def _imprimir_evento(ev: dict) -> None:
    """Formatea un evento del orquestador para la terminal."""
    tipo = ev.get("event")
    if tipo == "start":
        print(f"🚀 Ingesta → colección '{ev['collection']}' · {ev['num_docs']} documento(s)\n")
    elif tipo == "step":
        icono = {"running": "⏳", "done": "✅", "skipped": "⏭️", "error": "❌"}.get(ev["status"], "•")
        extra = {k: v for k, v in ev.items()
                 if k not in ("event", "doc", "filename", "step", "label", "status")}
        extra_txt = f"  {extra}" if extra else ""
        print(f"  {icono} [{ev['filename']}] {ev['label']} — {ev['status']}{extra_txt}")
    elif tipo == "doc_error":
        print(f"  ❌ [{ev['filename']}] ERROR: {ev['error']}")
    elif tipo == "done":
        print(f"\n🏁 Listo · colección '{ev['collection']}' · "
              f"{ev['num_vectors']} vectores · ruta de índice: {ev['index_route']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta local de PDFs → Qdrant (pipeline RAG).")
    parser.add_argument("--tenant", default=DEFAULT_TENANT,
                        help=f"Nombre del tenant (default: '{DEFAULT_TENANT}').")
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help="Carpeta con los PDFs a ingestar (default: rag/documents/).")
    args = parser.parse_args()

    files = _cargar_pdfs(args.dir)
    if not files:
        print(f"⚠️  No se encontraron PDFs en {args.dir}. Añade archivos y reintenta.")
        return

    print(f"📂 {len(files)} PDF(s) en {args.dir}: {', '.join(n for n, _ in files)}\n")
    for ev in ingest_documents(files, args.tenant):
        _imprimir_evento(ev)


if __name__ == "__main__":
    main()

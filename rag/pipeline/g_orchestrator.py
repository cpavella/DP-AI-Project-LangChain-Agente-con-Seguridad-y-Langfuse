"""
Pipeline de ingesta RAG · G · ORQUESTADOR
Ruta: Backend-RAGs/pipeline/g_orchestrator.py

El pegamento: encadena loader → a → (b OCR es) → c → d → e → f → embeddings →
upsert a Qdrant, y va EMITIENDO eventos de progreso (generator) para que el
frontend muestre el spinner por paso y la ruta de indexación elegida.

Eventos que emite (dicts):
  {"event":"start", "collection", "num_docs"}
  {"event":"step", "doc", "filename", "step", "label", "status", ...extra}
     step ∈ load|validate|ocr|normalize|boilerplate|chunk|embed|index
     status ∈ running|done|skipped|error
     el step "index" (done) trae: index_route ("brute_force"|"hnsw"), num_vectors, indexing_threshold
  {"event":"doc_error", "doc", "filename", "error"}
  {"event":"done", "collection", "index_route", "num_vectors", "indexing_threshold"}

La ruta de indexación la decide Qdrant por su `indexing_threshold` (default 20000):
por debajo → full scan (fuerza bruta); por encima → HNSW. Aquí solo la LEEMOS y
reportamos para que el usuario vea cuál tomó.
"""

from __future__ import annotations

from typing import Any, Iterator

from .embeddings import DEFAULT_DIMENSION, embed_documents
from .loaders import _load_pymupdf4llm
from .vectorstores import _qdrant_client

from .a_extraction_validator import validate_extraction
from .b_ocr import ocr_pages
from .c_normalize import normalize_documents
from .d_boilerplate import remove_boilerplate
from .e_chunking import chunk_markdown
from .f_metadata import build_payloads, document_checksum
from .naming import collection_name

DEFAULT_INDEXING_THRESHOLD = 20000  # default de Qdrant (por debajo = fuerza bruta)


# --------------------------------------------------------------------------- #
# Qdrant helpers (ingesta · distinto del benchmark de vectorstores.py)
# --------------------------------------------------------------------------- #
def _ensure_collection(client, name: str, dim: int) -> None:
    """Crea la colección del tenant si no existe (size=dim del modelo, COSINE)."""
    from qdrant_client import models as qm

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )


def _upsert(client, name: str, payloads: list[dict[str, Any]], vectors) -> None:
    """Upsert idempotente (IDs deterministas de f_metadata → mismo doc no duplica)."""
    from qdrant_client import models as qm

    points = [
        qm.PointStruct(id=p["id"], vector=vectors[i].tolist(), payload=p["payload"])
        for i, p in enumerate(payloads)
    ]
    B = 256
    for s in range(0, len(points), B):
        client.upsert(collection_name=name, points=points[s : s + B], wait=True)


def _index_route(client, name: str) -> tuple[str, int, int]:
    """Lee de Qdrant cuántos vectores hay y el umbral, y deduce la ruta activa.
    Si la colección no existe todavía (p.ej. ningún doc produjo chunks), reporta
    fuerza bruta con 0 vectores en vez de fallar."""
    if not client.collection_exists(name):
        return "brute_force", 0, DEFAULT_INDEXING_THRESHOLD
    info = client.get_collection(name)
    count = int(getattr(info, "points_count", 0) or 0)
    threshold = DEFAULT_INDEXING_THRESHOLD
    try:
        t = info.config.optimizer_config.indexing_threshold
        if t:
            threshold = int(t)
    except Exception:  # noqa: BLE001
        pass
    route = "hnsw" if count >= threshold else "brute_force"
    return route, count, threshold


# --------------------------------------------------------------------------- #
# Ingesta de UN documento (generator de eventos)
# --------------------------------------------------------------------------- #
def _ingest_one(client, collection: str, doc_i: int, filename: str, raw: bytes) -> Iterator[dict[str, Any]]:
    def ev(step: str, label: str, status: str, **extra: Any) -> dict[str, Any]:
        return {"event": "step", "doc": doc_i, "filename": filename,
                "step": step, "label": label, "status": status, **extra}

    checksum = document_checksum(raw)

    # 1 · LOAD (pymupdf4llm → markdown por página)
    yield ev("load", "Extrayendo texto (PyMuPDF4LLM)", "running")
    pages = _load_pymupdf4llm(raw, filename)  # [{text, metadata}] por página
    yield ev("load", "Extrayendo texto (PyMuPDF4LLM)", "done", pages=len(pages))

    # 2 · VALIDATE (capa nativa → qué páginas son escaneadas)
    yield ev("validate", "Validando extracción", "running")
    report = validate_extraction(raw)
    yield ev("validate", "Validando extracción", "done",
             doc_verdict=report["doc_verdict"], ocr_pages=len(report["ocr_pages"]))

    # 3 · OCR (español) SOLO en las páginas que lo necesitan → sobrescribe el
    #     auto-OCR en inglés de pymupdf4llm.
    ocr_idx = [i for i in report["ocr_pages"] if 0 <= i < len(pages)]
    if ocr_idx:
        yield ev("ocr", f"OCR español ({len(ocr_idx)} pág.)", "running")
        ocr_text = ocr_pages(raw, ocr_idx, language="spa+eng")
        for idx, txt in ocr_text.items():
            if 0 <= idx < len(pages):
                pages[idx]["text"] = txt
        yield ev("ocr", "OCR español", "done", ocr_pages=len(ocr_text))
    else:
        yield ev("ocr", "OCR (no necesario)", "skipped")

    # 4 · NORMALIZE
    yield ev("normalize", "Normalizando texto", "running")
    pages = normalize_documents(pages)
    yield ev("normalize", "Normalizando texto", "done")

    # 5 · BOILERPLATE
    yield ev("boilerplate", "Quitando encabezados/pies", "running")
    bp = remove_boilerplate([p["text"] for p in pages])
    yield ev("boilerplate", "Quitando encabezados/pies", "done", removed=bp["removed_lines"])

    # 6 · CHUNK (estructural sobre el markdown del doc completo)
    yield ev("chunk", "Fragmentando (chunking)", "running")
    full_md = "\n\n".join(bp["pages"])
    chunks = chunk_markdown(full_md, metadata={"source": filename})
    yield ev("chunk", "Fragmentando (chunking)", "done", chunks=len(chunks))

    if not chunks:
        yield ev("index", "Indexando (sin chunks)", "skipped")
        return

    # 7 · EMBED (OpenAI · text-embedding-ada-002 · mismo modelo que lee el agente)
    yield ev("embed", "Generando embeddings (OpenAI)", "running")
    vectors = embed_documents("openai", [c["text"] for c in chunks])
    yield ev("embed", "Generando embeddings (OpenAI)", "done",
             vectors=int(vectors.shape[0]), dim=int(vectors.shape[1]))

    # 8 · INDEX (Qdrant upsert + reporte de ruta)
    yield ev("index", "Indexando en Qdrant", "running")
    dim = int(vectors.shape[1]) if vectors.size else DEFAULT_DIMENSION
    _ensure_collection(client, collection, dim)
    payloads = build_payloads(chunks, source=filename, doc_id=checksum,
                              tenant_id=collection, checksum=checksum)
    _upsert(client, collection, payloads, vectors)
    route, count, threshold = _index_route(client, collection)
    yield ev("index", "Indexando en Qdrant", "done",
             index_route=route, num_vectors=count, indexing_threshold=threshold)


# --------------------------------------------------------------------------- #
# Ingesta de un LOTE (1..N documentos)
# --------------------------------------------------------------------------- #
def ingest_documents(files: list[tuple[str, bytes]], tenant_input: str) -> Iterator[dict[str, Any]]:
    """Ingesta 1..N PDFs a la colección del tenant, emitiendo eventos de progreso.

    files: [(filename, raw_bytes), ...]. Cada doc que falla NO tumba el lote.
    """
    collection = collection_name(tenant_input)  # valida el nombre (ValueError si inválido)
    client = _qdrant_client()

    yield {"event": "start", "collection": collection, "num_docs": len(files)}

    for i, (filename, raw) in enumerate(files):
        try:
            yield from _ingest_one(client, collection, i, filename, raw)
        except Exception as exc:  # noqa: BLE001  · aislar fallo por documento
            yield {"event": "doc_error", "doc": i, "filename": filename,
                   "error": f"{type(exc).__name__}: {exc}"}

    route, count, threshold = _index_route(client, collection)
    yield {"event": "done", "collection": collection,
           "index_route": route, "num_vectors": count, "indexing_threshold": threshold}

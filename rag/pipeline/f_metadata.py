"""
Pipeline de ingesta RAG · F · METADATA + IDEMPOTENCIA
Ruta: Backend-RAGs/pipeline/f_metadata.py

Paso 5 (ultimo antes de embeddings/Qdrant): a cada chunk le pega la metadata que
viajara al payload de Qdrant y le asigna un ID DETERMINISTA para que reprocesar el
mismo documento haga UPSERT (no duplique).

Piezas:
  - document_checksum(raw)  -> sha256 del PDF. Sirve para DEDUPLICAR (mismo archivo
    subido 2 veces) y para no re-indexar lo ya procesado.
  - chunk_point_id(...)     -> uuid5 estable a partir de (checksum + indice) →
    mismo doc + mismo chunk = mismo ID → upsert idempotente en Qdrant.
  - build_payloads(...)     -> arma [{id, payload}] listo para el vector store,
    con source/doc_id/tenant_id/checksum/chunk_index/ingest_ts/page/section.

NO habla con Qdrant (eso es del orquestador/vector store): este modulo es puro.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

# Namespace fijo del proyecto para los uuid5 (no cambiar: cambiaria todos los IDs).
_NAMESPACE = uuid.UUID("6f9b2c1e-1a2b-4c3d-8e5f-0a1b2c3d4e5f")


def document_checksum(raw: bytes) -> str:
    """sha256 hex del PDF crudo (dedup + idempotencia a nivel documento)."""
    return hashlib.sha256(raw).hexdigest()


def chunk_point_id(checksum: str, chunk_index: int) -> str:
    """ID determinista del vector: mismo (doc, chunk) → mismo ID → upsert."""
    return str(uuid.uuid5(_NAMESPACE, f"{checksum}:{chunk_index}"))


def _extract_section(meta: dict[str, Any]) -> str | None:
    """Reconstruye la ruta de headers de la seccion desde la metadata del chunk."""
    parts = [
        str(v)
        for k, v in meta.items()
        if k.lower().startswith("header") and v
    ]
    return " > ".join(parts) if parts else None


def build_payloads(
    chunks: list[dict[str, Any]],
    *,
    source: str,
    doc_id: str,
    tenant_id: str,
    checksum: str,
    ingest_ts: str | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Devuelve [{id, payload}] para upsert en Qdrant.

    `payload` lleva el texto + la metadata de plataforma + page/section derivadas.
    `ingest_ts`: ISO-8601; si None se genera ahora (UTC).
    """
    ts = ingest_ts or datetime.now(timezone.utc).isoformat()
    out: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        meta = ch.get("metadata") or {}
        payload: dict[str, Any] = {
            "text": ch.get("text", ""),
            "source": source,
            "doc_id": doc_id,
            "tenant_id": tenant_id,
            "checksum": checksum,
            "chunk_index": i,
            "ingest_ts": ts,
            "page": meta.get("page") or meta.get("page_number"),
            "section": _extract_section(meta),
        }
        if extra:
            payload.update(extra)
        out.append({"id": chunk_point_id(checksum, i), "payload": payload})
    return out

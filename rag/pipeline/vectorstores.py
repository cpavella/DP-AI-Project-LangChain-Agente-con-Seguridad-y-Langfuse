"""
Pipeline de ingesta RAG · CLIENTE QDRANT
Ruta: rag/pipeline/vectorstores.py

Módulo glue que provee el paso "index" del orquestador:
  _qdrant_client() -> QdrantClient

Construye el cliente con las MISMAS variables de entorno que el lector del agente
(tools/Base_de_conocimiento.py), resolviendo el puerto: 443 cuando Qdrant está
detrás de un proxy HTTPS (EasyPanel/Nginx/Caddy), 6333 en HTTP directo. Así la
ingesta y la lectura apuntan siempre al mismo endpoint.

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv
from qdrant_client import QdrantClient

load_dotenv(find_dotenv())


def _qdrant_client() -> QdrantClient:
    """Cliente Qdrant apuntando al endpoint del proyecto (REST)."""
    url = os.getenv("QDRANT_URL")
    if not url:
        raise ValueError(
            "❌ Falta QDRANT_URL en .env "
            "(ej. https://tu-servidor-qdrant o http://IP:6333)"
        )
    url = url.rstrip("/")
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 6333)
    return QdrantClient(
        url=url,
        api_key=os.getenv("QDRANT_API_KEY"),
        port=port,
        prefer_grpc=False,  # REST; usa True solo si expones gRPC (6334)
    )

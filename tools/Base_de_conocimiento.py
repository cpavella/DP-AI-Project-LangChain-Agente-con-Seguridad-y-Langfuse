"""
Tool: Base de Conocimiento (RAG con Qdrant)
Permite buscar información en la base de conocimiento de trámites del
Municipio de Girardota (Manual de Trámites).

Consume una colección existente de Qdrant (self-hosted en DigitalOcean) vía
QDRANT_URL + QDRANT_API_KEY. La colección debe haberse creado con embeddings
de 1536 dimensiones (text-embedding-ada-002) y distancia coseno.

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

import os
from urllib.parse import urlparse
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore

load_dotenv(find_dotenv())

# ============================================
# CONFIGURACIÓN DE QDRANT
# ============================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

if not QDRANT_URL:
    raise ValueError(
        "❌ Falta variable QDRANT_URL en .env "
        "(ej. https://tu-servidor-qdrant o http://IP:6333)"
    )

# Normalizar URL y puerto. qdrant-client, si la URL no trae puerto, asume 6333.
# Pero los despliegues detrás de proxy HTTPS (EasyPanel/Nginx/Caddy) exponen el
# 443, no el 6333 → daría "Connection refused". Resolvemos el puerto según el
# esquema: 443 para https sin puerto, 6333 para http sin puerto.
_qdrant_url = QDRANT_URL.rstrip("/")
_parsed = urlparse(_qdrant_url)
_qdrant_port = _parsed.port or (443 if _parsed.scheme == "https" else 6333)

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Conectar a la colección existente de Qdrant.
# from_existing_collection valida que la colección exista y lee su configuración.
vectorstore = QdrantVectorStore.from_existing_collection(
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
    url=_qdrant_url,
    api_key=QDRANT_API_KEY,
    port=_qdrant_port,
    prefer_grpc=False,   # REST; usa True solo si expones gRPC (6334)
    # El pipeline de ingesta (rag/pipeline/f_metadata.py → build_payloads) guarda
    # el texto del chunk en el payload bajo la clave "text", NO bajo el default
    # "page_content" de QdrantVectorStore. Sin esto, similarity_search devuelve
    # documentos con page_content vacío y el agente responde "no se encontró".
    content_payload_key="text",
)


# ============================================
# FUNCIÓN INTERNA DE BÚSQUEDA
# ============================================
def buscar_en_base_conocimiento_interno(query: str, top_k: int = 8) -> str:
    """
    Función interna de búsqueda RAG con Qdrant.

    Args:
        query: Consulta de búsqueda
        top_k: Número de documentos a retornar

    Returns:
        str: Información encontrada formateada
    """
    try:
        docs = vectorstore.similarity_search(query, k=top_k)

        if not docs:
            return "No encontré información relevante en la base de conocimientos."

        contexto = "Información encontrada:\n\n"
        for i, doc in enumerate(docs, 1):
            contexto += f"[{i}]\n{doc.page_content}\n\n"

        return contexto

    except Exception as e:
        return f"Error al buscar: {str(e)}"


# ============================================
# TOOL EXPORTABLE
# ============================================
@tool
def buscar_tramites(consulta: str) -> str:
    """
    Busca información sobre los trámites y servicios del Municipio de Girardota
    en la base de conocimiento oficial (Manual de Trámites).
    Usa esta herramienta cuando el ciudadano pregunte sobre:
    - Requisitos y documentos de un trámite
    - Propósito o en qué consiste un trámite
    - Tiempo de obtención / plazos
    - Costos del trámite
    - Secretaría o dependencia responsable
    - Cualquier información sobre trámites del municipio

    Args:
        consulta: La pregunta o trámite a buscar
    """
    print(f"   🔍 Buscando: '{consulta}'")
    resultado = buscar_en_base_conocimiento_interno(consulta)
    return resultado

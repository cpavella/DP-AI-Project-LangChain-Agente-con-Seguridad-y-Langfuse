"""
Pipeline de ingesta RAG · NAMING de colecciones Qdrant por tenant
Ruta: Backend-RAGs/pipeline/naming.py

Convención (fuente de verdad · server-side):

    nombre_colección = COLLECTION_PREFIX + slug(nombre_que_elige_el_usuario)

Ejemplos:
    "municipalidad_de_machala"  -> "tenant_id_municipalidad_de_machala"
    "Municipalidad de Machala"  -> "tenant_id_municipalidad_de_machala"
    "Cañete #1"                 -> "tenant_id_canete_1"
    "  José's  Café  "          -> "tenant_id_jose_s_cafe"

Por qué saneamos (robustez): no controlamos qué escribe el usuario. Los nombres de
colección de Qdrant deben ser identificadores limpios (minúsculas, ASCII, sin
espacios/símbolos). Si no saneamos: se rompe la creación de la colección, o se
crean nombres duplicados/inconsistentes ("Machala" vs "machala"). El slug además
evita colisiones y respeta el límite de 128 chars del schema de la plataforma
(backend-config.ts · qdrant_collection).

Guardar SIEMPRE el input original como display name (metadata del tenant); el slug
es solo el identificador técnico de la colección.
"""

from __future__ import annotations

import re
import unicodedata

# Prefijo fijo: todas las colecciones empiezan así (decisión de producto).
COLLECTION_PREFIX = "tenant_id_"

# Límite del schema de la plataforma (backend-config.ts: z.string().max(128)).
MAX_COLLECTION_LEN = 128
# Longitud mínima del slug del usuario (evita nombres vacíos o de 1 letra).
MIN_SLUG_LEN = 2


def slugify(raw: str) -> str:
    """Convierte texto libre en un identificador seguro: minúsculas, ASCII, con
    guiones bajos. Quita tildes/ñ (José->jose, Cañete->canete) y colapsa símbolos."""
    if not raw:
        return ""
    # NFKD + drop de diacríticos → ASCII (tildes/ñ fuera; son identificadores, no texto)
    s = unicodedata.normalize("NFKD", raw)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # todo lo no [a-z0-9] → '_'
    s = re.sub(r"_+", "_", s).strip("_")  # colapsa y recorta '_'
    return s


def collection_name(tenant_input: str) -> str:
    """Devuelve el nombre de colección Qdrant a partir del texto del usuario.

    Lanza ValueError si el input queda inutilizable tras sanear (vacío o muy corto).
    """
    slug = slugify(tenant_input)
    if len(slug) < MIN_SLUG_LEN:
        raise ValueError(
            f"El nombre del tenant '{tenant_input}' no es válido: tras sanear queda "
            f"'{slug}' (mínimo {MIN_SLUG_LEN} caracteres alfanuméricos)."
        )
    name = f"{COLLECTION_PREFIX}{slug}"
    return name[:MAX_COLLECTION_LEN]


def is_valid_tenant_input(tenant_input: str) -> bool:
    """True si el input produce un nombre de colección válido (para validar en la UI)."""
    try:
        collection_name(tenant_input)
        return True
    except ValueError:
        return False

"""
Middleware de PII (LangChain nativo) — material didáctico para el Programa AI Engineer.

Este módulo arma el `PIIMiddleware` nativo de LangChain v1 para que se pueda
enchufar a un agente creado con `create_agent(...)`.

¿Por qué un módulo aparte y no dentro del agente?
--------------------------------------------------
`PIIMiddleware` SOLO funciona con `create_agent` (la arquitectura de agente de
LangChain v1). El agente de producción de TramiBot usa un loop manual
(`chat.bind_tools(...).invoke(...)`), donde el middleware NO se ejecuta. Por eso
esta pieza vive suelta y reutilizable en `guardrails/`, y se demuestra en
`demo_pii_middleware.py`.

Diferencia con la Capa 5 (guardrails/pii_detector.py)
-----------------------------------------------------
- Capa 5 (Presidio + spaCy): detección con NLP + score de confianza, pensada
  para BLOQUEAR PII colombiano (cédula, NIT, teléfono CO) ANTES del agente.
- PIIMiddleware (este módulo): detección por REGEX/función, integrada en el
  ciclo del agente. Su valor añadido frente a la Capa 5:
    * Estrategias: además de `block`, puede `redact`, `mask` y `hash`
      (sanea y deja continuar la conversación, mejor UX que rechazar).
    * Cobertura: puede revisar también la SALIDA del modelo y los resultados
      de las tools (apply_to_output / apply_to_tool_results), no solo el input.

Tipos PII integrados en LangChain: email, credit_card (Luhn), ip, mac_address, url.
Todo lo demás (cédula/NIT/teléfono CO) se agrega con `detector` personalizado.

Estrategias disponibles:
    block  → lanza excepción cuando detecta
    redact → reemplaza por [REDACTED_{TIPO}]
    mask   → enmascara parcialmente (ej. ****-****-****-1234)
    hash   → reemplaza por un hash determinista

Requiere: langchain>=1.0.0

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

import re
from typing import Dict, List, Union

from langchain.agents.middleware import PIIMiddleware


# ============================================================
# DETECTORES PERSONALIZADOS (PII colombiano)
# ------------------------------------------------------------
# Un detector custom recibe el texto y devuelve una lista de dicts con las
# claves exactas: {"text", "start", "end"}. Así el middleware sabe QUÉ y DÓNDE
# aplicar la estrategia (redact/mask/hash/block).
# ============================================================

# Reutilizamos los mismos patrones que la Capa 5 para mantener coherencia.
_CEDULA_CO_RE = re.compile(r"\b\d{8,10}\b")
_NIT_CO_RE = re.compile(r"\b\d{9}-?\d\b")
_PHONE_CO_RE = re.compile(r"(?:\+?57)?\s*3\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b")


def _matches(patron: re.Pattern, contenido: str) -> List[Dict[str, Union[str, int]]]:
    """Convierte los matches de un regex al formato que espera PIIMiddleware."""
    return [
        {"text": m.group(0), "start": m.start(), "end": m.end()}
        for m in patron.finditer(contenido)
    ]


def detectar_cedula_co(contenido: str) -> List[Dict[str, Union[str, int]]]:
    """Cédula de Ciudadanía (Colombia): 8-10 dígitos. (Regex puro → posibles falsos positivos.)"""
    return _matches(_CEDULA_CO_RE, contenido)


def detectar_nit_co(contenido: str) -> List[Dict[str, Union[str, int]]]:
    """NIT (Colombia): 9 dígitos + dígito de verificación (con o sin guion)."""
    return _matches(_NIT_CO_RE, contenido)


def detectar_telefono_co(contenido: str) -> List[Dict[str, Union[str, int]]]:
    """Celular colombiano: 3XXXXXXXXX (10 dígitos), con +57 opcional."""
    return _matches(_PHONE_CO_RE, contenido)


# ============================================================
# FACTORY — lista de middlewares lista para create_agent(...)
# ============================================================
def crear_pii_middlewares(
    aplicar_a_salida: bool = False,
    aplicar_a_tools: bool = False,
) -> List[PIIMiddleware]:
    """
    Devuelve la lista de PIIMiddleware configurada para TramiBot.

    Cada middleware maneja UN tipo de PII. El orden importa poco (se aplican
    todos), pero conviene poner los `block` primero por claridad.

    Args:
        aplicar_a_salida: si True, también revisa/sanea la respuesta del modelo.
        aplicar_a_tools:  si True, también revisa/sanea los resultados de tools
                          (útil si tu RAG/Qdrant pudiera devolver PII).

    Returns:
        Lista de PIIMiddleware para pasar a create_agent(middleware=...).

    Demostración de las 4 estrategias:
        - api_key       → block  (corta la ejecución: dato crítico)
        - email         → redact ([REDACTED_EMAIL])
        - cedula / nit  → redact (identificadores colombianos)
        - credit_card   → mask   (****-****-****-1234)
        - phone_co      → mask
        - ip            → hash   (hash determinista)
    """
    comun = {
        "apply_to_input": True,
        "apply_to_output": aplicar_a_salida,
        "apply_to_tool_results": aplicar_a_tools,
    }

    return [
        # ── block: dato crítico, mejor cortar ──────────────────────────
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{20,}",
            strategy="block",
            **comun,
        ),
        # ── redact: reemplazo por etiqueta ─────────────────────────────
        PIIMiddleware("email", strategy="redact", **comun),  # tipo integrado
        PIIMiddleware("cedula_co", detector=detectar_cedula_co, strategy="redact", **comun),
        PIIMiddleware("nit_co", detector=detectar_nit_co, strategy="redact", **comun),
        # ── mask: enmascarado parcial ──────────────────────────────────
        PIIMiddleware("credit_card", strategy="mask", **comun),  # tipo integrado (Luhn)
        PIIMiddleware("phone_co", detector=detectar_telefono_co, strategy="mask", **comun),
        # ── hash: seudonimización determinista ─────────────────────────
        PIIMiddleware("ip", strategy="hash", **comun),  # tipo integrado
    ]

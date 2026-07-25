"""
PiiDetector — Capa 5 de Seguridad: Detección de Datos Personales Sensibles (PII)

Detecta información personal en mensajes de usuarios para Colombia.

Entidades detectadas:
  - Cédula      → Cédula de Ciudadanía (8-10 dígitos)
  - NIT         → Número de Identificación Tributaria (9 dígitos + verificación)
  - Email       → Correo electrónico
  - Teléfono    → Formato colombiano (+57 / 3XXXXXXXXX)
  - Tarjeta     → Números de tarjeta de crédito/débito (13-19 dígitos)

Acción configurable por entidad:
  - "block"  → Bloquea el mensaje completo
  - "mask"   → Enmascara el dato y permite pasar el mensaje (futuro)
  - "off"    → Desactiva la detección de esa entidad

Requiere: presidio-analyzer, presidio-anonymizer (opcional, ver requirements.txt)
Si Presidio no está instalado, funciona con detección pura por regex.

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURACIÓN DE ENTIDADES PII
# Cambiar acción por entidad según necesidad del proyecto
# ============================================================
PII_CONFIG: Dict[str, str] = {
    "CEDULA_CO":    "block",   # Cédula de Ciudadanía — 8-10 dígitos
    "NIT_CO":       "block",   # NIT — 9 dígitos + dígito de verificación
    "EMAIL":        "block",   # Correo electrónico
    "PHONE_CO":     "block",   # Teléfono/celular colombiano
    "CREDIT_CARD":  "block",   # Tarjeta de crédito/débito
}


# ============================================================
# PATRONES REGEX POR ENTIDAD
# ============================================================
_PII_PATTERNS = {
    "CEDULA_CO": re.compile(
        r"\b\d{8,10}\b"
    ),
    "NIT_CO": re.compile(
        r"\b\d{9}-?\d\b"
    ),
    "EMAIL": re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
    ),
    "PHONE_CO": re.compile(
        r"(\+?57)?\s*3\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b"
    ),
    "CREDIT_CARD": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"       # Visa
        r"5[1-5][0-9]{14}|"                     # MasterCard
        r"3[47][0-9]{13}|"                      # Amex
        r"6(?:011|5[0-9]{2})[0-9]{12})\b"       # Discover
    ),
}


class PiiDetector:
    """
    Detecta datos personales sensibles (PII) en mensajes.

    Intenta usar Presidio (Microsoft) si está instalado.
    Si no está instalado, usa detección por regex puro.
    """

    def __init__(self, config: Optional[Dict[str, str]] = None):
        self.config = config or PII_CONFIG
        self._presidio_analyzer = None
        self._init_presidio()

    def _init_presidio(self) -> None:
        """Intenta inicializar Presidio con reconocedores personalizados para Perú."""
        try:
            from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Configuración sin modelo spacy pesado (usa NLP mínimo)
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "es", "model_name": "es_core_news_sm"}],
            })
            nlp_engine = provider.create_engine()

            analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["es", "en"])

            # Reconocedor: Cédula de Ciudadanía (Colombia)
            analyzer.registry.add_recognizer(PatternRecognizer(
                supported_entity="CEDULA_CO",
                patterns=[Pattern("CEDULA_CO", r"\b\d{8,10}\b", 0.7)],
                supported_language="es",
            ))

            # Reconocedor: NIT (Colombia)
            analyzer.registry.add_recognizer(PatternRecognizer(
                supported_entity="NIT_CO",
                patterns=[Pattern("NIT_CO", r"\b\d{9}-?\d\b", 0.85)],
                supported_language="es",
            ))

            # Reconocedor: Teléfono/celular Colombia
            analyzer.registry.add_recognizer(PatternRecognizer(
                supported_entity="PHONE_CO",
                patterns=[Pattern("PHONE_CO", r"(\+?57)?\s*3\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b", 0.75)],
                supported_language="es",
            ))

            self._presidio_analyzer = analyzer
            logger.info("[PII] Presidio inicializado con reconocedores para Colombia (ES)")

        except ImportError:
            logger.info("[PII] Presidio no instalado — usando detección por regex puro")
        except Exception as e:
            logger.warning(f"[PII] Presidio no disponible ({e}) — usando regex puro")

    def detectar(self, texto: str) -> List[str]:
        """
        Analiza el texto y retorna lista de entidades PII detectadas
        cuya acción es 'block'.

        Returns:
            Lista de strings con los tipos detectados, ej: ["CEDULA_CO", "EMAIL"]
            Lista vacía si el mensaje es seguro.
        """
        if not texto or not texto.strip():
            return []

        activas = [k for k, v in self.config.items() if v == "block"]
        detectadas: List[str] = []

        if self._presidio_analyzer:
            detectadas = self._detectar_con_presidio(texto, activas)
        else:
            detectadas = self._detectar_con_regex(texto, activas)

        return detectadas

    def _detectar_con_presidio(self, texto: str, entidades: List[str]) -> List[str]:
        """Usa Presidio para detectar PII con NLP + regex."""
        try:
            resultados = self._presidio_analyzer.analyze(
                text=texto,
                language="es",
                entities=entidades,
            )
            detectadas = list({r.entity_type for r in resultados if r.score >= 0.6})
            if detectadas:
                logger.warning(f"[PII Presidio] Detectado: {detectadas}")
            return detectadas
        except Exception as e:
            logger.error(f"[PII Presidio] Error en análisis: {e}")
            return self._detectar_con_regex(texto, entidades)

    def _detectar_con_regex(self, texto: str, entidades: List[str]) -> List[str]:
        """Fallback: detección pura por regex."""
        detectadas = []
        for entidad in entidades:
            pattern = _PII_PATTERNS.get(entidad)
            if pattern and pattern.search(texto):
                logger.warning(f"[PII Regex] {entidad} detectado en: {texto[:60]!r}")
                detectadas.append(entidad)
        return detectadas

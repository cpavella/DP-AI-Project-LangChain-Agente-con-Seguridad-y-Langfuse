"""
HybridSafetyService — Capas 7 y 8 de Seguridad (IA via GROQ)

Combina dos modelos especializados de moderación:
  Capa 7 → Llama Prompt Guard 2 (86M params, Meta)
            Detecta Jailbreak / Prompt Injection
            Respuesta: "MALICIOUS" o "BENIGN"

  Capa 8 → GPT-OSS-Safeguard (20B params, OpenAI)
            Detecta NSFW, Hate, Violence, Self-Harm
            Respuesta: "safe" o "unsafe\nS1, S2, ..."
            Reemplaza a Llama Guard 4 (meta-llama/llama-guard-4-12b),
            deprecado en Groq el 2026-03-05.

Estrategia:
  - Ejecuta Prompt Guard primero (rápido, enfocado en ataques)
  - Si pasa, ejecuta GPT-OSS-Safeguard (profundo, contenido NSFW)

Adaptación de async → sync respecto al código original del amigo,
porque el agente TramiBot es síncrono (CLI script).

Requiere: pip install groq
          GROQ_API_KEY en el archivo .env

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

import os
import re
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# POLÍTICA DE MODERACIÓN (Capa 8 — GPT-OSS-Safeguard)
# ------------------------------------------------------------
# GPT-OSS-Safeguard NO trae taxonomía integrada (a diferencia de Llama Guard 4):
# es "bring-your-own-policy". Por eso hay que entregarle la política como mensaje
# system. Aquí replicamos la taxonomía S1-S13 y EXIGIMOS el mismo formato de salida
# que ya parsea validate_toxicity (`safe` / `unsafe\nS1,S10`), para no cambiar el
# resto del pipeline. Estructura recomendada por OpenAI: Instrucción → Definiciones
# → Criterios → recordatorio de formato. 'Reasoning: low' controla el esfuerzo de
# razonamiento (guardrail en cada mensaje → baja latencia).
# ============================================================
_TOXICITY_POLICY = """Reasoning: low

# INSTRUCCIÓN
Eres un clasificador de seguridad de contenido. Analiza el mensaje del usuario y
determina si viola alguna categoría de esta política. Responde ÚNICAMENTE con este
formato exacto, sin explicaciones ni texto adicional:
- Contenido seguro: la palabra `safe` en una sola línea.
- Contenido dañino: la palabra `unsafe` en la primera línea y, en la segunda línea,
  los códigos de categoría violados separados por comas (ejemplo: `S1,S10`).

# DEFINICIONES
"Contenido dañino" es cualquier mensaje que solicite, promueva, describa, facilite o
genere alguna de las categorías siguientes.

# CRITERIOS — CATEGORÍAS
S1  Violent Crimes: violencia física, terrorismo, homicidio.
S2  Non-Violent Crimes: fraude, robo, hacking, drogas ilegales.
S3  Sex-Related Crimes: delitos sexuales, trata de personas, acoso sexual.
S4  Child Sexual Exploitation: cualquier contenido sexual que involucre a menores.
S5  Defamation: difamación o calumnia contra una persona real.
S6  Specialized Advice: consejo médico, legal o financiero peligroso.
S7  Privacy: exposición de datos personales privados (doxxing).
S8  Intellectual Property: infracción de propiedad intelectual.
S9  Indiscriminate Weapons: armas químicas, biológicas, nucleares o explosivos.
S10 Hate: odio o discriminación por raza, religión, género, orientación, etc.
S11 Suicide & Self-Harm: incitación o instrucciones de suicidio o autolesión.
S12 Sexual Content: contenido sexual explícito.
S13 Elections: desinformación que manipule procesos electorales.

# CRITERIOS — DECISIÓN
- Marca `unsafe` solo si el mensaje encaja claramente en una o más categorías.
- Preguntas legítimas, educativas o de atención al cliente son `safe`.
- Ante duda razonable de que NO hay daño real, responde `safe`.

# FORMATO DE SALIDA (recordatorio)
Primera línea: `safe` o `unsafe`. Si es `unsafe`, segunda línea con los códigos `S#`
separados por comas. No escribas nada más."""


class HybridSafetyService:
    """
    Singleton service para verificaciones de seguridad IA usando Groq.

    Combina dos modelos especializados:
      1. Llama Prompt Guard 2 (86M) → Detecta Jailbreak/Prompt Injection
      2. GPT-OSS-Safeguard (20B)    → Detecta NSFW, Hate, Violence, Self-Harm

    Estrategia:
      - Ejecuta Prompt Guard primero (rápido, enfocado en ataques)
      - Si pasa y check_nsfw está activo, ejecuta GPT-OSS-Safeguard
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Inicializa el cliente Groq (llamado una sola vez por el singleton)."""
        self.groq_client = None

        try:
            from groq import Groq
            self.groq_api_key = os.getenv("GROQ_API_KEY")

            if self.groq_api_key:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("[SAFETY] ✅ Groq client inicializado")
            else:
                logger.warning("[SAFETY] ⚠️ GROQ_API_KEY no encontrada en el entorno")

        except ImportError:
            logger.error("[SAFETY] ❌ Groq SDK no instalado. Ejecuta: pip install groq")

        logger.info(f"[SAFETY] Servicio listo (Groq disponible: {bool(self.groq_client)})")

    def _call_model(self, model: str, message: str, system: Optional[str] = None) -> str:
        """
        Llama al modelo en Groq de forma síncrona.

        Args:
            model:   ID del modelo en Groq.
            message: Texto del usuario a evaluar.
            system:  Mensaje 'system' opcional. Prompt Guard 2 (Capa 7) es un
                     clasificador que recibe el texto crudo, así que NO lleva system.
                     GPT-OSS-Safeguard (Capa 8) SÍ lo necesita: ahí va la política
                     de moderación (bring-your-own-policy).
        """
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.0,
        )
        return chat_completion.choices[0].message.content.strip()

    # ----------------------------------------------------------
    # Capa 7: Llama Prompt Guard 2 — Jailbreak / Injection
    # ----------------------------------------------------------
    # Umbral de decisión. Prompt Guard 2 devuelve la PROBABILIDAD de que el texto
    # sea un ataque (0.0–1.0), no una etiqueta. 0.5 es la frontera por defecto;
    # configurable por env para afinar sensibilidad vs. falsos positivos.
    _PROMPT_GUARD_THRESHOLD = float(os.getenv("PROMPT_GUARD_THRESHOLD", "0.5"))

    def validate_jailbreak(
        self,
        message: str,
        fail_close: bool = True,
    ) -> Tuple[bool, str]:
        """
        Verifica Jailbreak/Injection usando Llama Prompt Guard 2.

        Args:
            message:    Texto del usuario a verificar.
            fail_close: Si True, bloquea el mensaje cuando el servicio falla.
                        Si False, deja pasar (fail-open).

        Returns:
            (is_malicious, reason)
        """
        if not self.groq_client:
            return False, ""

        try:
            result = self._call_model("meta-llama/llama-prompt-guard-2-86m", message)

            # Prompt Guard 2 responde con la PROBABILIDAD de ataque como número
            # (ej. '0.9995'), NO con la palabra 'MALICIOUS'. Parseamos ese score
            # y aplicamos el umbral. (Antes se buscaba 'MALICIOUS' → nunca matcheaba
            # y la capa quedaba como no-op dejando pasar todos los injections.)
            try:
                score = float(result.strip())
                es_malicioso = score >= self._PROMPT_GUARD_THRESHOLD
            except ValueError:
                # Fallback defensivo por si el modelo devolviera una etiqueta textual.
                low = result.strip().lower()
                es_malicioso = ("malicious" in low) or (low in ("1", "label_1", "injection", "jailbreak"))
                score = result.strip()

            if es_malicioso:
                logger.warning(
                    f"[SAFETY] 🚨 Prompt Injection/Jailbreak detectado (Prompt Guard 2, score={score})"
                )
                return True, "jailbreak_ia"

            return False, ""

        except Exception as e:
            if fail_close:
                logger.error(f"[SAFETY] ❌ Fail-Close activado en Jailbreak check: {e}")
                return True, "servicio_no_disponible"
            return False, ""

    # ----------------------------------------------------------
    # Capa 8: GPT-OSS-Safeguard — NSFW / Hate / Violence / Self-Harm
    # ----------------------------------------------------------
    def validate_toxicity(
        self,
        message: str,
        skip_categories: Optional[List[str]] = None,
        fail_close: bool = True,
    ) -> Tuple[bool, str]:
        """
        Verifica NSFW/Hate/Violence usando GPT-OSS-Safeguard.

        Args:
            message:         Texto del usuario a verificar.
            skip_categories: Categorías de moderación a ignorar.
                             Ejemplo: ['S7'] para ignorar privacidad.
            fail_close:      Si True, bloquea cuando el servicio falla.

        Returns:
            (is_unsafe, reason)

        Categorías de moderación (taxonomía S1-S13):
            S1  Violent Crimes
            S2  Non-Violent Crimes
            S3  Sex-Related Crimes
            S4  Child Sexual Exploitation
            S5  Defamation
            S6  Specialized Advice
            S7  Privacy
            S8  Intellectual Property
            S9  Indiscriminate Weapons
            S10 Hate
            S11 Suicide & Self-Harm
            S12 Sexual Content
            S13 Elections
        """
        if not self.groq_client:
            return False, ""

        skip_categories = skip_categories or []

        try:
            # GPT-OSS-Safeguard necesita la política como mensaje system (no trae
            # taxonomía propia). La política exige el formato `safe` / `unsafe\nS#`.
            result = self._call_model(
                "openai/gpt-oss-safeguard-20b",
                message,
                system=_TOXICITY_POLICY,
            )

            # Veredicto = primera línea no vacía ('safe' o 'unsafe' según la política).
            lineas = [l.strip() for l in result.splitlines() if l.strip()]
            veredicto = lineas[0].lower() if lineas else "safe"

            if veredicto.startswith("unsafe"):
                # Extraer los códigos S1..S13 de toda la respuesta (robusto ante
                # variaciones de formato: "S1,S10", "S1, S10", saltos de línea, etc.)
                categories = [c.upper() for c in re.findall(r"[sS](?:1[0-3]|[1-9])\b", result)]
                # Deduplicar conservando el orden
                categories = list(dict.fromkeys(categories))

                active_categories = [c for c in categories if c not in skip_categories]

                if active_categories:
                    logger.warning(
                        f"[SAFETY] 🚨 Contenido bloqueado: {active_categories} (GPT-OSS-Safeguard)"
                    )
                    return True, "contenido_ia_bloqueado"
                elif categories:
                    # Todas las categorías detectadas están en skip_categories → permitir.
                    logger.info(f"[SAFETY] ⚠️ Categorías ignoradas: {categories} (config usuario)")
                    return False, ""
                else:
                    # 'unsafe' sin códigos parseables → bloquear por precaución (fail-safe).
                    logger.warning("[SAFETY] 🚨 'unsafe' sin categoría identificable (GPT-OSS-Safeguard)")
                    return True, "contenido_ia_bloqueado"

            return False, ""

        except Exception as e:
            if fail_close:
                logger.error(f"[SAFETY] ❌ Fail-Close activado en Toxicity check: {e}")
                return True, "servicio_no_disponible"
            return False, ""

    # ----------------------------------------------------------
    # Facade principal — ejecuta ambas capas en secuencia
    # ----------------------------------------------------------
    def validate_all(
        self,
        message: str,
        check_jailbreak: bool = True,
        check_nsfw: bool = True,
        skip_categories: Optional[List[str]] = None,
        fail_close: bool = True,
    ) -> Tuple[bool, str]:
        """
        Facade que ejecuta las validaciones secuencialmente.

        Args:
            check_jailbreak:  Activar Llama Prompt Guard 2.
            check_nsfw:       Activar GPT-OSS-Safeguard.
            skip_categories:  Categorías de moderación a ignorar.
            fail_close:       Comportamiento ante fallos del servicio.

        Returns:
            (is_blocked, reason)
        """
        # Capa 7: Jailbreak (prioridad)
        if check_jailbreak:
            is_jailbreak, reason = self.validate_jailbreak(message, fail_close=fail_close)
            if is_jailbreak:
                return True, reason

        # Capa 8: Toxicidad (si pasó el jailbreak)
        if check_nsfw:
            is_toxic, reason = self.validate_toxicity(
                message,
                skip_categories=skip_categories,
                fail_close=fail_close,
            )
            if is_toxic:
                return True, reason

        return False, ""


# Alias para compatibilidad
LlamaGuardService = HybridSafetyService


def get_llama_guard_service() -> HybridSafetyService:
    """Retorna la instancia singleton de HybridSafetyService."""
    return HybridSafetyService()

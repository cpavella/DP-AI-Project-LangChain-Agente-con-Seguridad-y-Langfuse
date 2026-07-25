"""
Módulo de evaluación automática para TramiBot.

Contiene evaluadores desacoplados del agente principal.
Actualmente incluye LLM-as-a-Judge; se pueden agregar más en el futuro.
"""

from evaluation.llm_judge import evaluar_con_llm_judge

__all__ = [
    "evaluar_con_llm_judge",
]

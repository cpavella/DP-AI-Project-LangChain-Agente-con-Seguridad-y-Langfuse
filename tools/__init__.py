"""
Módulo de Tools para Agentes IA
Contiene todas las herramientas disponibles para los agentes.
"""

from tools.Base_de_conocimiento import buscar_tramites
from tools.Busqueda_internet import buscar_internet
from tools.Hora_y_fecha import obtener_fecha_hora
from tools.Transferir_humano import crear_tool_transferir_humano

# Lista de todas las tools disponibles
__all__ = [
    "buscar_tramites",
    "buscar_internet",
    "obtener_fecha_hora",
    "crear_tool_transferir_humano",
]

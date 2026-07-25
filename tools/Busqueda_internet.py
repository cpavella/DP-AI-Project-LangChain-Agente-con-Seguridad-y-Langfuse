"""
Tool: Búsqueda en Internet (Tavily)
Permite buscar información actualizada en internet.

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool

load_dotenv(find_dotenv())

# ============================================
# CONFIGURACIÓN DE TAVILY
# ============================================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError(
        "❌ Falta TAVILY_API_KEY en .env\n"
        "Obtén tu API key gratis en: https://tavily.com"
    )

# Usar la nueva API de langchain-tavily
try:
    from langchain_tavily import TavilySearch
    tavily_search = TavilySearch(max_results=5)
except ImportError:
    # Fallback a la versión antigua si no está instalada
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily_search = TavilySearchResults(max_results=5)


# ============================================
# TOOL EXPORTABLE
# ============================================
@tool
def buscar_internet(consulta: str) -> str:
    """
    Busca información pública actualizada sobre el Municipio de Girardota en
    internet usando Tavily (p. ej. datos de contacto, direcciones, horarios de
    atención o comunicados oficiales que no estén en la base de conocimiento).
    Usa esta herramienta ÚNICAMENTE para complementar información del municipio.

    NUNCA uses esta herramienta para:
    - Preguntas de cultura general (política, deportes, noticias, ciencia)
    - Temas no relacionados con el Municipio de Girardota o sus trámites

    Args:
        consulta: El aspecto del Municipio de Girardota a buscar
    """
    # Forzar que la búsqueda siempre esté en el contexto del municipio
    consulta_municipio = f"Municipio de Girardota {consulta}"
    print(f"   🌐 Buscando en internet: '{consulta_municipio}'")

    try:
        # Ejecutar búsqueda (siempre con contexto del municipio)
        resultados = tavily_search.invoke(consulta_municipio)
        
        if not resultados:
            return "No encontré información relevante en internet."
        
        # Formatear resultados
        respuesta = "Información encontrada en internet:\n\n"
        
        # Manejar diferentes formatos de respuesta
        if isinstance(resultados, list):
            for i, resultado in enumerate(resultados, 1):
                if isinstance(resultado, dict):
                    titulo = resultado.get("title", "Sin título")
                    contenido = resultado.get("content", "")
                    url = resultado.get("url", "")
                else:
                    titulo = f"Resultado {i}"
                    contenido = str(resultado)
                    url = ""
                
                respuesta += f"[{i}] {titulo}\n"
                respuesta += f"{contenido[:500]}...\n" if len(contenido) > 500 else f"{contenido}\n"
                if url:
                    respuesta += f"Fuente: {url}\n"
                respuesta += "\n"
        else:
            respuesta += str(resultados)
        
        return respuesta
        
    except Exception as e:
        return f"Error al buscar en internet: {str(e)}"

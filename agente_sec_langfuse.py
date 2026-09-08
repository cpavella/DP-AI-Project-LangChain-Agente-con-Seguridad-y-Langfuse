"""
Agente IA Completo: Base de Conocimiento + Internet + Histórico + Langfuse
- Tool 1: Base de Conocimiento (RAG con Qdrant)
- Tool 2: Búsqueda en Internet (Tavily)
- Histórico: Guarda conversaciones en PostgreSQL
- Observabilidad: Langfuse (trazas, tokens, costos, latencia)

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

import os
import sys
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Agregar el directorio actual al path para importar tools (portable para despliegue)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# LANGFUSE ▶ Importar decorador @observe y propagación de atributos
from langfuse import observe, propagate_attributes
# LANGFUSE ▶ Importar el CallbackHandler que intercepta todas las llamadas de LangChain
from langfuse.langchain import CallbackHandler

# Importar tools desde la carpeta tools/
from tools.Base_de_conocimiento import buscar_tramites
from tools.Hora_y_fecha import obtener_fecha_hora
# buscar_internet desactivada para el agente municipal (ver lista `tools` más abajo).
# from tools.Busqueda_internet import buscar_internet

# Importar guardrail de entrada (Capa 1 de Seguridad)
from guardrails.input_guardrail import verificar_input_guardrail, respuesta_bloqueada

# Importar el PIIMiddleware nativo de LangChain (guardrails/middleware.py)
from guardrails.middleware import crear_pii_middlewares

# Importar evaluador LLM-as-a-Judge (módulo evaluation/)
from evaluation.llm_judge import evaluar_con_llm_judge

# Importar histórico de conversación (PostgreSQL) desde chat_history/
from chat_history import crear_tabla_historial, get_session_history

# Importar config del modelo (desacoplada del código) desde model_config/
from model_config import load_model_config

# Importar el system prompt desde YAML (desacoplado del código) desde prompt/
from prompt import load_system_prompt

# LANGFUSE ▶ Cliente Langfuse singleton (Langfuse() + get_client() → observability/)
from observability.langfuse_setup import langfuse_client

# ============================================
# 2. LISTA DE TOOLS DISPONIBLES
# ============================================
tools = [
    buscar_tramites,      # Base de conocimiento de trámites del Municipio de Girardota
    obtener_fecha_hora,   # Fecha y hora actual por zona horaria (plazos/días hábiles)
    # buscar_internet,    # Desactivada: el agente municipal responde solo desde el
                          # Manual oficial (evita alucinaciones de la web abierta).
]

# ============================================
# 3. CONFIGURACIÓN DEL MODELO CON TOOLS
# ============================================
# La config del LLM vive en model_config/model.yaml
_model_cfg = load_model_config()
chat = init_chat_model(
    _model_cfg["llm"]["model"],
    temperature=_model_cfg["llm"]["temperature"],
)

# ============================================
# 4. PROMPT DEL AGENTE + CONTEXTO FECHA/HORA
# ============================================
AGENT_TIMEZONE = os.getenv("AGENT_TIMEZONE", "America/Lima")


def _contexto_fecha_hora() -> str:
    """Fecha y hora actual para inyectar en el system prompt (cada turno)."""
    try:
        tz = ZoneInfo(AGENT_TIMEZONE)
    except Exception:
        tz = ZoneInfo("America/Lima")
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S") + f" (zona {AGENT_TIMEZONE})"



# El system prompt del agente (TramiBot · Municipio de Girardota) vive en
# prompt/system_prompt.yaml y se carga con load_system_prompt() más abajo.



# ============================================
# OPCIÓN ACTIVA: PROMPT DESDE YAML (prompt/system_prompt.yaml)
# --------------------------------------------
# El system prompt vive desacoplado en prompt/system_prompt.yaml y se carga con
# load_system_prompt(). Para ajustar la persona del agente solo se edita el YAML.
# ============================================
#system_prompt = load_system_prompt()
#print("📝 Prompt cargado desde YAML: prompt/system_prompt.yaml")

# ============================================
# ALTERNATIVA: PROMPT DESDE LANGFUSE (Prompt Management)
# --------------------------------------------
# Para usarlo:
#   1. Ve a Langfuse UI → Prompts → Create Prompt
#   2. Nombre: "Prompt-del-Agente-para-Whatsapp-v2"   Tipo: text
#   3. Pega el contenido del system_prompt del YAML
#   4. Asigna el label "production"
#   5. Comenta las 2 líneas de load_system_prompt() de arriba
#   6. Descomenta el bloque de abajo
#
# Ventaja: puedes cambiar el prompt desde la UI sin tocar código ni reiniciar el servidor
# Desventaja: el prompt sale del repo y no puedes editarlo con vibe coding en Cursor
# ============================================
lf_prompt = langfuse_client.get_prompt("Prompt-del-Agente-Tramites-Girardota")
system_prompt = lf_prompt.compile()   # sin variables; si tuvieras usa compile(var=valor)
print(f"📝 Prompt cargado desde Langfuse: versión {lf_prompt.version}")

# ============================================
# 5. CREAR TABLA DE HISTORIAL (chat_history/)
# ============================================
# El backend de persistencia vive en chat_history/postgres_store.py
crear_tabla_historial()

# ============================================
# 6. FUNCIÓN DE CHAT CON AGENTE + TOOLS + LANGFUSE
# ============================================
# LANGFUSE ▶ @observe() convierte esta función en un Trace raíz en Langfuse.
#             Cada llamada a chat_con_agente() generará una traza independiente
#             que agrupa todas las observaciones del turno (LLM calls, tools, etc.)
@observe()
def chat_con_agente(
    mensaje_usuario: str,
    session_id: str,
    tools_extra: list | None = None,
) -> str:
    """
    Ejecuta el agente con tools y memoria.
    El agente decide si usar herramientas o responder directamente.

    Args:
        mensaje_usuario: Mensaje del usuario.
        session_id:      UUID de la sesión/conversación (para historial).
        tools_extra:     Tools adicionales por turno (ej. transferir_a_humano
                         con contact_id inyectado desde el webhook).
    """
    # ── Capa 1 de Seguridad: Guardrail de Entrada ──────────────────
    es_seguro, motivo = verificar_input_guardrail(mensaje_usuario)
    if not es_seguro:
        print(f"🚨 [GUARDRAIL] Mensaje bloqueado. Motivo: {motivo}")
        # LANGFUSE v4 ▶ propagate_attributes() reemplaza update_current_trace() para
        #               atributos de correlación (tags, session_id, etc.)
        with propagate_attributes(
            trace_name="tramibot-guardrail-bloqueado",  # LANGFUSE v4 ▶ 'name' ahora es 'trace_name'
            session_id=session_id,
            tags=["guardrail", "bloqueado"],
        ):
            pass
        # LANGFUSE v4 ▶ set_current_trace_io()
        langfuse_client.set_current_trace_io(
            input={"mensaje": mensaje_usuario},
            output={"bloqueado": "true", "motivo": motivo},
        )
        return respuesta_bloqueada(motivo)
    # ───────────────────────────────────────────────────────────────

    # Combinar tools base con tools dinámicas del turno
    tools_turno = tools + (tools_extra or [])

    # Inyectar la fecha/hora actual en el system prompt de este turno
    system_content = (
        system_prompt
        + "\n\n---\nFECHA Y HORA ACTUAL (referencia para este turno): "
        + _contexto_fecha_hora()
    )

    # Agente LangChain v1: create_agent maneja el loop de tools internamente
    # y ejecuta el PIIMiddleware (Capa 5b) antes/después de llamar al modelo.
    # Se construye por turno para refrescar la fecha/hora y las tools dinámicas.
    agente = create_agent(
        model=chat,
        tools=tools_turno,
        system_prompt=system_content,
        middleware=crear_pii_middlewares(),
    )

    # LANGFUSE ▶ CallbackHandler que intercepta cada llamada del grafo (LLM + tools).
    #             Al crearse dentro de @observe(), anida sus observaciones bajo el trace padre.
    langfuse_handler = CallbackHandler()

    # Memoria: historial de Postgres cargado como mensajes (sin checkpointer,
    # para conservar el backend actual de chat_history/).
    history = get_session_history(session_id)
    input_messages = list(history.messages) + [HumanMessage(content=mensaje_usuario)]

    # LANGFUSE v4 ▶ propagate_attributes() es el único método para atributos de correlación.
    #               En v4, 'update_current_trace()' fue eliminado; los atributos ahora viven
    #               en cada observación hija, no solo en el trace padre.
    #               trace_name: nombre visible en el dashboard (antes era 'name' en update_current_trace)
    #               metadata debe ser dict[str, str] con valores ≤ 200 chars (restricción de v4)
    with propagate_attributes(
        trace_name="tramibot-turno",                         # LANGFUSE v4 ▶ nombre del trace en el dashboard
        session_id=session_id,                               # LANGFUSE v4 ▶ agrupa trazas por conversación
        user_id=f"conv-{session_id[:8]}",                    # LANGFUSE v4 ▶ identifica al usuario en métricas
        tags=["produccion", "chatwoot", "tramibot"],         # LANGFUSE v4 ▶ etiquetas para filtrar
        metadata={"modelo": _model_cfg["llm"]["model"]},     # LANGFUSE v4 ▶ dict[str,str] obligatorio en v4
    ):
        # LANGFUSE ▶ config={"callbacks": [langfuse_handler]} activa el tracing de TODO el
        #             grafo del agente: llamadas al LLM, ejecución de tools y respuesta final
        #             (tokens, costo y latencia de cada paso).
        resultado = agente.invoke(
            {"messages": input_messages},
            config={"callbacks": [langfuse_handler]},
        )
        # create_agent devuelve el estado final; el último mensaje es la respuesta.
        respuesta_final = resultado["messages"][-1].content

    # LANGFUSE v4 ▶ set_current_trace_io()
    #               Registra el input/output del turno completo a nivel del trace raíz.
    langfuse_client.set_current_trace_io(
        input={"mensaje_usuario": mensaje_usuario},   # LANGFUSE v4 ▶ input del trace principal
        output={"respuesta": respuesta_final},        # LANGFUSE v4 ▶ output del trace principal
    )

    # JUDGE ▶ Obtener el trace_id del turno actual (disponible dentro del contexto @observe())
    #          y disparar la evaluación LLM-as-a-Judge (módulo evaluacion/llm_judge.py).
    #          Los scores quedan vinculados a este trace en el dashboard de Langfuse.
    trace_id_actual = langfuse_client.get_current_trace_id()
    if trace_id_actual:
        evaluar_con_llm_judge(mensaje_usuario, respuesta_final, trace_id_actual)

    # Guardar en historial
    history.add_user_message(mensaje_usuario)
    history.add_ai_message(respuesta_final)

    return respuesta_final


# ============================================
# 7. LOOP DE CONVERSACIÓN
# ============================================
def main():
    print("=" * 60)
    print("🤖 TramiBot - Agente de Trámites del Municipio de Girardota")
    print("=" * 60)
    print("🔧 Tools disponibles:")
    for t in tools:
        print(f"   - {t.name}")
    print("💾 Historial: PostgreSQL")
    
    # Menú de sesión
    print("\nOpciones de sesión:")
    print("  1. Nueva conversación")
    print("  2. Continuar sesión existente (pegar UUID)")
    
    opcion = input("\nElige (1/2): ").strip()
    
    if opcion == "2":
        session_id = input("Pega el UUID de la sesión: ").strip()
        try:
            uuid.UUID(session_id)
        except ValueError:
            print("⚠️ UUID inválido. Creando nueva sesión...")
            session_id = str(uuid.uuid4())
    else:
        session_id = str(uuid.uuid4())
    
    print(f"\n📝 Session ID: {session_id}")
    print("   (Guarda este ID para continuar después)")
    print("✅ El agente responde trámites del Municipio de Girardota (base de conocimiento)")
    print("Escribe 'salir' para volver al menú.\n")
    
    while True:
        usuario = input("Tú: ").strip()
        
        if usuario.lower() in ['salir', 'exit', 'quit']:
            print(f"\n💾 Tu sesión está guardada.")
            print(f"   UUID: {session_id}")
            # LANGFUSE ▶ flush() fuerza el envío de todos los eventos pendientes a Langfuse
            #             antes de cerrar el proceso. Esencial en scripts de vida corta.
            langfuse_client.flush()
            print("👋 ¡Hasta luego!")
            break
        
        if not usuario:
            continue
        
        try:
            respuesta = chat_con_agente(usuario, session_id)
            print(f"\n🤖 TramiBot: {respuesta}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()

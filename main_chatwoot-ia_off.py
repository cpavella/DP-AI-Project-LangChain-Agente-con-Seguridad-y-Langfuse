"""
Integración del Agente IA con Chatwoot
Webhook para recibir mensajes y responder automáticamente.

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

import os
import sys
import uuid
import asyncio
import logging
import requests
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request
import uvicorn

# Mostrar logs INFO de guardrails en consola
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Cargar variables de entorno
load_dotenv(find_dotenv())

# Agregar el directorio actual al path (portable para despliegue)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar directamente el agente (sin rutas locales)
#from agente_sec_langfuse import chat_con_agente
from agente_sec_langfuse import chat_con_agente


# Importar factory de la tool de handoff
from tools.Transferir_humano import crear_tool_transferir_humano

# Buffer de mensajes (concatena mensajes seguidos en una sola respuesta)
from message_buffer import encolar_mensaje, BUFFER_ENABLED, BUFFER_WINDOW_SECONDS

print("🤖 Cargando TramiBot (RAG con Qdrant)...")
print("✅ TramiBot cargado correctamente")

# ============================================
# CONFIGURACIÓN DE CHATWOOT
# ============================================
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_ACCESS_TOKEN")

# Etiqueta que activa el bot (opcional, para handoff)
BOT_LABEL = os.getenv("CHATWOOT_BOT_LABEL", "atiende-ia")
# Etiqueta que desactiva la IA: si el usuario/conversación tiene "ia-off", el agente NO responde
TAG_IA_OFF = "ia-off"

if not all([CHATWOOT_BASE_URL, CHATWOOT_ACCOUNT_ID, CHATWOOT_API_TOKEN]):
    print("⚠️  ADVERTENCIA: Faltan variables de Chatwoot en .env")
    print("   Requeridas: CHATWOOT_BASE_URL, CHATWOOT_ACCOUNT_ID, CHATWOOT_API_ACCESS_TOKEN")
else:
    print(f"✅ Chatwoot configurado: {CHATWOOT_BASE_URL}")

# ============================================
# FUNCIONES DE CHATWOOT
# ============================================
def send_chatwoot_message(conversation_id: int, message: str) -> bool:
    """
    Envía un mensaje de respuesta a una conversación en Chatwoot.
    
    Args:
        conversation_id: ID de la conversación
        message: Mensaje a enviar
    
    Returns:
        True si se envió correctamente, False si hubo error
    """
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {
        'api_access_token': CHATWOOT_API_TOKEN,
        'Content-Type': 'application/json'
    }
    payload = {
        'content': message,
        'message_type': 'outgoing'
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"   ✅ Mensaje enviado a conversación {conversation_id}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error al enviar mensaje: {e}")
        return False


def update_chatwoot_labels(conversation_id: int, labels: list) -> bool:
    """
    Actualiza las etiquetas de una conversación en Chatwoot.
    
    Args:
        conversation_id: ID de la conversación
        labels: Lista de etiquetas
    
    Returns:
        True si se actualizó correctamente
    """
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/labels"
    headers = {
        'api_access_token': CHATWOOT_API_TOKEN,
        'Content-Type': 'application/json'
    }
    payload = {'labels': labels}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"   ✅ Etiquetas actualizadas: {labels}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error al actualizar etiquetas: {e}")
        return False


def conversation_id_to_uuid(conversation_id: int) -> str:
    """
    Convierte un conversation_id de Chatwoot a un UUID válido.
    Esto permite usar el mismo session_id para la misma conversación.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chatwoot-{conversation_id}"))


def ejecutar_agente_y_responder(
    conversation_id: int,
    contact_id: int,
    mensaje: str,
) -> str:
    """
    Ejecuta el agente para un mensaje (o la concatenación de varios) y envía
    la respuesta a Chatwoot. Es síncrono: se ejecuta en un hilo aparte para
    no bloquear el event loop de FastAPI.
    """
    session_id = conversation_id_to_uuid(conversation_id)

    # Crear la tool de handoff con contact_id y conversation_id inyectados
    tools_extra = []
    if contact_id and conversation_id:
        tools_extra.append(crear_tool_transferir_humano(contact_id, conversation_id))

    respuesta = chat_con_agente(mensaje, session_id, tools_extra=tools_extra)
    print(f"   ✅ Respuesta generada ({len(respuesta)} chars)")

    send_chatwoot_message(conversation_id, respuesta)
    return respuesta


# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="TramiBot - Agente de Trámites (Municipio de Girardota) con Chatwoot",
    description="Webhook para integrar el agente TramiBot con Chatwoot",
    version="1.0.0"
)


@app.post("/webhook")
async def chatwoot_webhook(request: Request):
    """
    Endpoint que recibe los webhooks de Chatwoot.
    Procesa mensajes entrantes y responde usando TramiBot.
    """
    data = await request.json()
    
    # Extraer información del webhook
    event = data.get('event')
    message_type = data.get('message_type')
    conversation = data.get('conversation', {})
    labels = conversation.get('labels', [])
    message_content = data.get('content')
    conversation_id = conversation.get('id')
    sender = data.get('sender', {})
    sender_type = sender.get('type', '')
    contact_id = sender.get('id')  # ID del contacto para el tag "ia-off"

    # Debug
    print(f"\n{'='*60}")
    print(f"📩 Webhook recibido: {event}")
    print(f"   Conversación: {conversation_id} | Contacto: {contact_id}")
    print(f"   Tipo: {message_type}")
    print(f"   Etiquetas: {labels}")

    # Solo procesar mensajes entrantes (del usuario, no del bot)
    if event != 'message_created':
        return {"status": "ignored", "reason": "Not a message_created event"}

    if message_type != 'incoming':
        return {"status": "ignored", "reason": "Not an incoming message"}

    # No responder si el contacto/conversación tiene el tag "ia-off"
    if TAG_IA_OFF in labels:
        print(f"   ⏭️  Ignorado: tiene tag '{TAG_IA_OFF}' (IA desactivada)")
        return {"status": "ignored", "reason": f"Contact has tag '{TAG_IA_OFF}'"}

    if not message_content or not conversation_id:
        return {"status": "ignored", "reason": "Missing content or conversation_id"}

    print(f"   📝 Mensaje: {message_content[:100]}...")

    # Procesar con TramiBot
    try:
        # ── Modo BUFFER: acumular mensajes seguidos y responder UNA sola vez ──
        # Si el usuario manda "hola", "todo bien?", "tengo una consulta" seguidos,
        # se concatenan en Redis y el agente responde una sola vez tras la ventana.
        if BUFFER_ENABLED:
            async def _procesar_concatenacion(conv_id: int, mensaje_concatenado: str) -> None:
                print(f"   🤖 Procesando concatenación (conv={conv_id})...")
                await asyncio.to_thread(
                    ejecutar_agente_y_responder, conv_id, contact_id, mensaje_concatenado
                )

            await encolar_mensaje(conversation_id, message_content, _procesar_concatenacion)
            print(f"   🪣 Mensaje en buffer (ventana de {BUFFER_WINDOW_SECONDS:g}s)")
            return {"status": "buffered", "conversation_id": conversation_id}

        # ── Modo directo: responder mensaje por mensaje (buffer desactivado) ──
        print(f"   🤖 Procesando con TramiBot...")
        await asyncio.to_thread(
            ejecutar_agente_y_responder, conversation_id, contact_id, message_content
        )
        return {"status": "success", "action": "agent_response"}

    except Exception as e:
        print(f"   ❌ Error al procesar: {e}")

        error_message = "Disculpa, tuve un problema al procesar tu consulta. Un asesor te atenderá pronto."
        send_chatwoot_message(conversation_id, error_message)

        return {"status": "error", "message": str(e)}


@app.get("/")
def read_root():
    """Endpoint raíz con información del servicio."""
    return {
        "service": "TramiBot - Agente de Trámites (Municipio de Girardota)",
        "version": "1.0.0",
        "agent": "TramiBot (RAG + Memoria)",
        "model": "GPT-4.1",
        "tools": ["buscar_tramites", "obtener_fecha_hora", "transferir_a_humano"],
        "chatwoot_configured": all([CHATWOOT_BASE_URL, CHATWOOT_ACCOUNT_ID, CHATWOOT_API_TOKEN]),
        "bot_label": BOT_LABEL,
        "status": "ready"
    }


@app.get("/health")
def health_check():
    """Endpoint de salud del servicio."""
    return {
        "status": "healthy",
        "agent": "TramiBot",
        "chatwoot": "connected" if all([CHATWOOT_BASE_URL, CHATWOOT_ACCOUNT_ID, CHATWOOT_API_TOKEN]) else "not configured"
    }


# ============================================
# ENDPOINT DE PRUEBAS: /chat
# ============================================
# Para atacar la API con Red Turing o DeepTeam (objetivo tipo http) hace falta un
# endpoint que reciba un mensaje y devuelva la respuesta del agente en el mismo
# JSON. /webhook no sirve: espera eventos de Chatwoot y responde por su API.
#
#   POST /chat  {"mensaje": "...", "session_id": "uuid opcional"}
#   → {"respuesta": "...", "session_id": "uuid"}
#
# Pasa por el mismo chat_con_agente que el webhook (guardrail de 8 capas + LLM).
# No incluye la tool transferir_a_humano porque no hay conversación de Chatwoot.
# Cada llamada sin session_id abre una conversación nueva, que es lo que un
# arnés de pruebas necesita para que un ataque no vea el historial del anterior.
@app.post("/chat")
async def chat(request: Request):
    datos = await request.json()
    mensaje = str(datos.get("mensaje") or datos.get("message") or "").strip()
    if not mensaje:
        return {"error": "falta 'mensaje'"}
    session_id = str(datos.get("session_id") or "").strip()
    try:
        uuid.UUID(session_id)
    except ValueError:
        session_id = str(uuid.uuid4())
    respuesta = await asyncio.to_thread(chat_con_agente, mensaje, session_id)
    return {"respuesta": respuesta, "session_id": session_id}


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("🚀 INICIANDO TRAMIBOT CON CHATWOOT")
    print("=" * 60)
    print(f"🤖 Agente: TramiBot - Trámites del Municipio de Girardota (RAG + Memoria)")
    print(f"🧠 Modelo: GPT-4.1")
    print(f"🔧 Tools: buscar_tramites, obtener_fecha_hora, transferir_a_humano")
    print(f"💾 Historial: PostgreSQL")
    print(f"🏷️  Etiqueta bot (handoff): {BOT_LABEL or 'ninguna'}")
    print(f"🚫 No responde si tiene tag: {TAG_IA_OFF}")
    print("=" * 60)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

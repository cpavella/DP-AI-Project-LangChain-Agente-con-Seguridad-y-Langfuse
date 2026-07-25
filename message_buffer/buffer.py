"""
Buffer de mensajes con debounce — concatena mensajes seguidos en una respuesta única.

Problema que resuelve:
  El usuario escribe "hola", "todo bien?", "tengo una consulta" como tres
  mensajes seguidos y el bot responde TRES veces. Con este buffer, los
  mensajes se acumulan en Redis durante una ventana corta; si no llega
  ninguno nuevo, se concatenan y entran al agente como UN solo mensaje
  para producir UNA sola respuesta.

Mecanismo (debounce por secuencia):
  - Cada mensaje hace RPUSH a la lista de la conversación e INCR a un
    contador de secuencia. El mensaje captura su número de secuencia.
  - Se programa una tarea que espera WINDOW_SECONDS y, al despertar,
    solo procesa si su secuencia sigue siendo la última (es decir, no
    llegó un mensaje más nuevo). Así "gana" el último mensaje y los
    anteriores no disparan respuestas duplicadas.
  - Una vez el agente respondió y envió el mensaje, la concatenación se
    ELIMINA de Redis: solo sirvió para esa única inferencia. La siguiente
    inferencia concatena únicamente los mensajes nuevos.

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

import asyncio
import logging
from typing import Awaitable, Callable

from message_buffer.config import WINDOW_SECONDS, TTL_SECONDS, SEPARATOR
from message_buffer.redis_client import get_redis

logger = logging.getLogger(__name__)

# Claves de Redis por conversación
_MSGS_KEY = "tramibot:buffer:msgs:{conv}"   # lista con los mensajes acumulados
_SEQ_KEY = "tramibot:buffer:seq:{conv}"     # contador de secuencia (debounce)

# Referencias fuertes a las tareas en vuelo (evita que el GC las cancele).
_tareas: set = set()

# Firma del callback que procesa la concatenación: (conversation_id, mensaje) -> awaitable
ProcesarCallback = Callable[[int, str], Awaitable[None]]


async def encolar_mensaje(
    conversation_id: int,
    mensaje: str,
    procesar: ProcesarCallback,
) -> None:
    """
    Acumula un mensaje en el buffer de la conversación y programa el flush.

    Args:
        conversation_id: ID de la conversación de Chatwoot.
        mensaje:          Texto del mensaje entrante.
        procesar:         Callback async que recibe (conversation_id, mensaje_concatenado)
                          y se encarga de invocar al agente y responder.
    """
    r = get_redis()
    msgs_key = _MSGS_KEY.format(conv=conversation_id)
    seq_key = _SEQ_KEY.format(conv=conversation_id)

    # Añadir el mensaje y obtener su número de secuencia, todo en un pipeline.
    pipe = r.pipeline()
    pipe.rpush(msgs_key, mensaje)
    pipe.incr(seq_key)
    pipe.expire(msgs_key, TTL_SECONDS)
    pipe.expire(seq_key, TTL_SECONDS)
    resultados = await pipe.execute()
    mi_seq = int(resultados[1])

    logger.info(
        f"[BUFFER] Mensaje encolado (conv={conversation_id}, seq={mi_seq}). "
        f"Esperando {WINDOW_SECONDS}s de silencio..."
    )

    # Programar el flush en segundo plano (no bloquea la respuesta al webhook).
    tarea = asyncio.create_task(_esperar_y_procesar(conversation_id, mi_seq, procesar))
    _tareas.add(tarea)
    tarea.add_done_callback(_tareas.discard)


async def _esperar_y_procesar(
    conversation_id: int,
    mi_seq: int,
    procesar: ProcesarCallback,
) -> None:
    """Espera la ventana y, si este fue el último mensaje, concatena y procesa."""
    r = get_redis()
    msgs_key = _MSGS_KEY.format(conv=conversation_id)
    seq_key = _SEQ_KEY.format(conv=conversation_id)

    try:
        await asyncio.sleep(WINDOW_SECONDS)

        # ¿Sigue siendo este el último mensaje? Si llegó uno más nuevo, abortar:
        # esa otra tarea (con secuencia mayor) se encargará de procesar todo.
        seq_actual = await r.get(seq_key)
        if seq_actual is None or int(seq_actual) != mi_seq:
            logger.info(
                f"[BUFFER] Llegó un mensaje más nuevo (conv={conversation_id}); "
                f"este turno (seq={mi_seq}) se omite."
            )
            return

        # Soy el último: leo la concatenación (sin borrar todavía).
        mensajes = await r.lrange(msgs_key, 0, -1)
        if not mensajes:
            return

        combinado = SEPARATOR.join(mensajes)
        n_consumidos = len(mensajes)
        logger.info(
            f"[BUFFER] Concatenando {n_consumidos} mensaje(s) → 1 respuesta "
            f"(conv={conversation_id})."
        )

        # Ejecutar agente + enviar respuesta; la limpieza ocurre SIEMPRE después,
        # haya respondido bien o haya fallado (la concatenación ya cumplió su única
        # inferencia y no debe contaminar la siguiente).
        try:
            await procesar(conversation_id, combinado)
        finally:
            await _limpiar_concatenacion(
                r, msgs_key, seq_key, mi_seq, n_consumidos, conversation_id
            )

    except Exception as e:
        logger.error(f"[BUFFER] Error procesando buffer (conv={conversation_id}): {e}")


async def _limpiar_concatenacion(
    r,
    msgs_key: str,
    seq_key: str,
    mi_seq: int,
    n_consumidos: int,
    conversation_id: int,
) -> None:
    """
    Elimina de Redis la concatenación ya respondida.

    Una vez el agente envió su respuesta, esos mensajes solo sirvieron para UNA
    inferencia y deben borrarse. Se contempla la carrera de que lleguen mensajes
    nuevos mientras el agente respondía:

      - Si NO llegó nada nuevo (la secuencia sigue siendo la mía) → se borra todo.
      - Si SÍ llegaron mensajes nuevos → se eliminan solo los ya consumidos
        (LTRIM) y los nuevos quedan para la siguiente inferencia.
    """
    seq_final = await r.get(seq_key)

    if seq_final is not None and int(seq_final) == mi_seq:
        await r.delete(msgs_key, seq_key)
        logger.info(
            f"[BUFFER] 🧹 Concatenación eliminada tras responder (conv={conversation_id})."
        )
    else:
        # Conservar los mensajes que entraron durante la respuesta del agente.
        await r.ltrim(msgs_key, n_consumidos, -1)
        logger.info(
            f"[BUFFER] 🧹 Eliminados {n_consumidos} mensaje(s) ya respondidos; "
            f"los nuevos se conservan para la próxima inferencia (conv={conversation_id})."
        )

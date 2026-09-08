"""
LLM-as-a-Judge: evaluación automática de respuestas de TramiBot.

Usa un LLM barato (GPT-4o-mini) para evaluar cada respuesta del agente de trámites
del Municipio de Girardota en siete dimensiones y envía los scores a Langfuse.

Scores registrados (escala 0.0-1.0):
  - relevancia-tramites : ¿La respuesta habló de trámites/servicios del municipio?
  - calidad-respuesta   : ¿Fue útil, clara y correcta?
  - alucinacion         : ¿Inventó requisitos, documentos, tiempos, costos o dependencias?
  - completitud-tramite : ¿Entregó la información clave del trámite solicitado?
  - rechazo-correcto    : ¿Rechazó bien preguntas fuera del ámbito municipal?

Scores registrados (escala Likert 1-5):
  - sentimiento-usuario  : Ánimo del ciudadano en su mensaje (1=triste, 5=muy contento)
  - expresividad-agente  : Qué tan expresivo fue el bot frente a la postura del ciudadano
                           (1=plano/robótico, 5=muy expresivo y bien sintonizado)

Diseño intencionado:
- Módulo independiente: no importa nada del agente, solo Langfuse y LangChain.
- Cliente Langfuse obtenido con get_client() (singleton, ya inicializado en el agente).
- Silencioso: cualquier fallo del juez es capturado con try/except para no
  interrumpir la conversación del usuario.

Uso desde el agente:
    from evaluation.llm_judge import evaluar_con_llm_judge
    ...
    trace_id = langfuse_client.get_current_trace_id()
    if trace_id:
        evaluar_con_llm_judge(mensaje_usuario, respuesta_final, trace_id)
"""

import json

from langchain.chat_models import init_chat_model
from langfuse import get_client

# ============================================
# MODELO JUEZ
# ============================================
# Usamos GPT-4o-mini para mantener el costo de evaluación bajo.
# Temperatura 0 para respuestas deterministas y JSON consistente.
_chat_judge = init_chat_model("gpt-4.1", temperature=0)

# ============================================
# PROMPT DEL JUEZ
# ============================================
# Pide un JSON con exactamente 8 campos. Las llaves dobles {{ }} son
# literales en str.format() (escapan las llaves del JSON de la respuesta).
_JUDGE_PROMPT = """Eres un evaluador experto de chatbots de atención ciudadana.

Evalúa la siguiente interacción de TramiBot, el asistente virtual de trámites del Municipio de Girardota (Colombia).

MENSAJE DEL CIUDADANO:
{mensaje}

RESPUESTA DEL BOT:
{respuesta}

Evalúa en siete dimensiones y responde ÚNICAMENTE con JSON válido (sin markdown, sin explicaciones extra):
{{
  "relevancia": <float 0.0-1.0>,
  "calidad": <float 0.0-1.0>,
  "alucinacion": <float 0.0-1.0>,
  "completitud": <float 0.0-1.0>,
  "rechazo_correcto": <float 0.0-1.0>,
  "sentimiento_usuario": <entero 1-5>,
  "expresividad_agente": <entero 1-5>,
  "razon": "<máximo 100 caracteres>"
}}

Definiciones:
- relevancia:       ¿La respuesta está enfocada en trámites o servicios del Municipio de Girardota? 0=nada relevante, 1=totalmente relevante
- calidad:          ¿La respuesta es útil, clara y correcta? 0=pésima, 1=excelente
- alucinacion:      ¿El bot inventó datos específicos que no puede conocer (requisitos, documentos, tiempos de obtención, costos, nombres de secretarías/dependencias)? 0=no inventó nada, 1=inventó datos concretos. Si la pregunta no aplica, pon 0.
- completitud:      Cuando el ciudadano pregunta por un trámite, ¿la respuesta entregó la información clave solicitada (propósito, requisitos/documentos, tiempo de obtención y/o dependencia responsable)? 0=muy incompleta, 1=completa y accionable. Si la pregunta no aplica (ej. un saludo), pon 0.5.
- rechazo_correcto: Si el ciudadano preguntó algo fuera del ámbito del municipio (otro municipio, cultura general, etc.), ¿el bot lo rechazó correctamente con amabilidad? 0=respondió sin rechazar (MAL), 1=rechazó correctamente (BIEN). Si la pregunta SÍ era sobre el municipio, pon 1.
- sentimiento_usuario: Estado de ánimo que transmite el MENSAJE DEL CIUDADANO, en escala Likert:
                       1=muy triste, molesto o frustrado (queja, reclamo, desesperación)
                       2=triste, incómodo o impaciente
                       3=neutral (consulta informativa sin carga emocional)
                       4=contento, amable o agradecido
                       5=muy contento, entusiasta o efusivo
                       Juzga solo el mensaje del ciudadano, no la respuesta del bot. Si no hay señales emocionales, pon 3.
- expresividad_agente: ¿Qué tan expresivo fue el BOT frente a la postura emocional del ciudadano (la que mediste en sentimiento_usuario)? Escala Likert:
                       1=plano y robótico, ignora por completo el estado emocional del ciudadano
                       2=apenas cortés, formulismo sin reconocer la emoción
                       3=reconoce el tono de forma genérica ("entiendo", "con gusto")
                       4=expresivo y bien sintonizado: nombra la emoción y ajusta el tono (empatiza si está molesto, acompaña si está contento)
                       5=muy expresivo y perfectamente calibrado con la emoción del ciudadano, sin sonar exagerado ni falso
                       Penaliza el desajuste: entusiasmo festivo ante un ciudadano molesto, o frialdad ante un ciudadano angustiado, no pasa de 2.
                       Si sentimiento_usuario=3 (neutral), una respuesta informativa, cortés y que resuelve la consulta es un 3; reserva 1-2 para respuestas cortantes o descuidadas, y 4-5 para las que además cierran con calidez o se ofrecen a acompañar.
- razon:            Razón breve que justifica los puntajes más bajos o llamativos"""


# ============================================
# HELPERS
# ============================================
def _a_likert(valor, defecto: int = 3) -> int:
    """Convierte un valor del juez a un entero Likert válido (1-5)."""
    try:
        return max(1, min(5, int(round(float(valor)))))
    except (TypeError, ValueError):
        return defecto


# ============================================
# FUNCIÓN PÚBLICA
# ============================================
def evaluar_con_llm_judge(
    mensaje_usuario: str,
    respuesta_final: str,
    trace_id: str,
) -> None:
    """
    Evalúa la respuesta de TramiBot con un LLM juez y envía los scores a Langfuse.

    Scores en escala 0-1 (float):
      - "relevancia-tramites" : ¿La respuesta habló de trámites/servicios del municipio?
      - "calidad-respuesta"   : ¿Fue útil, clara y correcta?
      - "alucinacion"         : ¿Inventó requisitos, documentos, tiempos u otros datos?
      - "completitud-tramite" : ¿Entregó la información clave del trámite solicitado?
      - "rechazo-correcto"    : ¿Rechazó bien preguntas fuera del ámbito municipal?

    Scores en escala Likert 1-5 (int):
      - "sentimiento-usuario" : Ánimo del ciudadano (1=triste, 3=neutral, 5=muy contento)
      - "expresividad-agente" : Expresividad del bot frente a la postura del ciudadano
                                (1=plano/robótico, 5=muy expresivo y bien calibrado)

    Todos los scores quedan visibles en:
      Langfuse UI → Tracing → (click en el trace) → sección Scores
      Langfuse UI → Evaluation → Scores → Analytics

    Args:
        mensaje_usuario: El mensaje que envió el ciudadano en este turno.
        respuesta_final: La respuesta que generó el agente.
        trace_id:        El trace_id de Langfuse del turno actual.
    """
    try:
        # Construir el prompt con los datos del turno actual
        prompt_eval = _JUDGE_PROMPT.format(
            mensaje=mensaje_usuario,
            respuesta=respuesta_final,
        )

        # Llamar al LLM juez SIN callbacks de Langfuse: esta llamada interna
        # no debe aparecer en el trace del usuario para no generar ruido en el dashboard
        eval_response = _chat_judge.invoke([{"role": "user", "content": prompt_eval}])

        # Parsear el JSON devuelto por el juez y sanear todos los valores entre 0 y 1
        eval_data        = json.loads(eval_response.content.strip())
        relevancia       = max(0.0, min(1.0, float(eval_data.get("relevancia",       0.0))))
        calidad          = max(0.0, min(1.0, float(eval_data.get("calidad",          0.0))))
        alucinacion      = max(0.0, min(1.0, float(eval_data.get("alucinacion",      0.0))))
        completitud      = max(0.0, min(1.0, float(eval_data.get("completitud",      0.0))))
        rechazo_correcto = max(0.0, min(1.0, float(eval_data.get("rechazo_correcto", 1.0))))
        razon            = str(eval_data.get("razon", ""))[:100]

        # Escalas Likert 1-5: se sanean aparte porque no comparten el rango 0-1.
        # Por defecto 3 = neutral / expresividad genérica, para no premiar ni castigar
        # cuando el juez no devuelve el campo.
        sentimiento_usuario = _a_likert(eval_data.get("sentimiento_usuario"), defecto=3)
        expresividad_agente = _a_likert(eval_data.get("expresividad_agente"), defecto=3)

        # Obtener el cliente Langfuse (singleton ya inicializado en el agente)
        lf = get_client()

        # Langfuse v4: método create_score() — enviar los 7 scores al trace actual
        lf.create_score(
            trace_id=trace_id,
            name="relevancia-tramites",
            value=relevancia,
            data_type="NUMERIC",
            comment=razon,
        )
        lf.create_score(
            trace_id=trace_id,
            name="calidad-respuesta",
            value=calidad,
            data_type="NUMERIC",
            comment=razon,
        )
        # alucinacion: 0=no inventó nada (bueno), 1=inventó datos (malo)
        lf.create_score(
            trace_id=trace_id,
            name="alucinacion",
            value=alucinacion,
            data_type="NUMERIC",
            comment=razon,
        )
        # completitud-tramite: 0=respuesta incompleta, 1=entregó la info clave del trámite
        lf.create_score(
            trace_id=trace_id,
            name="completitud-tramite",
            value=completitud,
            data_type="NUMERIC",
            comment=razon,
        )
        # rechazo-correcto: 1=rechazó bien (o era pregunta válida), 0=respondió sin rechazar
        lf.create_score(
            trace_id=trace_id,
            name="rechazo-correcto",
            value=rechazo_correcto,
            data_type="NUMERIC",
            comment=razon,
        )
        # sentimiento-usuario: Likert 1-5 sobre el mensaje del ciudadano
        # (1=triste/molesto, 3=neutral, 5=muy contento). No mide al bot.
        lf.create_score(
            trace_id=trace_id,
            name="sentimiento-usuario",
            value=sentimiento_usuario,
            data_type="NUMERIC",
            comment=razon,
        )
        # expresividad-agente: Likert 1-5 sobre cuánta expresividad mostró el bot
        # frente a la postura emocional del ciudadano (1=plano, 5=muy expresivo)
        lf.create_score(
            trace_id=trace_id,
            name="expresividad-agente",
            value=expresividad_agente,
            data_type="NUMERIC",
            comment=razon,
        )

        print(
            f"   📊 [JUDGE] "
            f"relevancia={relevancia:.2f} | "
            f"calidad={calidad:.2f} | "
            f"alucinacion={alucinacion:.2f} | "
            f"completitud={completitud:.2f} | "
            f"rechazo={rechazo_correcto:.2f} | "
            f"sentimiento={sentimiento_usuario}/5 | "
            f"expresividad={expresividad_agente}/5 | "
            f"{razon[:50]}"
        )

    except Exception as e:
        # Si el juez falla (timeout, JSON malformado, API error, etc.)
        # el agente NO se cae; simplemente se omite la evaluación de este turno
        print(f"   ⚠️ [JUDGE] Evaluación omitida: {e}")

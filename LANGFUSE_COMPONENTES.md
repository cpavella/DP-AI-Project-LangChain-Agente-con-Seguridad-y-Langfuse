# Componentes de Observabilidad y Evaluación con Langfuse — Proyecto DataBot

**Autor:** Ing. Kevin Inofuente Colque — DataPath  
**Proyecto:** Agente IA Multi-Tool con Seguridad para WhatsApp / Chatwoot  
**Módulo:** 1 — Sesión 1 — Programa AI Engineer

---

## ¿Qué es Langfuse?

**Langfuse** es una plataforma open-source de **observabilidad, monitoreo y evaluación para aplicaciones de Inteligencia Artificial**. Es el equivalente a un panel de control completo que registra todo lo que hace el agente IA: cada mensaje que recibe, cada herramienta que usa, cuántos tokens consume, cuánto tarda en responder y qué tan buenas son sus respuestas.

En términos simples: así como un negocio tiene un sistema de cámaras y registros de atención al cliente, **Langfuse es el sistema de cámaras de DataBot**.

### ¿Por qué es necesario en producción?

Un agente de IA en producción (atendiendo clientes reales por WhatsApp) necesita ser monitoreado constantemente porque:

- **Los LLMs no son perfectos:** pueden alucinar datos, responder fuera de tema o dar información incorrecta
- **El costo importa:** cada mensaje consume tokens que tienen un costo en dólares; sin monitoreo no sabes si estás gastando de más
- **La calidad debe medirse:** no es suficiente con que "funcione", hay que saber si las respuestas son útiles para los prospectos de DATAPATH
- **Los errores ocurren en silencio:** sin observabilidad, no sabrías si el agente falló en responder o respondió mal a un prospecto

### Integración en el proyecto

Langfuse se integró en el archivo `agente_sec_langfuse.py` (el cerebro del agente) y en el módulo `evaluation/llm_judge.py` (el sistema de evaluación automática).

---

## Arquitectura de Observabilidad

Cada mensaje de WhatsApp que llega a DataBot genera una **traza (trace)** en Langfuse que registra el ciclo de vida completo del turno:

```
Usuario envía mensaje por WhatsApp
              │
              ▼
   Chatwoot recibe el mensaje
              │
              ▼
   FastAPI (main_chatwoot-ia_off.py)
              │
              ▼
┌─────────────────────────────────────────────────────┐
│          AGENTE DATABOT (GPT-4.1)                   │
│                                                     │
│  @observe()  ◄── Langfuse crea el Trace raíz        │
│       │                                             │
│       ├── Guardrail evaluation (seguridad)          │
│       ├── LLM Call 1 (decide si usar tools)  ◄── Langfuse registra tokens, latencia, costo
│       ├── Tool Execution (buscar_datapath, etc.)    │
│       └── LLM Call 2 (respuesta final)       ◄── Langfuse registra tokens, latencia, costo
│                                                     │
│  LLM-as-a-Judge ◄── Evalúa la respuesta (7 scores) │
└─────────────────────────────────────────────────────┘
              │
              ▼
   Langfuse Dashboard (cloud)
   → Tracing, Sessions, Users, Scores
              │
              ▼
   Respuesta enviada al usuario por WhatsApp
```

> Cada trace agrupa todas las observaciones de un turno: las llamadas al LLM, las herramientas ejecutadas, los tokens consumidos y los scores de evaluación. Todo queda indexado por `session_id` para ver conversaciones completas.

---

## Componente 1 — Observabilidad (Tracing)

**Tecnología:** Langfuse Python SDK v4 + LangChain CallbackHandler  
**Archivo principal:** `agente_sec_langfuse.py`  
**Visibilidad en UI:** Langfuse → Tracing

### ¿Qué es el Tracing?

El **Tracing** es el registro detallado de todo lo que ocurre dentro del agente durante un turno de conversación. Cada llamada al modelo GPT-4.1, cada herramienta ejecutada y cada resultado quedan guardados como observaciones anidadas dentro de un trace padre.

### ¿Cómo se implementó?

Se usaron tres mecanismos de Langfuse v4 en combinación:

**1. Decorador `@observe()`**  
Se colocó sobre la función principal `chat_con_agente()`. Este decorador convierte automáticamente cada ejecución de la función en un **Trace raíz** en Langfuse. Todo lo que ocurre dentro de la función queda anidado bajo ese trace.

```python
@observe()
def chat_con_agente(mensaje_usuario, session_id, tools_extra=None):
    ...
```

**2. `CallbackHandler` de LangChain**  
Se inyectó en cada llamada al LLM mediante el parámetro `config`. Este handler intercepta automáticamente cada invocación del modelo y registra: el prompt enviado, la respuesta recibida, los tokens de entrada y salida, la latencia y el costo estimado.

```python
langfuse_handler = CallbackHandler()
response = chat_turno.invoke(messages, config={"callbacks": [langfuse_handler]})
```

**3. `propagate_attributes()`**  
Se usa para enriquecer cada trace con metadatos de correlación: el `session_id` de la conversación, el `user_id`, las etiquetas de entorno (`produccion`, `chatwoot`, `databot`) y el modelo utilizado.

```python
with propagate_attributes(
    trace_name="databot-turno",
    session_id=session_id,
    user_id=f"conv-{session_id[:8]}",
    tags=["produccion", "chatwoot", "databot"],
    metadata={"modelo": "gpt-4.1"},
):
```

### ¿Qué se puede ver en el dashboard?

| Métrica | Descripción |
|---|---|
| **Tokens de entrada** | Cuántos tokens se enviaron al modelo (system prompt + historial + mensaje) |
| **Tokens de salida** | Cuántos tokens generó el modelo en su respuesta |
| **Costo** | Precio en USD de cada llamada al LLM, calculado automáticamente |
| **Latencia** | Tiempo en milisegundos que tardó el modelo en responder |
| **Herramientas usadas** | Qué tools invocó el agente (buscar_datapath, buscar_internet, etc.) |
| **Historial completo** | El prompt exacto que recibió el modelo y la respuesta que generó |
| **Session ID** | Permite agrupar todos los turnos de una misma conversación de WhatsApp |

### ¿Por qué importa en un contexto de WhatsApp?

DataBot puede recibir decenas de conversaciones simultáneas. Con Tracing, si un prospecto se queja de una respuesta incorrecta, se puede buscar su `session_id` en Langfuse y ver exactamente qué información recibió el modelo, qué herramienta usó y qué respondió, sin depender de los logs del servidor.

---

## Componente 2 — Gestión de Sesiones y Usuarios

**Tecnología:** Langfuse Session Tracking  
**Visibilidad en UI:** Langfuse → Sessions / Users

### ¿Qué es?

Langfuse agrupa automáticamente los traces por `session_id` para que se pueda ver el hilo completo de una conversación de WhatsApp, no turno por turno sino como una secuencia continua.

### ¿Cómo se implementó?

El `session_id` de Langfuse es el mismo identificador de sesión que usa el histórico de PostgreSQL. Esto garantiza que la conversación completa esté sincronizada entre la base de datos y el panel de Langfuse.

- **Sessions:** cada conversación de WhatsApp (identificada por el número de conversación de Chatwoot) tiene su propia sesión en Langfuse
- **Users:** cada contacto de Chatwoot aparece como un usuario en Langfuse, con todos sus turnos agrupados

### ¿Para qué sirve en producción?

Si un prospecto de DATAPATH tuvo una mala experiencia con el bot, se puede buscar su conversación por `session_id` y ver toda la interacción de principio a fin, incluyendo scores de calidad de cada turno, tokens consumidos y las herramientas que el agente utilizó.

---

## Componente 3 — Prompt Management

**Tecnología:** Langfuse Prompt Management  
**Visibilidad en UI:** Langfuse → Prompt Management → Prompts

### ¿Qué es?

El **Prompt Management** es el sistema de Langfuse para almacenar, versionar y servir el `system_prompt` del agente DataBot desde la nube, en lugar de tenerlo hardcodeado en el código Python.

### ¿Cómo se implementó?

Se implementó una alternativa comentada en el agente que permite cambiar entre el prompt en código y el prompt desde Langfuse con solo descomentar 3 líneas:

```python
# OPCIÓN ACTIVA: prompt en código (editable con vibe coding en Cursor)
system_prompt = """Eres DataBot, el asistente virtual oficial de DATAPATH..."""

# OPCIÓN ALTERNATIVA: prompt desde Langfuse UI (comentada, para usar cuando se quiera)
# lf_prompt = langfuse_client.get_prompt("databot-system-prompt")
# system_prompt = lf_prompt.compile()
# print(f"📝 Prompt cargado desde Langfuse: versión {lf_prompt.version}")
```

### ¿Cómo funciona el flujo?

1. Se crea el prompt en **Langfuse UI → Prompts → Create Prompt**
2. Se le asigna el nombre `databot-system-prompt` y el label `production`
3. En tiempo de ejecución, el agente descarga el prompt con `get_prompt()`
4. Cualquier cambio al prompt en la UI se refleja en el agente sin reiniciar el servidor

### Ventajas y desventajas en este proyecto

| | Prompt en código | Prompt en Langfuse |
|---|---|---|
| Edición con IA (vibe coding) | ✅ Sí | ❌ No |
| Git versioning | ✅ Sí | ❌ No (Langfuse tiene el suyo) |
| Cambiar sin reiniciar servidor | ❌ No | ✅ Sí |
| Edición por persona no técnica | ❌ No | ✅ Sí |
| A/B testing de prompts | ❌ No | ✅ Sí |

### Control de versiones de prompts

Cada vez que se edita el prompt en la UI, Langfuse crea una **nueva versión numerada**. Se puede asignar el label `production` a cualquier versión para activarla, y revertir a una versión anterior si la nueva no funciona bien. Cada trace en Langfuse queda vinculado a la versión del prompt que lo generó, permitiendo comparar el rendimiento entre versiones.

---

## Componente 4 — Evaluación Automática (LLM-as-a-Judge)

**Tecnología:** GPT-4o-mini como modelo juez + Langfuse Scores API  
**Archivo principal:** `evaluation/llm_judge.py`  
**Visibilidad en UI:** Langfuse → Evaluation → Scores

### ¿Qué es LLM-as-a-Judge?

**LLM-as-a-Judge** es una técnica de evaluación automática donde se usa un segundo modelo de lenguaje (el "juez") para evaluar la calidad de las respuestas del agente principal. En lugar de revisar manualmente cada conversación, el juez analiza cada turno automáticamente y asigna puntuaciones numéricas.

### ¿Por qué un modelo separado y no el mismo GPT-4.1?

Se usa **GPT-4o-mini** (no GPT-4.1) por tres razones:
1. **Costo:** GPT-4o-mini es significativamente más barato que GPT-4.1
2. **Separación:** el juez debe ser independiente del agente para no tener conflicto de interés
3. **Velocidad:** GPT-4o-mini es más rápido para tareas de clasificación y evaluación

### Arquitectura del módulo

El código de evaluación se separó deliberadamente del agente en su propio módulo `evaluation/` para mantener las responsabilidades bien definidas:

```
evaluation/
├── __init__.py        ← exporta evaluar_con_llm_judge
└── llm_judge.py       ← modelo juez, prompt de evaluación, envío de scores
```

> **Decisión de diseño importante:** el directorio no se llama `langfuse/` porque colisionaría con el paquete Python instalado `langfuse`, causando errores de importación. Se nombró `evaluation/` para ser descriptivo y evitar conflictos.

### ¿Cuándo se ejecuta?

Después de cada respuesta del agente, dentro del mismo turno de conversación:

```
Usuario: "¿Qué cursos tiene DATAPATH?"
                │
                ▼
        Agente genera respuesta
                │
                ▼
        set_current_trace_io()  ← registra input/output en Langfuse
                │
                ▼
        evaluar_con_llm_judge() ← juez evalúa la respuesta
                │
                ▼
        7 scores enviados a Langfuse
                │
                ▼
        Respuesta enviada al usuario por WhatsApp
```

### Los 7 Scores de Evaluación

Cada turno de conversación recibe 7 puntuaciones automáticas: cinco en escala de 0.0 a 1.0 y dos en escala Likert de 1 a 5.

#### Score 1 — `relevancia-datapath`
**¿La respuesta estuvo enfocada en DATAPATH?**

- `0.0` = La respuesta no tuvo nada que ver con DATAPATH
- `1.0` = La respuesta estuvo 100% enfocada en DATAPATH

Este score detecta si el agente se salió de su ámbito (responder sobre cultura general, noticias u otros temas que debería rechazar).

#### Score 2 — `calidad-respuesta`
**¿La respuesta fue útil, clara y correcta?**

- `0.0` = Respuesta pésima, confusa o incompleta
- `1.0` = Respuesta excelente, clara y completa

Este score mide la calidad general de la respuesta independientemente del tema.

#### Score 3 — `alucinacion`
**¿El agente inventó datos concretos que no puede conocer?**

- `0.0` = No inventó nada (esto es lo deseable, bajo es bueno)
- `1.0` = Inventó precios, fechas, nombres de docentes u otros datos concretos

> Este score es crítico para una escuela como DATAPATH: el agente **nunca debe inventar precios de programas, fechas de inicio o nombres de docentes**. Si lo hace, el score `alucinacion` se dispara y se puede detectar el problema.

#### Score 4 — `llamada-a-accion`
**¿La respuesta invitó al usuario a dar un siguiente paso comercial?**

- `0.0` = La respuesta solo informó pero no motivó ninguna acción
- `1.0` = La respuesta invitó claramente a inscribirse, contactar un asesor, pedir el temario, etc.

Este score mide si el agente está cumpliendo su rol comercial: no solo informar, sino convertir prospectos. Si este score es consistentemente bajo, significa que el `system_prompt` debe ajustarse para que el agente sea más proactivo en invitar a la acción.

#### Score 5 — `rechazo-correcto`
**¿El agente rechazó correctamente preguntas fuera del ámbito de DATAPATH?**

- `0.0` = El agente respondió una pregunta fuera de tema sin rechazarla (MAL)
- `1.0` = El agente rechazó correctamente la pregunta fuera de tema, O la pregunta sí era válida (BIEN)

> Los guardrails de seguridad bloquean mensajes peligrosos **antes** de que lleguen al agente. Este score es diferente: mide si el agente rechaza preguntas válidas pero fuera de su ámbito (ej. "¿quién ganó el Mundial?"), que los guardrails dejan pasar porque no son peligrosas.

#### Score 6 — `sentimiento-usuario` (Likert 1-5)
**¿Con qué ánimo escribió el ciudadano?**

- `1` = Muy triste, molesto o frustrado (queja, reclamo, desesperación)
- `2` = Triste, incómodo o impaciente
- `3` = Neutral (consulta informativa sin carga emocional)
- `4` = Contento, amable o agradecido
- `5` = Muy contento, entusiasta o efusivo

Este score mide **al ciudadano, no al bot**: se juzga únicamente el mensaje de entrada. Sirve para detectar si el canal está recibiendo sobre todo consultas neutrales o una ola de reclamos, y es la base de comparación del Score 7.

#### Score 7 — `expresividad-agente` (Likert 1-5)
**¿Qué tan expresivo fue el bot frente a la postura emocional del ciudadano?**

- `1` = Plano y robótico, ignora por completo el estado emocional
- `2` = Apenas cortés, formulismo sin reconocer la emoción
- `3` = Reconoce el tono de forma genérica ("entiendo", "con gusto")
- `4` = Expresivo y bien sintonizado: nombra la emoción y ajusta el tono
- `5` = Muy expresivo y perfectamente calibrado, sin sonar exagerado ni falso

> Este score es **relativo** al Score 6: no premia expresividad en abstracto, sino el ajuste a la postura del humano. El desajuste se penaliza en ambas direcciones — entusiasmo festivo ante un ciudadano molesto, o frialdad ante un ciudadano angustiado, no pasa de `2`. Cuando el ciudadano llega neutral (`sentimiento-usuario = 3`), una respuesta informativa y cortés que resuelve la consulta vale `3`: la rúbrica no exige efusividad donde no hay emoción que acompañar. Leer los dos scores juntos (`sentimiento-usuario` vs `expresividad-agente`) muestra si el tono del `system_prompt` está acompañando bien al ciudadano.

### Funcionamiento silencioso

El juez se ejecuta de forma silenciosa: si falla por cualquier razón (timeout, respuesta mal formateada, error de API), el agente **no se cae** ni se interrumpe la conversación. El fallo del juez solo genera un log de advertencia en la terminal:

```
⚠️ [JUDGE] Evaluación omitida: <motivo del error>
```

### ¿Qué se ve en el dashboard?

En **Langfuse → Tracing → (click en un trace)**: los 7 scores aparecen en la sección **Scores** del panel derecho, vinculados directamente al trace del turno evaluado.

En **Langfuse → Evaluation → Scores → Analytics**: vista agregada de todos los scores en el tiempo, con gráficos de tendencia para monitorear si la calidad del agente sube o baja con el tiempo.

---

## Componente 5 — Evaluación Manual (Human Annotation)

**Tecnología:** Langfuse Human Annotation Queues  
**Visibilidad en UI:** Langfuse → Evaluation → Human Annotation  
**Requiere cambios en código:** No — 100% configurado desde la UI

### ¿Qué es?

La **Human Annotation** es el sistema de Langfuse para que una persona (el administrador del proyecto, un evaluador de calidad o alguien del equipo de DATAPATH) revise manualmente conversaciones seleccionadas y les asigne puntuaciones subjetivas que un modelo de IA no puede evaluar con precisión.

### ¿Por qué es necesaria si ya tenemos LLM-as-a-Judge?

El LLM-as-a-Judge evalúa dimensiones objetivas y cuantificables. Pero algunas dimensiones de calidad son profundamente **subjetivas** y requieren criterio humano:

- **Tono empático:** ¿El agente sonó frío y robótico, o cálido y cercano? Solo una persona leyendo el contexto completo puede evaluar esto con precisión
- **Oportunidad perdida:** ¿El agente debió ofrecer algo más y no lo hizo? Requiere criterio comercial del negocio
- **Experiencia general:** Una calificación holística 1-5 de toda la conversación

### Score implementado para prueba

Se creó el score `tono-empatico` como primer score de Human Annotation:
- **Tipo:** NUMERIC
- **Rango:** 1 a 5
- **Significado:** 1 = frío y robótico / 5 = cálido, cercano y motivador

### Flujo de uso

1. En la UI se define una **Queue** (cola de revisión) con los traces a evaluar
2. Los traces se agregan a la queue manualmente o con filtros automáticos (ej. todos los de un día)
3. El evaluador abre la queue y ve los traces uno por uno
4. Para cada trace lee la conversación y asigna el score numérico
5. El score queda guardado en Langfuse junto a los scores automáticos del juez

### Combinación de ambos enfoques

El verdadero valor de tener ambos sistemas (automático y manual) es la **correlación**: si el LLM-as-a-Judge asigna `calidad-respuesta = 0.90` pero el evaluador humano asigna `tono-empatico = 2/5`, hay una señal clara de que el agente da información correcta pero con un tono inadecuado para vender un programa de formación en IA.

| Trace | `calidad-respuesta` (juez) | `tono-empatico` (humano) | Conclusión |
|---|---|---|---|
| Trace A | 0.95 | 5/5 | Excelente en todo |
| Trace B | 0.90 | 2/5 | Correcto pero frío — ajustar prompt |
| Trace C | 0.40 | 4/5 | Empático pero con info incorrecta |
| Trace D | 0.85 | 4/5 | Buen equilibrio |

---

## Resumen Comparativo de los 5 Componentes

| Componente | Tecnología | Requiere código | Visibilidad en UI |
|---|---|---|---|
| Observabilidad (Tracing) | `@observe()` + `CallbackHandler` | Sí | Tracing, Sessions, Users |
| Sesiones y Usuarios | Session Tracking automático | Mínimo (session_id) | Sessions, Users |
| Prompt Management | `get_prompt()` del SDK | Opcional (comentado) | Prompt Management |
| LLM-as-a-Judge (7 scores) | GPT-4o-mini + `create_score()` | Sí (`evaluation/llm_judge.py`) | Evaluation → Scores |
| Human Annotation | Queues de revisión manual | No | Evaluation → Human Annotation |

---

## Métricas que se pueden monitorear en producción

Con toda la integración funcionando, el dashboard de Langfuse permite monitorear DataBot en producción con las siguientes métricas:

### Métricas de Costo y Uso
- **Tokens por conversación:** cuánto cuesta en promedio atender a un prospecto
- **Costo diario/semanal:** seguimiento del gasto en API de OpenAI
- **Conversaciones con uso de tools:** qué porcentaje de mensajes requirieron buscar información

### Métricas de Calidad (LLM-as-a-Judge)
- **`relevancia-datapath` promedio:** ¿el agente se mantiene enfocado en DATAPATH?
- **`calidad-respuesta` promedio:** ¿la calidad general sube o baja con el tiempo?
- **`alucinacion` promedio:** ¿el agente está inventando datos? (debe mantenerse cercano a 0)
- **`llamada-a-accion` promedio:** ¿el agente está convirtiendo prospectos?
- **`rechazo-correcto` promedio:** ¿el agente rechaza bien lo que está fuera de su ámbito?

### Métricas de Latencia
- **Tiempo de respuesta promedio:** cuánto espera el usuario por una respuesta en WhatsApp
- **Turnos con tools vs. sin tools:** los turnos con herramientas son más lentos (dos llamadas al LLM)

---

## Principio de Diseño: Observabilidad como Capa Transversal

A diferencia del sistema de seguridad (que es un pipeline lineal de 8 capas), Langfuse funciona como una **capa transversal** que atraviesa todo el sistema sin modificar su comportamiento.

- El agente funciona exactamente igual con o sin Langfuse
- Si Langfuse no está disponible, el agente sigue respondiendo (nunca bloquea)
- Los scores del juez se envían de forma silenciosa y asíncrona
- El `flush()` al final garantiza que no se pierda ningún evento al cerrar

Este diseño sigue el principio **"observability as a side effect"**: la observabilidad se agrega al sistema sin que el sistema tenga que cambiar su lógica principal.

---

*Documento generado para el Programa AI Engineer — DataPath*  
*Ing. Kevin Inofuente Colque — Módulo 1, Sesión 1*

# DataBot — Agente IA Multi-Tool con Seguridad y Observabilidad

Asistente virtual inteligente para **DATAPATH**, construido con LangChain y GPT-4.1. Responde consultas sobre programas, cursos, precios e inscripciones usando RAG (Pinecone), búsqueda en internet (Tavily) y memoria persistente (PostgreSQL). Se integra con **Chatwoot** vía webhook para atención automatizada por WhatsApp con handoff a humanos. Incluye un sistema de seguridad de 8 capas y observabilidad completa con **Langfuse**.

---

## Arquitectura

```
Usuario (WhatsApp)
        │
        ▼
   Chatwoot  ──webhook──▶  FastAPI (/webhook)
                                  │
                          InputGuardrail (8 capas)
                                  │ (si pasa)
                               Agente IA (GPT-4.1)
                           ┌──────┼──────┬──────────────┐
                    buscar_      buscar_  obtener_   transferir_
                    datapath    internet  fecha_hora  a_humano
                    (Pinecone)  (Tavily)  (stdlib)   (Chatwoot API)
                                  │
                           PostgreSQL (historial)
                                  │
                          LLM-as-a-Judge (GPT-4o-mini)
                                  │
                           Langfuse Dashboard
                        (trazas, tokens, scores, sesiones)
```

---

## Herramientas del Agente

| Tool | Descripción |
|---|---|
| `buscar_datapath` | RAG sobre la base de conocimiento de DATAPATH (Pinecone + `text-embedding-ada-002`) |
| `buscar_internet` | Búsqueda web contextualizada a DATAPATH via Tavily |
| `obtener_fecha_hora` | Fecha y hora actual por zona horaria IANA (sin APIs externas) |
| `transferir_a_humano` | Agrega el tag `ia-off` al contacto y conversación en Chatwoot para desactivar el bot |

---

## Capas de Seguridad (InputGuardrail)

El pipeline de seguridad valida **cada mensaje entrante** antes de llegar al agente:

| Capa | Tecnología | Qué detecta |
|---|---|---|
| 1 | REGEX | API keys y tokens secretos (`sk-...`, `ghp_...`, `AIza...`) |
| 2 | REGEX | Prompt injection, jailbreak, reconocimiento de infraestructura (EN + ES) |
| 3 | REGEX | Amenazas, hate speech, acoso sexual, incitación a la violencia, autolesión |
| 4 | REGEX (ReDoS-safe) | Patrones personalizados configurables por el admin |
| 5 | Presidio (Microsoft) | PII: DNI, RUC, email, teléfono peruano |
| 6 | REGEX | URLs, acortadores y dominios en blacklist |
| 7 | Llama Prompt Guard 2 (Groq) | Jailbreak/injection por IA (86M params, 8 idiomas) |
| 8 | GPT-OSS-Safeguard (Groq) | NSFW, violencia, odio, autolesión — 13 categorías (20B params) |

---

## Observabilidad con Langfuse

El proyecto integra **Langfuse** como capa transversal de observabilidad, monitoreo y evaluación. Cada turno de conversación genera un trace completo en el dashboard.

### Tracing automático

Cada mensaje procesado registra automáticamente en Langfuse:
- Tokens de entrada y salida de cada llamada al LLM
- Costo estimado en USD por turno
- Latencia de respuesta del modelo
- Herramientas invocadas y sus resultados
- Session ID para agrupar toda la conversación

### LLM-as-a-Judge (evaluación automática)

Después de cada respuesta, **GPT-4o-mini** evalúa la calidad del agente con 5 scores en escala 0.0–1.0:

| Score | Qué mide |
|---|---|
| `relevancia-datapath` | ¿La respuesta estuvo enfocada en DATAPATH? |
| `calidad-respuesta` | ¿Fue útil, clara y correcta? |
| `alucinacion` | ¿Inventó precios, fechas u otros datos concretos? *(bajo = bueno)* |
| `llamada-a-accion` | ¿Invitó al usuario a inscribirse o contactar a DATAPATH? |
| `rechazo-correcto` | ¿Rechazó bien preguntas fuera del ámbito de DATAPATH? |

### Prompt Management

El `system_prompt` puede gestionarse directamente desde la UI de Langfuse (versiones, labels `production`/`staging`) sin tocar el código. La opción está implementada y comentada en el agente para activarse cuando se necesite.

### Human Annotation

Queues de revisión manual para que el equipo de DATAPATH evalúe conversaciones seleccionadas con scores subjetivos como `tono-empatico` (1–5) que un LLM no puede evaluar con precisión.

---

## Integración con Chatwoot

- **Etiqueta `atiende-ia`**: activa el bot en una conversación.
- **Etiqueta `ia-off`**: desactiva la IA — el bot no responde.
- **Handoff automático**: si el usuario pide hablar con un humano, el bot usa la tool `transferir_a_humano` que agrega `ia-off` tanto al contacto como a la conversación y envía un mensaje de despedida.
- El historial de conversación se asocia al `conversation_id` de Chatwoot mediante un UUID determinista.

---

## Estructura del Proyecto

```
.
├── main_chatwoot-ia_off.py                          # Webhook FastAPI + lógica Chatwoot
├── agente_sec_langfuse.py                           # Agente con Langfuse
├── agente_sec.py                                    # Agente sin Langfuse (referencia)
├── tools/
│   ├── __init__.py
│   ├── Base_de_conocimiento.py                      # Tool RAG con Pinecone
│   ├── Busqueda_internet.py                         # Tool Tavily
│   ├── Hora_y_fecha.py                              # Tool fecha/hora
│   └── Transferir_humano.py                         # Tool handoff a humano (Chatwoot API)
├── guardrails/
│   ├── input_guardrail.py                           # Orquestador de seguridad (8 capas)
│   ├── pii_detector.py                              # Capa 5 — Presidio
│   └── llama_guard_service.py                       # Capas 7 y 8 — LlamaGuard via Groq
├── evaluation/
│   ├── __init__.py
│   └── llm_judge.py                                 # LLM-as-a-Judge (7 scores → Langfuse)
├── requirements.txt
└── .env                                             # Variables de entorno (no subir a git)
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd LangChain-AgenteIA-MultiTool-Seguridad

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Modelo de spaCy para PII (Capa 5)
python -m spacy download es_core_news_sm
```

---

## Configuración (.env)

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Pinecone (Base de Conocimiento)
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=langchain-pinecone-asistente-de-ventas

# Tavily (Búsqueda internet)
TAVILY_API_KEY=tvly-...

# PostgreSQL (Historial de conversación)
DB_USER=postgres
DB_PASSWORD=...
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres

# Chatwoot
CHATWOOT_BASE_URL=https://app.chatwoot.com
CHATWOOT_ACCOUNT_ID=1
CHATWOOT_API_ACCESS_TOKEN=...
CHATWOOT_BOT_LABEL=atiende-ia

# Groq (Capas 7 y 8 — LlamaGuard)
GROQ_API_KEY=gsk_...

# Langfuse (Observabilidad)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

# Zona horaria del agente
AGENT_TIMEZONE=America/Lima
```

---

## Ejecución

### Modo Chatwoot (webhook)

```bash
python main_chatwoot-ia_off.py
# Servidor en http://0.0.0.0:8000
```

Endpoints disponibles:

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/webhook` | Recibe eventos de Chatwoot |
| `POST` | `/chat` | Prueba el agente sin Chatwoot: `{"mensaje": "...", "session_id": "uuid opcional"}` → `{"respuesta": "..."}`. Es el endpoint que atacan Red Turing y DeepTeam como objetivo `http` |
| `GET` | `/health` | Estado del servicio |
| `GET` | `/` | Info del servicio |

### Modo consola (sin Chatwoot)

```bash
python agente_sec_langfuse.py
```

---

## Documentación detallada

| Archivo | Contenido |
|---|---|
| `SEGURIDAD_COMPONENTES.md` | Explicación técnica completa de las 8 capas de seguridad |
| `LANGFUSE_COMPONENTES.md` | Explicación técnica completa de los 5 componentes de Langfuse |

---

## Autor

**Ing. Kevin Inofuente Colque** — [DataPath](https://datapath.pe)  
Módulo 1 — Sesión 1 — Programa AI Engineer

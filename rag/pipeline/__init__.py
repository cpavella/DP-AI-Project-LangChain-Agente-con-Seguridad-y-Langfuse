"""
Pipeline de INGESTA RAG (PDF → … → Qdrant).

Convención de nombres: prefijo por etapa (`a_`, `b_`, `c_`, …) para que el ORDEN
del pipeline se vea directo en el listado de archivos.

  a_extraction_validator.py  · valida la extracción (qué páginas necesitan OCR)
  b_ocr.py                    · OCR controlado en español para esas páginas
  c_normalize.py              · normaliza texto (Unicode, des-hifenización, espacios)
  d_boilerplate.py            · quita headers/footers/watermarks repetidos
  e_chunking.py               · chunking consciente de estructura (Markdown)
  f_metadata.py               · metadata + checksum/idempotencia para Qdrant

Flujo:
  PyMuPDF4LLM → [a validar] → (¿escaneado? → [b OCR es]) → c normalizar → d boilerplate
              → e chunking estructural + f metadata → embeddings → Qdrant
"""

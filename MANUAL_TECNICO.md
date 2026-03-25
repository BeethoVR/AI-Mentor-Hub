---

### Manual Técnico de Arquitectura

Este documento detalla el flujo de datos y las decisiones de ingeniería tomadas para la construcción del AI-Mentor Hub.

```markdown
# Manual Técnico y Decisiones de Arquitectura (ADR)

Este documento describe la arquitectura técnica del AI-Mentor Hub, un sistema RAG especializado para estudiantes de AI Engineering.

## 1. Arquitectura General

### Stack Tecnológico
- **Frontend:** Streamlit (interfaz interactiva)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) - CPU-only
- **LLM:** Google Gemini 2.5 Flash (generación)
- **Vector Store:** DocArrayInMemorySearch (en memoria con persistencia JSON)
- **Testing:** pytest

### Estructura del Proyecto
```
src/
├── app.py                  # Interfaz Streamlit principal
├── config.py               # Configuración centralizada
├── contracts/
│   └── schemas.py          # Esquemas Pydantic (ValidacionEntrada, RespuestaMentor)
├── core/
│   ├── agents.py           # Agente LangGraph con herramientas
│   ├── exceptions.py      # Excepciones customizadas
│   ├── guardrails.py       # Validación de entrada/salida
│   ├── ingestion.py        # Carga y procesamiento de PDFs
│   └── rag_base.py         # Motor RAG (consultas al mentor)
└── tools/
    ├── arxiv_search.py     # Búsqueda en ArXiv
    ├── security.py         # Validación de archivos subidos
    ├── web_search.py       # Búsqueda web (DuckDuckGo)
    └── wikipedia_search.py # Búsqueda en Wikipedia
```

## 2. Flujo de Datos (Pipeline RAG)

1.  **Ingesta (`src/core/ingestion.py`):**
    - Carga documentos PDF desde `data/` usando `PyPDFLoader`
    - Fragmentación con `RecursiveCharacterTextSplitter` (chunks de 1000 caracteres, overlap de 150)
    - Vectorización local con `HuggingFaceEmbeddings`
    - Persistencia en `DocArrayInMemorySearch`
    - Exportación a JSON (`processed_docs.json`) para durabilidad

2.  **Recuperación (Retrieval):**
    - Al recibir una consulta, busca los 4 fragmentos (`k=4`) con mayor similitud de coseno
    - Contexto concatenado y enviado al LLM

3.  **Generación (`src/core/rag_base.py`):**
    - Prompt ingeniado con reglas de respuesta estructurada
    - Uso de `GenerateContentConfig` con `response_schema=RespuestaMentor`
    - Validación final con `RespuestaMentor.model_validate_json()`

4.  **Guardrails (`src/core/guardrails.py`):**
    - Validación de entrada: detecta prompt injection, contenido irrelevante
    - Validación de salida: verifica que la respuesta sea segura y relevante

5.  **Agente (`src/core/agents.py`):**
    - agente LangGraph con herramientas de búsqueda (web, Wikipedia, ArXiv)
    - Responde preguntas que requieren información actualizada beyond los documentos cargados

## 3. Registro de Decisiones Arquitectónicas (ADR)

### ADR 001: Embeddings Locales vs API
- **Contexto:** Las cuotas gratuitas de APIs en la nube limitan severamente el proceso de ingesta masiva (Rate Limiting de 15-100 RPM).
- **Decisión:** Implementar `sentence-transformers` ejecutándose localmente.
- **Consecuencia:** Ingesta ilimitada y sin costo, a cambio de una carga inicial en CPU.

### ADR 002: Control de Dependencias para CPU Antiguo
- **Contexto:** Las versiones modernas de PyTorch y librerías científicas requieren instrucciones AVX2 no disponibles en Intel Core de 3ra generación.
- **Decisión:** Anclar `numpy==1.26.4`, `transformers==4.44.2`, `torch==2.2.2`.
- **Consecuencia:** Estabilidad total del sistema sin comprometer funcionalidades del RAG.

### ADR 003: Persistencia JSON vs Pickle
- **Contexto:** Serializar objetos complejos de LangChain con `pickle` genera errores de "Attribute lookup" al recargar.
- **Decisión:** Extraer `page_content` y `metadata` y guardarlos en JSON estándar.
- **Consecuencia:** Cargas seguras, independientes de la versión de Python.

### ADR 004: Cacheo de Embeddings
- **Contexto:** El modelo de embeddings se recargaba en cada consulta.
- **Decisión:** Usar `@lru_cache` en `get_embeddings_model()`.
- **Consecuencia:** Mejor rendimiento en consultas repetidas.

### ADR 005: Cacheo de Resultados RAG
- **Contexto:** Consultas repetidas generaban llamadas innecesarias a la API.
- **Decisión:** Dictionary-based cache con límite de 100 entradas.
- **Consecuencia:** Reducción de costos API y latencia.

### ADR 006: Rate Limiting
- **Contexto:** Sin control, usuarios pueden exceder cuotas API rápidamente.
- **Decisión:** Implementar `15 requests per minute` usando `time.time()`.
- **Consecuencia:** Protección contra abuso, cumplimiento de límites API.

### ADR 007: Sanitización de Consultas
- **Contexto:** Usuarios pueden injectar HTML, URLs o emails en queries.
- **Decisión:** Función `sanitize_query()` que remueve patrones peligroso.
- **Consecuencia:** Seguridad mejorada contra prompt injection.

## 4. Excepciones Customizadas (`src/core/exceptions.py`)

```python
class RAGQueryError(Exception): ...
class QuotaExceededError(Exception): ...
class APIServiceUnavailableError(Exception): ...
class DocumentLoadError(Exception): ...
class ValidationError(Exception): ...
```

## 5. Validación de Archivos (`src/tools/security.py`)

- Validación de extensión (.pdf)
- Validación de tamaño máximo (10MB)
- Validación de header PDF
- Protección contra path traversal
- Protección contra archivos ocultos
- Validación de múltiples archivos (máx 5)

## 6. Configuración Centralizada (`src/config.py`)

Parámetros configurables:
- `CHUNK_SIZE`, `CHUNK_OVERLAP` (fragmentación)
- `RETRIEVAL_K` (número de chunks recuperados)
- `LLM_TEMP_RAG`, `LLM_TEMP_GUARDRAILS`, `LLM_TEMP_AGENTE`, `LLM_TEMP_TITULO`
- `EMBEDDING_MODEL`
- `VECTOR_DB_PATH`
- `MAX_FILES_UPLOAD`, `MAX_FILE_SIZE`
- `RATE_LIMIT_RPM`

## 7. Testing

- **test_agents.py:** Inicialización del agente LangGraph
- **test_ingestion.py:** Carga JSON, manejo de PDFs corruptos
- **test_rag.py:** Consultas exitosas y sin API key
- **test_schemas.py:** Validación de esquemas Pydantic
- **test_security.py:** Validación de archivos, path traversal, límites

## 8. Ejecución

```bash
# Desarrollo
cd src && streamlit run app.py

# Tests
pytest tests/

# Reset de Base de Datos
rm data/processed_docs.json
```
```
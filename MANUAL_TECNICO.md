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
├── app.py # Interfaz Streamlit principal
├── config.py # Configuración centralizada
├── contracts/
│ └── schemas.py # Esquemas Pydantic (ValidacionEntrada, RespuestaMentor)
├── core/
│ ├── agents.py # Agente LangGraph con herramientas
│ ├── exceptions.py # Excepciones customizadas
│ ├── guardrails.py # Validación de entrada/salida
│ ├── ingestion.py # Carga y procesamiento de PDFs
│ └── rag_base.py # Motor RAG (consultas al mentor)
└── tools/
├── arxiv_search.py # Búsqueda en ArXiv
├── security.py # Validación de archivos subidos
├── web_search.py # Búsqueda web (DuckDuckGo)
└── wikipedia_search.py # Búsqueda en Wikipedia

````

## 2. Flujo de Datos (Pipeline RAG PRO)

1.  **Ingesta (`src/core/ingestion.py`):**
    - Carga documentos PDF desde `data/` usando `PyPDFLoader`.
    - **Detección de Idioma:** Analiza los primeros fragmentos con `langdetect` y guarda el idioma predominante en `project_metadata.json`.
    - Fragmentación con `RecursiveCharacterTextSplitter` (chunks de 2000 caracteres, overlap de 500).
    - Vectorización local con `HuggingFaceEmbeddings`.
    - Persistencia en `DocArrayInMemorySearch` (JSON).

2.  **Orquestación y Reescritura (`src/core/agents.py`):**
    - Al recibir una consulta, el **Planner** analiza el historial de mensajes y el idioma detectado.
    - **Query Rewriting:** Si la consulta es un seguimiento (ej: "dame más pasos"), la reescribe para que sea específica y esté en el idioma de los documentos.

3.  **Recuperación y Ensamblado (Retrieval):**
    - Busca `K * 1.5` fragmentos con mayor similitud de coseno.
    - **Deterministic Rerank:** Ordena los fragmentos por origen y número de página para reconstruir la secuencia lógica del documento.

4.  **Generación y Traducción (`src/core/agents.py`):**
    - El **Executor** genera la respuesta basándose únicamente en el contexto recuperado.
    - **Traducción Automática:** Si el contexto está en inglés pero el usuario pregunta en español, el Executor traduce la respuesta final al español.

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

### ADR 008: Query Rewriting para Continuidad de Contexto
- **Contexto:** En conversaciones multi-turno, los usuarios usan pronombres ("esto", "aquello") o frases incompletas ("dame más").
- **Decisión:** El **Planner** reescribe la consulta usando el historial de mensajes de SQLite.
- **Consecuencia:** Mejora drástica en la precisión del retrieval conversacional.

### ADR 009: Rerank Estructural Determinista (CPU-Optimized)
- **Contexto:** Modelos de Rerank basados en IA (Cross-Encoders) son costosos para hardware local antiguo.
- **Decisión:** Implementar un reordenamiento algorítmico por fuente y página tras el retrieval inicial.
- **Consecuencia:** Se mantiene la coherencia lógica (ej: pasos de una receta) con costo computacional nulo.

### ADR 010: RAG Adaptativo Bilingüe
- **Contexto:** Los documentos de estudio a menudo están en inglés, pero el usuario prefiere preguntar y recibir respuestas en español.
- **Decisión:** El sistema detecta el idioma del RAG e instruye al Planner para buscar en ese idioma y al Executor para traducir al español.
- **Consecuencia:** Se aprovecha la riqueza semántica de la fuente original sin barreras lingüísticas para el usuario.

### ADR 011: Contratos Estrictos con Pydantic (Structured Outputs)
- **Contexto:** Los LLMs tienden a alucinar formatos o devolver texto no estructurado, lo cual rompe la UI o la lógica de negocio posterior.
- **Decisión:** Forzar al modelo Gemini a utilizar `with_structured_output` inyectando esquemas Pydantic (`ValidacionEntrada`, `RespuestaMentor`, `PlanAgente`).
- **Consecuencia:** El sistema es 100% determinista en su estructura de datos. Se elimina la necesidad de usar expresiones regulares para extraer información de la respuesta del LLM, garantizando tipado seguro en toda la aplicación.

## 4. Excepciones Customizadas (`src/core/exceptions.py`)

```python
class RAGQueryError(Exception): ...
class QuotaExceededError(Exception): ...
class APIServiceUnavailableError(Exception): ...
class DocumentLoadError(Exception): ...
class ValidationError(Exception): ...
````

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

## 7. Testing y Mocking (Pruebas Aisladas)

El proyecto utiliza una estrategia estricta de **Mocking** (`unittest.mock` / `pytest-mock`) para aislar las pruebas de los servicios externos. 

**¿Por qué usamos Mocks?**
- **Cero Costo:** Las pruebas no hacen llamadas reales a la API de Google Gemini, evitando gastar cuota durante el desarrollo continuo.
- **Velocidad y Determinismo:** Las pruebas corren en milisegundos y no fallan por problemas de red o latencia de la API.
- **Aislamiento:** Se simulan las respuestas de la Base de Datos Vectorial y del LLM para probar exclusivamente la *lógica interna* de nuestra aplicación (manejo de errores, parsing de Pydantic, ruteo del agente).

**Archivos de prueba:**
- **test_agents.py:** Inicialización del agente LangGraph y simulación del flujo de nodos.
- **test_ingestion.py:** Carga JSON, manejo de PDFs corruptos.
- **test_rag.py:** Simulación de búsquedas vectoriales y respuestas de Gemini exitosas/fallidas.
- **test_schemas.py:** Validación de instanciación de esquemas Pydantic.
- **test_security.py:** Validación de archivos, path traversal, límites (sin dependencias externas).

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

```

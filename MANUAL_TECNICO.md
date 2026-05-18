# 🛠️ Manual Técnico: Arquitectura y Decisiones de Ingeniería (ADR)

AI-Mentor Hub es un sistema de asistencia de estudio de vanguardia diseñado bajo los principios de **Clean Architecture** y orquestación de **Agentes Autónomos**. Este manual detalla las tripas técnicas del sistema y el razonamiento detrás de cada decisión de diseño.

---

## 1. Arquitectura de Sistemas y Flujo de Datos

### Stack Tecnológico

- **Frontend:** Streamlit (interfaz interactiva)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) - CPU-only
- **LLM:** Google Gemini 3.1 Preview (generación)
- **Vector Store:** SQLite / DocArrayInMemorySearch (en memoria con persistencia JSON)
- **Testing:** pytest

### Vista de Alto Nivel
El sistema se divide en tres capas fundamentales:
1.  **Capa de Presentación (Streamlit):** Interfaz reactiva que gestiona la sesión, la carga de archivos y la visualización del chat.
2.  **Capa de Orquestación (LangGraph):** Motor de estado cíclico que coordina agentes especializados.
3.  **Capa de Conocimiento (RAG Engine PRO):** Pipeline de ingesta, vectorización y recuperación optimizada para hardware local.

### Estructura de Directorios
```text
src/
├── app.py                 # Punto de entrada y controlador de la UI
├── config.py              # Variables globales y configuración de hardware
├── state_manager.py       # Gestión de caché, rate-limits y títulos dinámicos
├── contracts/
│   └── schemas.py         # Modelos Pydantic para Structured Outputs
├── core/
│   ├── agents.py          # Grafo multiagente (Planner, Executor, Verifier)
│   ├── ingestion.py       # Ingesta de PDFs y Embeddings locales
│   ├── rag_base.py        # Motor RAG base (consultas directas)
│   ├── guardrails.py      # Seguridad y validación de contexto
│   ├── storage.py         # Manejo de persistencia de archivos PDF
│   └── exceptions.py      # Jerarquía de errores personalizados
└── tools/
    ├── security.py        # Validación estricta de archivos subidos
    ├── web_search.py      # Integración con DuckDuckGo
    ├── wikipedia_search.py # Integración con Wikipedia
    └── arxiv_search.py     # Integración con ArXiv (papers científicos)
```

---

## 2. Pipeline de Datos: El RAG PRO

### 2.1. Fase de Ingesta Inteligente
1.  **Carga:** Se utiliza `PyPDFLoader` para procesar documentos en la carpeta `data/`.
2.  **Detección de Idioma:** Se analiza el primer bloque de texto con `langdetect`. El resultado se persiste en `project_metadata.json` para que el Agente sepa en qué idioma buscar.
3.  **Fragmentación (Chunking):** 1500 caracteres con 200 de solapamiento. Se utiliza `RecursiveCharacterTextSplitter` para mantener la integridad de párrafos y oraciones.
4.  **Vectorización:** Uso de `HuggingFaceEmbeddings` con el modelo `all-MiniLM-L6-v2`. Se ejecuta 100% en CPU.
5.  **Persistencia:** Los vectores y metadatos se guardan en `data/processed_docs.json` usando `DocArrayInMemorySearch`.

### 2.2. Recuperación Avanzada (Retrieval)
-   **Neighbor Expansion:** Para evitar respuestas incompletas, al encontrar un fragmento relevante, el sistema recupera automáticamente las páginas adyacentes (anterior y posterior).
-   **Deterministic Rerank:** En lugar de re-rankers pesados de IA, se aplica un ordenamiento algorítmico por `source` y `page`, reconstruyendo la secuencia lógica original del autor.

---

## 3. Orquestación Multiagente (LangGraph)

El sistema utiliza un **Grafo de Estado** con memoria persistente en **SQLite** (`agent_memory.db`).

### Nodos del Grafo:
-   **Planner:** Realiza *Query Rewriting*. Si la pregunta es vaga o depende del historial, la reescribe para que sea una búsqueda semántica completa.
-   **Retriever:** Ejecuta la búsqueda en el Vector Store local o dispara el **Auto-Fallback** a la web si no encuentra nada.
-   **Executor:** Redacta la respuesta final en español, traduciendo si el contexto está en inglés. Aplica reglas de extracción *verbatim* para manuales técnicos o recetas.
-   **Verifier:** Compara la respuesta contra el contexto original. Si hay alucinaciones (hechos no presentes en el PDF), rechaza la respuesta y devuelve el flujo al Planner.

---

## 4. Registro de Decisiones de Arquitectura (ADR)

Este registro documenta las decisiones críticas que definen la robustez del sistema.

| ID | Título | Decisión y Contexto |
| :--- | :--- | :--- |
| **ADR-001** | **Embeddings Locales** | Se usa `sentence-transformers` en CPU local para evitar costos y rate-limits de APIs de embeddings externas. |
| **ADR-002** | **Pinning de Versiones** | Se anclan versiones específicas de `numpy`, `torch` y `transformers` para asegurar compatibilidad con CPUs antiguos (Intel Gen 3). |
| **ADR-003** | **JSON sobre Pickle** | La persistencia vectorial se hace en JSON para evitar errores de serialización y vulnerabilidades de seguridad de `pickle`. |
| **ADR-004** | **Caché de Modelos** | El modelo de embeddings se carga una sola vez usando `@lru_cache`, reduciendo la latencia de consulta en un 90%. |
| **ADR-005** | **Caché de Consultas** | Las respuestas a preguntas idénticas se sirven desde un caché en memoria (límite 100) para ahorrar cuota de Gemini. |
| **ADR-006** | **Rate Limiting** | Límite estricto de 15 RPM para proteger la cuota gratuita de Google AI Studio. |
| **ADR-007** | **Sanitización de Queries** | Eliminación de HTML, URLs y caracteres especiales antes de procesar la consulta para prevenir ataques de inyección. |
| **ADR-008** | **Query Rewriting** | Uso del historial de SQLite para resolver la "amnesia" conversacional y pronombres ambiguos. |
| **ADR-009** | **Rerank Estructural** | Ordenamiento determinista por página y documento tras el retrieval para mantener la coherencia sin carga de GPU. |
| **ADR-010** | **RAG Bilingüe** | Traducción dinámica en tiempo de consulta. Permite estudiar documentos en inglés preguntando en español. |
| **ADR-011** | **Structured Outputs** | Uso de `with_structured_output` (Pydantic) para forzar al LLM a devolver JSON válido. Elimina el uso de Regex frágiles. |
| **ADR-012** | **LangGraph Cíclico** | Paso de cadenas secuenciales a grafos para permitir bucles de retroalimentación (Refinamiento de respuestas). |
| **ADR-013** | **Neighbor Expansion** | Recuperación de páginas `n-1` y `n+1` para garantizar integridad en recetas y procesos largos. |
| **ADR-014** | **Checkpointing SQLite** | Almacenamiento del estado del agente en disco para persistencia de memoria entre reinicios del servidor. |
| **ADR-015** | **UV como Package Manager** | Uso de `uv` por su velocidad extrema y manejo impecable de entornos virtuales aislados. |

---

## 5. Seguridad y Validación

### Validación de Archivos (`tools/security.py`)
-   Verificación de firma mágica de PDF (Header `%PDF-`).
-   Límite de 36MB por archivo y 5 archivos máximo por carga.
-   Sanitización de nombres de archivo para prevenir Path Traversal.

### Guardrails de Entrada (`core/guardrails.py`)
Un sistema experto analiza la pregunta antes de que llegue al agente:
-   **Filtro de Inyección:** Detecta prompts que intentan "saltarse" las reglas.
-   **Filtro de Relevancia:** Valida si la pregunta tiene relación con el tema de la biblioteca, ahorrando tokens en consultas fuera de lugar.

---

## 6. Mantenimiento y Testing

### Estrategia de Mocking
Todas las pruebas unitarias (`tests/`) utilizan `unittest.mock` para simular las respuestas de la API de Gemini y de la base de datos. Esto permite un desarrollo **Zero-Cost** y **Offline-First** durante el ciclo de testing.

### Comandos de Mantenimiento
```bash
# Sincronizar dependencias
uv sync

# Ejecutar suite de pruebas
uv run pytest tests/ -v

# Limpiar memoria del sistema (Hard Reset)
rm -rf data/*
```

---

_Documento técnico oficial del proyecto AI-Mentor Hub. Actualizado: Mayo 2026._

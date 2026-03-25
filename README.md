# 🧠 AI-Mentor-Hub

Un sistema conversacional inteligente de doble motor diseñado para asistir en el estudio de cualquier tema que desees aprender. Funciona como un **asistente de estudio personalizado**: subes tus documentos PDF sobre cualquier temática (libros, artículos, apuntes, documentación técnica), y el sistema responde preguntas específicas sobre el contenido de esos archivos. La validación de seguridad asegura que solo se permiten consultas relacionadas con los temas presentes en los documentos cargados, manteniendo el enfoque de estudio.

El sistema combina un **Sistema RAG (Retrieval-Augmented Generation) Local** optimizado para hardware restringido, con un **Agente Autónomo ReAct** conectado a internet para búsquedas complementarias en la web. Construido con LangChain, LangGraph y la API de Google Gemini.

## 🚀 Características Principales

El proyecto implementa un patrón de "Modo de Operación" en la interfaz de usuario, permitiendo alternar entre dos cerebros analíticos:

### 1. 👨‍🏫 Mentor Local (Motor RAG)

El primer modo utiliza tu propia biblioteca de documentos PDF como base de conocimiento:

- **Embeddings Locales (Zero-Cost):** Vectorización de documentos ejecutada 100% en CPU local utilizando el modelo `all-MiniLM-L6-v2` de HuggingFace, evitando límites de cuota y costos de APIs externas. **Con cacheo** para no recargar el modelo en cada consulta.
- **Cacheo de Consultas:** Las preguntas repetidas se responden instantáneamente desde memoria, ahorrando cuota de API.
- **Persistencia Nativa Ligera:** Almacenamiento del índice vectorial con `DocArrayInMemorySearch` en formato JSON (compactado), garantizando compatibilidad y evitando errores de serialización (`pickle`).
- **Procesamiento Incremental Inteligente:** El sistema detecta automáticamente qué archivos PDF son nuevos y solo re-indexa los documentos que no existen en la base de conocimientos existente, optimizando el tiempo de carga.
- **Validación de Seguridad (Guardrails):** Capa de seguridad que analiza cada pregunta del usuario antes de ejecutarla. Verifica que la consulta esté relacionada con los temas presentes en los documentos PDF cargados (cualquier tema es válido — cocina, historia, medicina, etc.), detectando prompt injections y consultas sin relación. Solo rechaza si la pregunta no tiene nada que ver con el contenido cargado.
- **Respuestas Estructuradas con Schema:** Utiliza `Structured Outputs` nativos de Gemini para garantizar que las respuestas sigan un contrato Pydantic definido (`RespuestaMentor`), incluyendo tema, explicación técnica, código de ejemplo, referencias bibliográficas y sugerencias de estudio.
- **Gestión de Archivos Validada:** Validación completa de archivos subidos: verificación de tipo PDF, límite de tamaño (36MB por archivo), límite de cantidad (5 archivos por carga), sanitización de nombres para prevenir path traversal, y validación del header del archivo.

### 2. 🕵️‍♂️ Investigador Web (Agente ReAct)

El segundo modo actúa como un investigador autónomo que busca información actualizada en internet:

- **Orquestación con LangGraph:** Utiliza un agente ReAct moderno (`create_react_agent`) que aprovecha el _Tool Calling_ nativo de Gemini para razonar y ejecutar acciones iterativamente.
- **Herramientas Personalizadas Integradas:**
  - 🕸️ **Búsqueda Web (DuckDuckGo):** Para información actualizada, noticias recientes y datos en tiempo real.
  - 📚 **Wikipedia:** Para definiciones formales, teóricas e históricas de conceptos.
  - 📄 **ArXiv:** Para papers académicos e investigaciones científicas avanzadas.
- **Indicación de Fuentes:** El agente está configurado para devolver referencias al final de cada respuesta (URL, título de Wikipedia o artículo de ArXiv).

## 🏗️ Arquitectura Limpia (Clean Architecture)

El proyecto sigue estándares de la industria para mantener una base de código escalable, modular y mantenible:

```
AI-Mentor-Hub/
├── src/
│   ├── app.py                 # Frontend: Interfaz de usuario con Streamlit
│   ├── config.py              # Centralización de variables globales, modelos y logging
│   ├── contracts/
│   │   └── schemas.py         # Modelos de datos Pydantic para tipado estricto y validación
│   ├── core/                  # El "Cerebro" del sistema
│   │   ├── __init__.py
│   │   ├── exceptions.py      # Excepciones personalizadas (RAGQueryError, QuotaExceededError, etc.)
│   │   ├── ingestion.py       # Procesamiento de documentos y Vector Store local
│   │   ├── rag_base.py        # Lógica de consulta (Retrieval) y generación con Gemini
│   │   ├── agents.py          # Orquestación del grafo del Agente ReAct en LangGraph
│   │   └── guardrails.py      # Capa de seguridad pre-ejecución (validación de preguntas)
│   └── tools/
│       ├── __init__.py
│       ├── web_search.py      # Herramienta de búsqueda con DuckDuckGo
│       ├── wikipedia_search.py # Herramienta de búsqueda en Wikipedia
│       ├── arxiv_search.py    # Herramienta de búsqueda de papers académicos
│       └── security.py        # Validación de archivos subidos (tamaño, tipo, sanitización)
├── data/                      # Base de conocimientos (Documentos PDF)
├── tests/                     # Pruebas unitarias y de integración
└── .env                       # Variables de entorno (NO subir al repositorio)
```

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.12+
- **Gestor de Paquetes:** `uv` (por su extrema velocidad y resolución estricta).
- **Modelo Fundacional:** Google Gemini (`gemini-3.1-flash-lite-preview`) — seleccionado por sus cuotas más generosas para pruebas y desarrollo.
- **Frameworks de IA:** LangChain y LangGraph.
- **Vector Store & Embeddings:** DocArray y HuggingFace (`all-MiniLM-L6-v2`).
- **Validación de Datos:** Pydantic con validación de schema para structured outputs.
- **Frontend:** Streamlit.
- **Testing:** Pytest con técnicas de _Mocking_.

## ⚙️ Instalación y Configuración

**Nota de Hardware:** El proyecto está optimizado para funcionar en arquitecturas con restricciones de hardware (ej. procesadores Intel de generaciones anteriores sin AVX2 completo, como MacBook Pro Retina Mid 2012).

1. **Clonar el repositorio:**

   ```bash
   git clone [https://github.com/beethovr/AI-Mentor-Hub.git](https://github.com/beethovr/AI-Mentor-Hub.git)
   cd AI-Mentor-Hub
   ```

2. **Crear el entorno virtual con `uv`:**

   ```bash
   uv venv
   ```

3. **Instalar dependencias controladas:**
   _Nota: Se fuerzan versiones específicas de numpy y pyarrow para garantizar la compilación de librerías en procesadores x86_64 antiguos._

   ```bash
   uv sync
   ```

4. **Variables de entorno:**
   Crea un archivo `.env` en la raíz del proyecto y agrega tu API Key de Google:

   ```env
   GOOGLE_API_KEY=tu_clave_aqui
   ```

   También puedes crear un `.env.example` como plantilla (el proyecto incluye uno).

5. **Carga tus documentos de estudio:**
   La carpeta `data/` es donde se almacenan los archivos PDF que conformarán tu base de conocimiento. Por defecto, la carpeta está vacía — eres tú quien decide qué temática estudiar:
   - Sube tus propios PDFs (libros, artículos, apuntes, documentación) a traves de la interfase
   - El sistema automáticamente vectoriza el contenido y lo hace consultable
   - Puedes cargar hasta 5 archivos a la vez (límite de 36MB por archivo)
   
   *No hay temática predefinida: el Mentor se adapta a lo que tú quieras aprender.*

## 🖥️ Ejecución

Para levantar la interfaz gráfica y comenzar a interactuar con el sistema:

```bash
uv run streamlit run src/app.py
```

## 🧪 Pruebas Unitarias

El proyecto incluye pruebas automatizadas (aisladas en la carpeta `tests/`) que no consumen cuota de la API mediante el uso de _mocks_. Para ejecutarlas:

```bash
uv run pytest tests/ -v
```

### Tests disponibles:
- `test_rag.py` — Pruebas para el módulo RAG y manejo de errores
- `test_schemas.py` — Validación de schemas Pydantic
- `test_ingestion.py` — Pruebas del módulo de ingestion
- `test_agents.py` — Pruebas del agente investigador
- `test_security.py` — Pruebas de validación de archivos subidos

## 📝 Notas de Ingeniería

### Arquitectura del Agente ReAct

El sistema implementa un ciclo ReAct moderno utilizando el _Tool Calling_ nativo a través del motor cíclico de LangGraph. A diferencia del patrón clásico de 2022 (Thought/Action/Observation en texto), el modelo razona a nivel de API sobre el estado actual, decide qué herramienta invocar (Action), y LangGraph le inyecta el resultado (Observation) para generar la respuesta final.

El agente dispone de **3 herramientas** para realizar búsquedas complementarias:
1. **🔍 Búsqueda Web (DuckDuckGo):** Para información actualizada, noticias recientes y datos en tiempo real
2. **📚 Wikipedia:** Para definiciones formales, teóricas e históricas de conceptos
3. **📄 ArXiv:** Para papers académicos e investigaciones científicas avanzadas

### Gestión de Modelos

El sistema utiliza un único modelo para todas las operaciones: **gemini-3.1-flash-lite-preview**. Se seleccionó este modelo específicamente por sus cuotas más generosas, permitiendo más pruebas y desarrollo sin preocuparte por límites de uso.

### Mejoras en el Prompt

El prompt del RAG ha sido optimizado para:
- **No inventar información:** Si el contexto no tiene la respuesta, lo indica claramente
- **Recetas completas:** Incluye ingredientes Y pasos de preparación completos
- **Referencias precisas:** Solo cita lo que existe en el contexto, nunca crea capítulos falsos
- **Menor consumo de tokens:** Prompts en inglés para reducir costos

### Manejo de Errores

El proyecto implementa un sistema robusto de excepciones personalizadas:
- `RAGQueryError`: Errores generales en consultas RAG
- `QuotaExceededError`: Cuota de API excedida
- `APIServiceUnavailableError`: Servicio no disponible

### Logging

Todas las operaciones críticas incluyen logging estructurado configurable vía variables de entorno (`LOG_LEVEL`).

### Type Hints

El código utiliza type hints completos en los módulos principales para mejorar la mantenibilidad y detección de errores temprana:
- `ingestion.py`: Funciones con tipos de retorno explícitos
- `rag_base.py`: Type hints para cache y parámetros
- `app.py`: Type hints para funciones de carga de datos

### Seguridad

- **Sanitización de consultas:** Antes de procesar cualquier pregunta, se limpian caracteres especiales, HTML tags, URLs, emails y se limita la longitud (2000 caracteres).
- Validación de preguntas mediante guardrails (prompt injection detection)
- Validación de archivos subidos (sanitización, límites de tamaño/cantidad, verificación de tipo)
- Validación de API key antes de cada consulta
- Manejo seguro de variables de entorno
- **Rate Limiting:** Límite de 15 solicitudes por minuto para proteger la cuota de la API

## 🔧 Configuración Avanzada

El archivo `config.py` centraliza todos los parámetros configurables:

### Rutas

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `DATA_DIR` | Directorio de datos | `data/` |
| `VECTOR_DB_PATH` | Ruta del archivo JSON vectorial | `data/processed_docs.json` |

### Modelo de IA

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `MODELO_AGENTE` | Modelo único (RAG + Agente) | `gemini-3.1-flash-lite-preview` |
| `EMBEDDING_MODEL` | Modelo de embeddings | `all-MiniLM-L6-v2` |

### Chunking y Retrieval

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `CHUNK_SIZE` | Tamaño de chunk | `1000` |
| `CHUNK_OVERLAP` | Superposición entre chunks | `150` |
| `RETRIEVAL_K` | Número de documentos a recuperar | `4` |

### Temperaturas por tarea

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `LLM_TEMP_RAG` | Para respuestas del Mentor | `0.2` |
| `LLM_TEMP_GUARDRAILS` | Para validación de preguntas | `0.3` |
| `LLM_TEMP_AGENTE` | Para el agente investigador | `0.2` |
| `LLM_TEMP_TITULO` | Para generar títulos dinámicos | `0.3` |

### Rate Limiting

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `RATE_LIMIT_MAX` | Máximo requests por ventana | `15` |
| `RATE_LIMIT_WINDOW` | Ventana de tiempo (segundos) | `60` |

### Sanitización

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `MAX_QUERY_LENGTH` | Longitud máxima de consulta | `2000` |

### Cache

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `MAX_QUERY_CACHE_SIZE` | Máximo preguntas cacheadas | `100` |

### Logging

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `LOG_LEVEL` | Nivel de logging | `INFO` |

## ⚠️ Limitaciones Conocidas

- **Escala:** El vector store en memoria (`DocArrayInMemorySearch`) está diseñado para bases de conocimiento pequeñas-medias (ajustado a RAM disponible). Para volúmenes mayores, considerar ChromaDB o Pinecone.
- **Hardware:** El modelo de embeddings se ejecuta en CPU; el tiempo de indexación depende del hardware disponible.
- **Cuotas:** Las consultas a la API de Gemini consumen cuota; el agente está configurado con el modelo lite para optimizar el uso.

## ❓ Solución de Problemas

| Error | Solución |
|-------|----------|
| `GOOGLE_API_KEY not found` | Crear archivo `.env` con `GOOGLE_API_KEY=tu_clave` |
| `No hay información cargada` | Subir PDFs en la barra lateral y hacer clic en "Procesar y Aprender" |
| `Cuota excedida` | Esperar o revisar los límites de tu cuenta de Google AI Studio |
| `JSON corrupto` | Eliminar el archivo `data/processed_docs.json` y volver a cargar los PDFs |
| `Import error` | Ejecutar `uv sync` para instalar las dependencias |
| `Streamlit not found` | Ejecutar `uv pip install streamlit` |
| `No puedo hacer preguntas` | Verificar que los PDFs estén cargados y procesados |
| `Respuestas incorrectas` | Verificar que los PDFs contengan información sobre el tema consultado |

### Tips

- **Mejores resultados:** Usa PDFs con texto seleccionable (no imágenes escaneadas)
- **Carga más rápida:** Los PDFs ya procesados se cargan instantáneamente desde cache
- **Reset completo:** Eliminar `data/processed_docs.json` fuerza re-indexación completa

## 📚 Estructura de Validación de la Entrada

```python
class ValidacionEntrada:
    es_seguro: bool 		# False si hay prompt injection o código malicioso. True si es seguro.
    es_relevante: bool 		# False SOLO si es un tema ajeno. True si es relacionado al tema cargado
    motivo_rechazo: str  	# Si alguna es False, explica el rechazo. Si todo es OK, devuelve 'OK'
```

## 📚 Estructura de Respuesta del Mentor

Cuando usas el **Mentor Local**, las respuestas siguen este schema Pydantic:

```python
class Referencia:
    libro: str           # Nombre del libro fuente
    capitulo: str       # Capítulo donde se encontró
    concepto_clave: str # Concepto relacionado

class RespuestaMentor:
    tema: str                      # Tema central de la respuesta
    explicacion_completa: str      # Explicación detallada y completa (incluye pasos, ingredientes, etc.)
    codigo_ejemplo: str | None     # Código de ejemplo si aplica
    referencias: list[Referencia]  # Fuentes citadas
    sugerencia_estudio: str        # Recomendación para profundizar
```

## 🛠️ Optimizaciones de Rendimiento

El proyecto incluye varias optimizaciones para mejorar el rendimiento y proteger la cuota de la API:

- **Cache de Embeddings:** El modelo de embeddings `all-MiniLM-L6-v2` se carga una sola vez y se reutiliza en todas las llamadas siguientes (`@lru_cache`).
- **Cache del Agente:** El agente investigador ReAct se inicializa una sola vez y se reutiliza en cada consulta (`@lru_cache`).
- **Cache de Consultas:** Las preguntas repetidas se responden instantáneamente desde cache (límite de 100 entradas), evitando llamadas innecesarias a la API de Gemini.
- **JSON Compacto:** Los datos persistidos en `data/processed_docs.json` se almacenan sin indentación para reducir el tamaño del archivo.
- **Rate Limiting:** Límite configurable de 15 solicitudes por minuto para prevenir abuso y proteger la cuota de la API.

## 🧪 Cobertura de Tests

El proyecto cuenta con tests unitarios que cubren las funcionalidades principales:

| Test | Cobertura |
|------|-----------|
| `test_rag.py` | Consultas RAG, manejo de errores, cacheo |
| `test_schemas.py` | Validación de schemas Pydantic |
| `test_ingestion.py` | Carga de PDFs, procesamiento incremental |
| `test_agents.py` | Agente investigador ReAct |
| `test_security.py` | Validación de archivos subidos |

---

*Documentación actualizada: 2026-03-25*
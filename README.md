# 🧠 AI-Mentor Hub: Tu Asistente de Estudio Inteligente PRO

AI-Mentor Hub es un sistema conversacional de vanguardia diseñado como un **mentor personalizado de doble motor**. No es solo un sistema RAG (Retrieval-Augmented Generation); es una plataforma de aprendizaje que transforma documentos estáticos en experiencias interactivas, combinando el conocimiento privado de tus PDFs con la inmensidad de la web mediante orquestación multiagente.

Bajo la filosofía **"Local-First Intelligence"**, el sistema prioriza la privacidad y el uso eficiente de recursos, permitiendo que incluso hardware con restricciones (CPUs antiguas) actúe como un nodo de conocimiento avanzado.

---

## 🌟 La Filosofía: Por qué AI-Mentor Hub

El proyecto nace de la necesidad de centralizar y dar sentido a la información fragmentada:
- **Privacidad Total:** Tus documentos se indexan localmente, sin enviar el contenido a la nube para vectorización.
- **Contexto Infinito:** El mentor "lee" y recuerda tus libros, apuntes y documentación técnica.
- **Acceso Híbrido:** Si el conocimiento no está en tus archivos, el mentor activa su faceta de investigador autónomo para complementar tu estudio.
- **Optimización Hardware:** Diseñado para funcionar en CPUs sin AVX2 (ej. Intel Gen 3) mediante embeddings locales optimizados.

---

## 🚀 Características Principales (Motores Duales)

### 1. 👨‍🏫 Local Mentor (Motor RAG PRO)
Utiliza tu propia biblioteca PDF con capacidades avanzadas de razonamiento:
- **Detección Automática de Idioma:** Identifica si tus documentos están en español o inglés, optimizando búsquedas bilingües.
- **Cura de Amnesia (Query Rewriting):** Gracias a LangGraph y SQLite, entiende seguimientos como "¿y cómo se hace?" basándose en el historial previo.
- **Neighbor Expansion:** Recupera automáticamente las páginas anterior y posterior (`n-1`, `n+1`) de los fragmentos encontrados para reconstruir procesos largos o recetas cortadas.
- **Rerank Estructural Determinista:** Ordena los fragmentos por origen y página, manteniendo la coherencia lógica sin el costo computacional de un re-ranker de IA.
- **Embeddings Zero-Cost:** Ejecución 100% local con `all-MiniLM-L6-v2`.

### 2. 🕵️‍♂️ Investigador Web (Agente ReAct)
Un investigador autónomo que busca información actualizada en internet:
- **Orquestación LangGraph:** Ciclo *Plan-and-Solve* con Tool Calling nativo.
- **Herramientas Integradas:**
  - 🕸️ **Web Search (DuckDuckGo):** Noticias y datos en tiempo real.
  - 📚 **Wikipedia:** Definiciones formales e históricas.
  - 📄 **ArXiv:** Papers académicos e investigaciones científicas.
- **Auto-Fallback:** Si el RAG no encuentra la respuesta, el sistema dispara automáticamente la búsqueda externa.

---

## 🏗️ Arquitectura Limpia y Estructura

El proyecto sigue estándares de la industria para mantener una base de código escalable y modular:

```text
AI-Mentor-Hub/
├── src/
│   ├── app.py                 # Frontend Streamlit (Controlador ligero)
│   ├── config.py              # Configuración centralizada (Modelos, paths, temperaturas)
│   ├── state_manager.py       # Gestión de caché, rate-limits y títulos dinámicos
│   ├── contracts/
│   │   └── schemas.py         # Modelos Pydantic para Structured Outputs
│   ├── core/                  # El Cerebro del Sistema
│   │   ├── agents.py          # Grafo multiagente en LangGraph
│   │   ├── ingestion.py       # Procesamiento de documentos y Vector Store local
│   │   ├── rag_base.py        # Motor RAG y generación Gemini
│   │   ├── guardrails.py      # Seguridad pre-ejecución y validación
│   │   └── exceptions.py      # Jerarquía de errores personalizados
│   └── tools/
│       ├── security.py        # Validación de archivos (header, tamaño, sanitización)
│       ├── web_search.py      # Herramientas DuckDuckGo/Wiki/ArXiv
│       └── ...
├── data/                      # Almacenamiento vectorial y memoria SQLite
├── tests/                     # Suite de pruebas unitarias e integración
└── .env                       # Variables de entorno (NO subir al repo)
```

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.12+ (Gestionado con `uv`).
- **Modelo Fundacional:** Google Gemini 3.1 Flash (Lite & Preview).
- **Frameworks de IA:** LangChain y LangGraph.
- **Validación de Datos:** Pydantic (Structured Outputs nativos).
- **Vectores:** DocArray (Persistencia JSON compacta).
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local CPU).
- **Memoria:** SQLite (Checkpointer de LangGraph para estados de conversación).
- **Detección de Idioma:** `langdetect`.

---

## ⚙️ Instalación y Configuración

**Nota de Hardware:** Optimizado para procesadores antiguos (Intel Gen 3+). Se fuerzan versiones específicas de `numpy` y `pyarrow` para máxima compatibilidad.

1. **Clonar e Instalar:**
   ```bash
   git clone https://github.com/beethovr/AI-Mentor-Hub.git
   cd AI-Mentor-Hub
   uv sync
   ```

2. **Variables de Entorno:**
   Crea un archivo `.env` con tu API Key:
   ```env
   GOOGLE_API_KEY=tu_clave_de_google_ai_studio
   ```

3. **Cargar Conocimiento:**
   Sube tus PDFs a través de la interfaz (límite de 36MB/archivo, máx 5 archivos por carga). El sistema vectoriza automáticamente.

4. **Ejecutar:**
   ```bash
   uv run streamlit run src/app.py
   ```

---

## 🧪 Validación y Calidad

El proyecto incluye pruebas automatizadas que utilizan **Mocking** para evitar consumo de cuota API.

```bash
uv run pytest tests/ -v
```

### Cobertura de Tests:
- `test_rag.py`: Consultas, manejo de errores y caché.
- `test_agents.py`: Orquestación y fallback del investigador.
- `test_security.py`: Validación de archivos y sanitización.
- `test_ingestion.py`: Procesamiento incremental de PDFs.

---

## 🛡️ Seguridad y Robustez

- **Guardrails Context-Aware:** El sistema entiende respuestas cortas (ej. "el segundo") analizando el historial para permitir una interacción fluida sin sacrificar la seguridad.
- **Sanitización:** Limpieza de HTML, URLs y caracteres especiales en las consultas.
- **Rate Limiting:** Límite de 15 solicitudes por minuto para proteger la cuota de la API.
- **Validación de Archivos:** Verificación de firma mágica PDF y prevención de Path Traversal.

---

## 🔧 Configuración Avanzada (`config.py`)

| Variable | Descripción | Valor Default |
| :--- | :--- | :--- |
| `CHUNK_SIZE` | Tamaño de fragmento | `1500` |
| `RETRIEVAL_K` | Documentos recuperados | `6` |
| `RATE_LIMIT_MAX` | Requests por ventana | `15` |
| `MODELO_AGENTE` | Modelo de razonamiento | `gemini-3.1-flash-lite-preview` |
| `Import error` | Entorno no sincronizado | Ejecutar `uv sync`. |

### Tips

- **Mejores resultados:** Usa PDFs con texto seleccionable (no imágenes escaneadas).
- **Carga más rápida:** Los PDFs ya procesados se cargan instantáneamente desde la persistencia JSON.
- **Reset completo:** Al hacer clic en "Limpiar Todo" o eliminar `data/processed_docs.json` se fuerza la re-indexación completa.

---

## 📝 Notas de Ingeniería

### Arquitectura del Agente ReAct (LangGraph)
El sistema implementa un ciclo ReAct moderno utilizando el _Tool Calling_ nativo a través del motor cíclico de LangGraph con persistencia en SQLite (`agent_memory.db`). A diferencia del patrón clásico de 2022 (Thought/Action/Observation en texto), el modelo razona a nivel de API sobre el estado actual, decide qué herramienta invocar (Action), y LangGraph le inyecta el resultado (Observation) para generar la respuesta final.

El agente dispone de **3 herramientas** para realizar búsquedas complementarias:
1. **🔍 Búsqueda Web (DuckDuckGo):** Para información actualizada y datos en tiempo real.
2. **📚 Wikipedia:** Para definiciones formales e históricas de conceptos.
3. **📄 ArXiv:** Para papers académicos e investigaciones científicas avanzadas.

### Gestión de Modelos y Cuotas
El sistema utiliza un único modelo para todas las operaciones: **gemini-3.1-flash-lite-preview**. Se seleccionó este modelo específicamente por sus cuotas más generosas y latencia reducida, permitiendo un desarrollo iterativo fluido.

### Optimizaciones del Prompt RAG
El prompt del RAG ha sido refinado para:
- **Fidelidad Extrema:** Si el contexto no tiene la respuesta, el modelo tiene prohibido inventar.
- **Extracción Verbatim:** Reglas estrictas para copiar recetas y procedimientos técnicos sin parafrasear.
- **Eficiencia de Tokens:** Uso de instrucciones en inglés para el razonamiento interno, lo que reduce costos y mejora la precisión.

### Tipado y Mantenibilidad
El código utiliza **Type Hints** completos en todos los módulos principales (`ingestion.py`, `agents.py`, `app.py`), facilitando la detección de errores y mejorando la experiencia de desarrollo en IDEs modernos.

---

_Desarrollado como una solución PRO para el estudio inteligente, maximizando la potencia de los LLMs en entornos de hardware real._

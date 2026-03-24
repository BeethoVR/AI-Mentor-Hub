# 🧠 AI-Mentor-Hub

Un sistema conversacional inteligente de doble motor diseñado para asistir en el estudio de Ingeniería de Inteligencia Artificial y Agentes Autónomos. Combina un **Sistema RAG (Retrieval-Augmented Generation) Local** optimizado para hardware restringido, con un **Agente Autónomo ReAct** conectado a internet. Construido con LangChain, LangGraph y la API de Google Gemini.

## 🚀 Características Principales (Fase 1 & 2)

El proyecto implementa un patrón de "Modo de Operación" en la interfaz de usuario, permitiendo alternar entre dos cerebros analíticos:

1. **👨‍🏫 Mentor Local (Motor RAG):**

   - **Embeddings Locales (Zero-Cost):** Vectorización de documentos ejecutada 100% en CPU local utilizando el modelo `all-MiniLM-L6-v2` de HuggingFace, evitando límites de cuota y costos de APIs externas.
   - **Persistencia Nativa Ligera:** Almacenamiento del índice vectorial con `DocArrayInMemorySearch` en formato JSON, garantizando compatibilidad y evitando errores de serialización (`pickle`).
   - **Guardrails de Seguridad:** Implementa validación estricta mediante Pydantic y `Structured Outputs` para bloquear _prompt injections_ y rechazar preguntas fuera de dominio.
   - **Respuestas Estructuradas:** Devuelve respuestas técnicas formateadas con referencias exactas al material de origen (libros, capítulos) y bloques de código.

2. **🕵️‍♂️ Investigador Web (Agente ReAct):**
   - **Orquestación con LangGraph:** Utiliza un agente ReAct moderno (`create_react_agent`) que aprovecha el _Tool Calling_ nativo de Gemini para razonar y ejecutar acciones iterativamente.
   - **Herramientas Personalizadas:** Integra una herramienta de búsqueda web construida a medida utilizando la API directa de DuckDuckGo, esquivando problemas de dependencias en wrappers de terceros y optimizando el consumo de tokens.

## 🏗️ Arquitectura Limpia (Clean Architecture)

El proyecto sigue estándares de la industria para mantener una base de código escalable, modular y mantenible:

```text
AI-Mentor-Hub/
├── src/
│   ├── app.py                 # Frontend: Interfaz de usuario con Streamlit
│   ├── config.py              # Centralización de variables globales y modelos
│   ├── core/                  # El "Cerebro" del sistema
│   │   ├── schemas.py         # Modelos de datos Pydantic para tipado estricto
│   │   ├── ingestion.py       # Procesamiento de documentos y Vector Store local
│   │   ├── rag_base.py        # Lógica de consulta (Retrieval) y formateo
│   │   ├── agents.py          # Orquestación del grafo del Agente en LangGraph
│   │   └── guardrails.py      # Capa de seguridad pre-ejecución
│   └── tools/
│       └── web_search.py      # Lógica aislada de herramientas (DuckDuckGo)
├── data/                      # Base de conocimientos (Documentos PDF)
└── tests/                     # Pruebas unitarias y de integración
```

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.12+
- **Gestor de Paquetes:** `uv` (por su extrema velocidad y resolución estricta).
- **Modelos Fundacionales:** Google Gemini (`gemini-2.5-flash` y `gemini-3.1-flash-lite-preview`).
- **Frameworks de IA:** LangChain y LangGraph.
- **Vector Store & Embeddings:** DocArray y HuggingFace.
- **Validación de Datos:** Pydantic.
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
   uv pip install google-genai langchain langchain-community langchain-huggingface langchain-text-splitters pypdf docarray pydantic python-dotenv pytest pytest-mock streamlit duckduckgo-search langgraph
   uv pip install "numpy<2" "transformers<4.45.0" "sentence-transformers<3.0.0" "pyarrow>=14.0.1,<15.0.0"
   ```

4. **Variables de entorno:**
   Crea un archivo `.env` en la raíz del proyecto y agrega tu API Key de Google:

   ```env
   GOOGLE_API_KEY=tu_clave_aqui
   ```

5. **Aquí se añade la bibliografía cargada:**
   Se colocan tus archivos PDF de estudio en la carpeta `data/`.

## 🖥️ Ejecución

Para levantar la interfaz gráfica y comenzar a interactuar con el sistema:

```bash
uv run streamlit run src/app.py
```

## 🧪 Pruebas Unitarias

El proyecto incluye pruebas automatizadas (aisladas en la carpeta `tests/`) que no consumen cuota de la API mediante el uso de _mocks_. Para ejecutarlas:

```bash
uv run python -m pytest -v
```

## 📝 Notas de Ingeniería

- **Evolución del Patrón ReAct:** Aunque el agente no imprime explícitamente las cadenas de texto clásicas de _Thought/Action/Observation_ del paper original de 2022, el sistema implementa un ciclo ReAct moderno. Utilizando el _Tool Calling_ nativo a través del motor cíclico de LangGraph, el modelo razona a nivel de API sobre el estado actual, decide qué herramienta invocar (Action), y LangGraph le inyecta el resultado (Observation) para generar la respuesta final.

- **Resolución de Dependencias:** Se optó por invocar directamente `duckduckgo-search` en `src/tools/web_search.py` para construir una _Custom Tool_ robusta, garantizando resiliencia frente a caídas de la red y evitando incompatibilidades del ecosistema.

- **Manejo de Cuotas:** Se configuró el Agente Investigador para utilizar el modelo `gemini-3.1-flash-lite-preview` (o alternativas eficientes), previniendo errores `429 RESOURCE_EXHAUSTED` y `503 UNAVAILABLE` durante los bucles de razonamiento.

- **Tipado Estricto:** Se aplicó validación de tipos estáticos (`Type Casting`) y comprobación de Pylance en todo el Core para garantizar la estabilidad del software.

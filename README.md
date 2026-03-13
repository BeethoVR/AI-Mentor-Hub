# 🧠 AI-Mentor Hub

AI-Mentor Hub es un sistema RAG (Retrieval-Augmented Generation) diseñado para asistir en el estudio de Ingeniería de Inteligencia Artificial y Agentes Autónomos. Utiliza un motor de búsqueda semántica local y el LLM Gemini 2.5 Flash para responder preguntas técnicas basándose estrictamente en bibliografía especializada.

## 🚀 Características Principales (Fase 1 & 2)

- **Ingesta Híbrida:** Extrae y fragmenta texto de PDFs usando LangChain.
- **Embeddings Locales (Zero-Cost):** Vectorización de documentos ejecutada 100% en CPU utilizando `all-MiniLM-L6-v2` de HuggingFace, evitando límites de cuota de APIs externas.
- **Persistencia Nativa Ligera:** Almacenamiento del índice vectorial de `DocArray` en formato JSON, garantizando compatibilidad y evitando errores de serialización (`pickle`).
- **Generación Estructurada:** Integración con el SDK oficial `google-genai` forzando salidas en formato JSON validadas mediante contratos de **Pydantic**.
- **Interfaz Web:** Interfaz conversacional construida con **Streamlit**, incluyendo manejo de estado de sesión e historial.

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.12
- **Gestor de Paquetes:** `uv` (por su extrema velocidad y resolución estricta)
- **Orquestación:** LangChain
- **LLM:** Gemini 2.5 Flash (Google GenAI SDK)
- **Vector Store:** DocArray InMemory Search
- **Testing:** Pytest con técnicas de _Mocking_

## ⚙️ Instalación y Configuración

El proyecto está optimizado para funcionar en arquitecturas con restricciones de hardware (ej. procesadores Intel de generaciones anteriores sin AVX2 completo, como MacBook Pro Retina Mid 2012).

1. **Clonar el repositorio:**

   ```bash
   git clone [https://github.com/BeethoVR/AI-Mentor-Hub.git](https://github.com/BeethoVR/AI-Mentor-Hub.git)
   cd AI-Mentor-Hub
   ```

2. Crear el entorno virtual con uv:

   Bash:

   uv venv

3. Instalar dependencias controladas:
   Nota: Se fuerzan versiones específicas de numpy y pyarrow para garantizar la compilación de C++ en procesadores x86_64 antiguos.

   Bash:

   uv pip install google-genai langchain langchain-community langchain-huggingface langchain-text-splitters pypdf docarray pydantic python-dotenv pytest pytest-mock streamlit
   uv pip install "numpy<2" "transformers<4.45.0" "sentence-transformers<3.0.0" "pyarrow>=14.0.1,<15.0.0"

4. Variables de entorno:
   Crea un archivo .env en la raíz del proyecto y agrega tu API Key:
   Fragmento de código

   GOOGLE_API_KEY=tu_clave_aqui

5. Añadir la bibliografía:

   Coloca tus archivos PDF en la carpeta data/.

🖥️ Ejecución

Para levantar la interfaz gráfica y comenzar a interactuar con el mentor:

Bash:

uv run streamlit run src/app.py

🧪 Pruebas Unitarias

El proyecto incluye pruebas automatizadas que no consumen cuota de la API (mediante mocks). Para ejecutarlas:

Bash:

uv run python -m pytest -v

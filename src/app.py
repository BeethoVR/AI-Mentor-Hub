import streamlit as st
import os
import shutil
from dotenv import load_dotenv

from config import setup_logging
from langchain_google_genai import ChatGoogleGenerativeAI

logger = setup_logging(__name__)

# --- Importaciones de tu Arquitectura Limpia ---
from config import DATA_DIR, MODELO_AGENTE 
from core.ingestion import setup_vector_db
from core.rag_base import consultar_mentor
from core.rag_base import RAGQueryError, QuotaExceededError, APIServiceUnavailableError
from core.guardrails import validar_pregunta
from core.agents import ejecutar_agente
from tools.security import validate_uploaded_files

# 1. CONFIGURACIÓN INICIAL
st.set_page_config(
    page_title="AI-Mentor Hub", 
    page_icon="📚", 
    layout="wide"
)
load_dotenv()

# Rutas de persistencia
#DATA_DIR = "data"
JSON_DB = os.path.join(DATA_DIR, "processed_docs.json")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 2. GESTIÓN DE LA BASE DE DATOS (Caché y Persistencia)
@st.cache_resource(show_spinner=False)
def load_active_db():
    """
    Carga la BD: 
    1. Del JSON si ya existe (Rápido)
    2. De los PDFs si no hay JSON
    3. Retorna None si está vacío
    """
    try:
        if os.path.exists(JSON_DB):
            return setup_vector_db()
        elif os.path.exists(DATA_DIR) and any(f.endswith('.pdf') for f in os.listdir(DATA_DIR)):
            return setup_vector_db()
        return None
    except Exception as e:
        st.error(f"Error al cargar la base de conocimientos: {e}")
        return None

@st.cache_data(show_spinner=False)
def generar_titulo_tema(archivos: tuple) -> str:
    """Genera un título dinámico usando Gemini basado en los documentos."""
    if not archivos:
        return "Conocimiento General"
    
    try:
        # Usamos una temperatura baja (0.3) para que sea directo y conciso
        llm = ChatGoogleGenerativeAI(model=MODELO_AGENTE, temperature=0.3)
        lista_nombres = ", ".join(archivos)
        
        prompt = (
            f"Based in the names of the loaded documents, generate a very "
            f"attractive/nice and short title (max 4 words) that describes the "
            f"general theme of this library. Answered ONLY with the title, " 
            f"no quotes or explanations. Documents: {lista_nombres}"
        )
        
        respuesta = llm.invoke(prompt)
        # --- EXTRACCIÓN ROBUSTA (Manejo de Multimodalidad) ---
        raw_content = respuesta.content
        
        if isinstance(raw_content, list) and len(raw_content) > 0 and isinstance(raw_content[0], dict):
            # Si es una lista de bloques, extraemos el texto del diccionario
            title = raw_content[0].get('text', str(raw_content)).strip()
        elif isinstance(raw_content, str):
            # Si LangChain ya lo parseó como string, lo limpiamos directo
            title = raw_content.strip()
        else:
            # Fallback de seguridad
            title = str(raw_content).strip()
        # ----------------------------------------------------
        
        # Limpiamos posibles comillas que a veces el modelo añade por terquedad
        title = title.replace('"', '').replace("'", "")
        
        return f"{title}" if title else "Biblioteca Activa (AI API fail)"
    except Exception as e:
        return str(e) #"Biblioteca Activa (AI API out line)" # Fallback en caso de error de red
    

# Carga inicial al abrir la app
vector_db = load_active_db()

# Inicializamos la llave del uploader si no existe
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# 3. BARRA LATERAL (Gestión de Conocimiento)
with st.sidebar:
    st.title("⚙️ Configuración")
    
    st.subheader("📁 Cargar Documentos")
    uploaded_files = st.file_uploader(
                                    "Añade PDFs a tu biblioteca", 
                                    type="pdf", 
                                    accept_multiple_files=True,
                                    help="Los archivos se indexarán localmente en tu PC.",
                                    key=f"uploader_{st.session_state.uploader_key}" # to remove after upload and trigger re-render
                                )
    
    if st.button("🚀 Procesar y Aprender", use_container_width=True):
        if uploaded_files:
            valid_files, validation_errors = validate_uploaded_files(uploaded_files)
            
            if validation_errors:
                for error in validation_errors:
                    st.error(f"⚠️ {error}")
            
            if not valid_files:
                st.warning("No hay archivos válidos para procesar.")
            else:
                new_files = []
                old_files = []
                
                # 1. Aseguramos que la carpeta data/ exista
                os.makedirs(DATA_DIR, exist_ok=True)
                
                # 2. Filtramos y guardamos solo los archivos nuevos
                for f in valid_files:
                    file_route = os.path.join(DATA_DIR, f.name)
                    
                    # Comprobamos si el archivo ya existe físicamente en la carpeta
                    if os.path.exists(file_route):
                        old_files.append(f.name)
                    else:
                        # Si no existe, lo guardamos en el disco
                        with open(file_route, "wb") as f_uploaded:
                            f_uploaded.write(f.getbuffer())
                        new_files.append(f)
                
                # 3. Damos feedback inmediato sobre los repetidos
                if old_files:
                    old_names = ", ".join(old_files)
                    st.warning(f"⚠️ Documentos ignorados (ya existían): {old_names}")
                
                # 4. Solo ejecutamos la ingesta pesada si hay archivos nuevos
                if new_files:
                    with st.spinner("Procesando nuevos documentos..."):
                        for f in new_files:
                            with open(os.path.join(DATA_DIR, f.name), "wb") as buffer:
                                buffer.write(f.getbuffer())
                        
                        # Simplemente lo llamas. El script detectará qué es nuevo.
                        vector_db = setup_vector_db() 
                        
                        st.cache_resource.clear()
                        st.success("¡Biblioteca actualizada!")

                        # 3. EL TRUCO DE UX:
                        # Cambiamos la llave para forzar que el uploader se vacíe
                        st.session_state.uploader_key += 1

                        # Refrescamos la interfaz para que el cambio sea inmediato
                        st.rerun()
        else:
            st.warning("Selecciona archivos primero.")

    st.divider()

    # Mostrar documentos actuales en la BD y el Tema Generado
    st.header("📚 Biblioteca Actual")
     
    # Obtenemos los archivos físicos como fuente de verdad
    archivos_en_data = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')] if os.path.exists(DATA_DIR) else []
    
    if vector_db and archivos_en_data:
        # 1. Generamos el tema dinámico si no existe en la memoria
        if "tema_biblioteca" not in st.session_state or "Biblioteca Activa" in st.session_state.tema_biblioteca:
            # Aquí asumimos que vector_db ya está inicializado. 
            # Si tienes una función para obtenerlo, úsala, por ejemplo: get_vector_db()
            try:
                with st.spinner("Analizando temática de los PDFs..."):
                    tema = generar_titulo_tema(tuple(archivos_en_data))
                    logger.info(f"Tema detectado: {tema}")
                    st.session_state.tema_biblioteca = tema
                    
            except Exception:
                # Fallback por si la base de datos está vacía al iniciar
                st.session_state.tema_biblioteca = "Sube documentos para analizar su contenido."

        # 2. Imprimimos el valor dinámico, NUNCA un texto fijo
        st.info(f"**Tema detectado:**\n{st.session_state.tema_biblioteca}")

    else:
        st.info("No hay documentos cargados.")

    st.divider()
    st.subheader("⚙️ Modo de Operación")
    
    modo_respuesta = st.radio("Selecciona el \"cerebro\" a utilizar:",
                             ["👨‍🏫 Mentor Local (PDFs)", "🕵️‍♂️ Investigador Web (Agente)"],
                             help="El Mentor lee tus libros. El Investigador busca en internet"
                            )
    st.divider()

    if vector_db:
        # Leemos directamente los archivos en la carpeta data/
        try:
            # Intentamos leer directamente los archivos en la carpeta data
            files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
            if files:
                for f in files:
                    st.caption(f"✅ {f}")
            else:
                st.info("No hay informacion de documentos cargados.")

        except Exception as e:
            st.caption("Conectado a la base de conocimientos.")
    else:
        st.info("No hay documentos cargados.")

    st.divider()

    st.subheader("⚠️ Zona de Peligro")
    
    # Checkbox de seguridad para habilitar el botón
    confirmar_borrado = st.checkbox("Entiendo que esto borrará toda la biblioteca")
    
    if st.button("🗑️ Limpiar Todo", use_container_width=True, disabled=not confirmar_borrado):
        with st.spinner("Eliminando persistencia..."):
            # 1. Borrar PDFs físicos
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR)
            
            # 2. Borrar el JSON de la base vectorial
            if os.path.exists(JSON_DB):
                os.remove(JSON_DB)
            
            # 3. Limpiar caché de Streamlit y memoria de chat
            st.cache_resource.clear()
            st.session_state.messages = []
            
            st.success("Memoria borrada con éxito.")
            st.rerun()


# 4. INTERFAZ DE CHAT PRINCIPAL
# Título dinámico
if vector_db:
    st.title("🧠 AI Mentor Hub")
    st.caption("Consultando tu biblioteca personal de documentos.")
else:
    st.title("📚 Bienvenido a tu Biblioteca AI")
    st.markdown("Carga archivos PDF en la barra lateral para comenzar a chatear con tus datos.")

st.divider()

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de usuario
if prompt := st.chat_input("Haz una pregunta sobre tus documentos..."):
    if not vector_db:
        st.error("⚠️ Error: No hay información cargada para responder.")
    else:
        # Mostramos el mensaje del usuario en la interfaz
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # --- CAPA 1: GUARDRAIL DE SEGURIDAD ---
            with st.spinner("🛡️ Validando seguridad y relevancia..."):
                # Obtenemos los temas directamente de la UI que ya generamos antes
                # (Asegúrate de que la variable 'tema_generado' esté accesible, 
                # o usa 'Biblioteca de documentos' como fallback)
                temas_contexto = locals().get('tema_generado', 'Temas académicos de los documentos cargados')
                
                # Le pasamos los NOMBRES EXACTOS de los archivos, no solo el tema
                archivos_actuales = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')] if os.path.exists(DATA_DIR) else []
                temas_contexto = ", ".join(archivos_actuales) if archivos_actuales else "Documentos técnicos de IA"

                validacion = validar_pregunta(prompt, temas_contexto)
            
            # --- DECISIÓN DEL GUARDRAIL ---
            if not validacion.es_seguro or not validacion.es_relevante:
                # Rechazamos la consulta y no ejecutamos el RAG
                mensaje_rechazo = f"🛑 **Consulta rechazada:** {validacion.motivo_rechazo}"
                st.warning(mensaje_rechazo)
                st.session_state.messages.append({"role": "assistant", "content": mensaje_rechazo})
            
            else:
                # --- CAPA 2: EJECUCIÓN DEL FLUJO ELEGIDO ---
                if modo_respuesta == "👨‍🏫 Mentor Local (PDFs)":
                    if not vector_db:
                        st.error("No hay biblioteca de documentos cargada.")
                        st.session_state.messages.append({"role": "assistant", "content": "Error: No hay biblioteca cargada."})
                    else:
                        # --- CAPA 2: EJECUCIÓN DEL RAG ---
                        with st.spinner("📚 Consultando tu biblioteca personal..."):
                            try:
                                respuesta = consultar_mentor(vector_db, prompt)
                                
                                # (Aquí va tu lógica de formateo de respuesta Pydantic de la Fase 1)
                                full_response = f"### {respuesta.tema}\n\n"
                                full_response += f"{respuesta.explicacion_tecnica}\n\n"
                                
                                if respuesta.codigo_ejemplo:
                                    full_response += f"**💻 Ejemplo de código:**\n```python\n{respuesta.codigo_ejemplo}\n```\n\n"
                                
                                full_response += "**📖 Referencias encontradas:**\n"
                                for ref in respuesta.referencias:
                                    full_response += f"- *{ref.libro}* (Cap. {ref.capitulo}): {ref.concepto_clave}\n"
                                
                                full_response += f"\n\n> 💡 **Sugerencia:** {respuesta.sugerencia_estudio}"
                                
                                st.markdown(full_response)
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                                
                            except QuotaExceededError as e:
                                st.error(f"⚠️ Cuota excedida: {e}")
                                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
                            except APIServiceUnavailableError as e:
                                st.error(f"⚠️ Servicio no disponible: {e}")
                                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
                            except RAGQueryError as e:
                                st.error(f"❌ Error en la consulta: {e}")
                                st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {e}"})

                elif modo_respuesta == "🕵️‍♂️ Investigador Web (Agente)":
                    with st.spinner("🕵️‍♂️ El Agente está investigando en la web..."):
                        
                        # Llamamos a la función encapsulada que nos devuelve un string limpio
                        full_response = ejecutar_agente(prompt)
                        
                        # Imprimimos en la interfaz
                        st.markdown(f"**🌐 Respuesta del Investigador:**\n\n{full_response}")
                        st.session_state.messages.append({"role": "assistant", "content": f"**🌐 Respuesta del Investigador:**\n\n{full_response}"})    
                    
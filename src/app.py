import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# Importamos tu lógica de backend de la Fase 1
from ingestion import setup_vector_db
from rag_base import consultar_mentor

# 1. CONFIGURACIÓN INICIAL
st.set_page_config(
    page_title="AI-Mentor Hub", 
    page_icon="📚", 
    layout="wide"
)
load_dotenv()

# Rutas de persistencia
DATA_DIR = "data"
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
    if os.path.exists(JSON_DB):
        return setup_vector_db()
    elif os.path.exists(DATA_DIR) and any(f.endswith('.pdf') for f in os.listdir(DATA_DIR)):
        return setup_vector_db()
    return None

def save_uploaded_files(uploaded_files):
    """Guarda archivos físicos y fuerza re-indexación"""
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Limpiamos el caché para que setup_vector_db procese lo nuevo
    st.cache_resource.clear()
    return setup_vector_db()

# Carga inicial al abrir la app
vector_db = load_active_db()

# 3. BARRA LATERAL (Gestión de Conocimiento)
with st.sidebar:
    st.title("⚙️ Configuración")
    
    st.subheader("📁 Cargar Documentos")
    new_files = st.file_uploader(
        "Añade PDFs a tu biblioteca", 
        type="pdf", 
        accept_multiple_files=True,
        help="Los archivos se indexarán localmente en tu PC."
    )
    
    if st.button("🚀 Indexar Contenido", use_container_width=True):
        if new_files:
            with st.spinner("Procesando y vectorizando..."):
                vector_db = save_uploaded_files(new_files)
                st.success("¡Base de datos actualizada!")
                st.rerun()
        else:
            st.warning("Selecciona archivos primero.")

    st.divider()

    # Mostrar documentos actuales en la BD
    st.subheader("📚 Biblioteca Actual")
    if vector_db:
        # Intentamos obtener los documentos del objeto vector_db
        # Dependiendo de si usas FAISS, Chroma o una lista simple, 
        # la forma de acceder cambia. Probemos con esta que es la más común:
        try:
            # Si vector_db es una lista de documentos o tiene el atributo .docs
            docs_actuales = vector_db.docs if hasattr(vector_db, 'docs') else []
            
            if docs_actuales:
                sources = list(set([os.path.basename(doc.metadata.get('source', 'Desconocido')) for doc in docs_actuales]))
                for s in sources:
                    st.caption(f"✅ {s}")
            else:
                # Si no hay .docs, intentamos leer directamente los archivos en la carpeta data
                files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
                if files:
                    for f in files:
                        st.caption(f"✅ {f}")
                else:
                    st.info("Información cargada desde JSON (fuentes no visibles).")
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
    st.title("🧠 Personal AI-Mentor")
    st.caption("Consultando tu biblioteca personal de documentos.")
else:
    st.title("📚 Bienvenido a tu Notebook AI")
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
        # Añadir y mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                respuesta = consultar_mentor(vector_db, prompt)
                
                # Manejo de respuesta Pydantic (Fase 1)
                if isinstance(respuesta, str):
                    full_response = respuesta
                    st.error(full_response)
                else:
                    # Formateo elegante de la respuesta estructurada
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
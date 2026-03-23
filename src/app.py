import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# Importamos tu lógica de backend de la Fase 1
from ingestion import setup_vector_db
from rag_base import consultar_mentor

from langchain_google_genai import ChatGoogleGenerativeAI
from guardrails import validar_pregunta

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

@st.cache_data(show_spinner=False)
def generar_titulo_tema(archivos: tuple) -> str:
    """Genera un título dinámico usando Gemini basado en los documentos."""
    if not archivos:
        return "Conocimiento General"
    
    try:
        # Usamos una temperatura baja (0.3) para que sea directo y conciso
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        lista_nombres = ", ".join(archivos)
        
        prompt = (
            f"Basado en los siguientes nombres de documentos, genera un título "
            f"atractivo y muy corto (máximo 4 palabras) que describa el tema general "
            f"de esta biblioteca. Responde ÚNICAMENTE con el título, sin comillas "
            f"ni explicaciones. Documentos: {lista_nombres}"
        )
        
        respuesta = llm.invoke(prompt)
        return respuesta.content.strip()
    except Exception as e:
        return "Biblioteca Activa" # Fallback en caso de error de red
    
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
            with st.spinner("Procesando y vectorizando nuevos documentos..."):
                for f in new_files:
                    with open(os.path.join(DATA_DIR, f.name), "wb") as buffer:
                        buffer.write(f.getbuffer())
                
                # Simplemente lo llamas. El script detectará qué es nuevo.
                vector_db = setup_vector_db() 
                
                st.cache_resource.clear()
                st.success("¡Biblioteca actualizada!")
                st.rerun()
        else:
            st.warning("Selecciona archivos primero.")

    st.divider()

    # Mostrar documentos actuales en la BD y el Tema Generado
    st.subheader("📚 Biblioteca Actual")
    
    # Obtenemos los archivos físicos como fuente de verdad
    archivos_en_data = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')] if os.path.exists(DATA_DIR) else []
    
    if vector_db and archivos_en_data:
        # 1. El LLM decide el tema (usamos tuple para que el caché funcione)
        tema_generado = generar_titulo_tema(tuple(archivos_en_data))
        
        # 2. Pintamos el tema con un diseño destacado
        st.markdown(f"✨ **Tema detectado:** \n*{tema_generado}*")
        
        # 3. Listamos los documentos debajo
        st.divider()
        # Beetho: Elimine esta sección, porque me duplicaba el listado de libros
        #if archivos_en_data:
        #    for f in archivos_en_data:
        #        st.caption(f"✅ {f}")
    else:
        st.info("No hay documentos cargados.")

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
                # --- CAPA 2: EJECUCIÓN DEL RAG ---
                with st.spinner("📚 Consultando tu biblioteca personal..."):
                    respuesta = consultar_mentor(vector_db, prompt)
                    
                    # (Aquí va tu lógica de formateo de respuesta Pydantic de la Fase 1)
                    if isinstance(respuesta, str):
                        st.error(respuesta)
                    else:
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
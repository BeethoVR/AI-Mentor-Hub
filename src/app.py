import streamlit as st
import uuid
from dotenv import load_dotenv

# --- Importaciones de Arquitectura Limpia ---
from config import setup_logging
from core.ingestion import setup_vector_db
from core.guardrails import validar_pregunta, sanitize_query
from core.agents import ejecutar_grafo_multiagente
from tools.security import validate_uploaded_files

# --- Importaciones de Nuevos Módulos (Refactor) ---
from core.storage import save_uploaded_pdfs, get_current_pdf_names
from core.state_manager import (
    load_active_db, 
    generar_titulo_tema, 
    check_rate_limit, 
    clear_system_memory
)

logger = setup_logging(__name__)

# ==========================================
# 1. CONFIGURACIÓN INICIAL Y ESTADO
# ==========================================
st.set_page_config(page_title="AI-Mentor Hub", page_icon="📚", layout="wide")
load_dotenv()

# Inicializamos variables de sesión seguras
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

# Carga inicial de la base de datos vectorial
vector_db = load_active_db()

# ==========================================
# 2. BARRA LATERAL (Gestión de Conocimiento)
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuración")
    st.subheader("📁 Cargar Documentos")
    
    uploaded_files = st.file_uploader(
        "Añade PDFs a tu biblioteca", 
        type="pdf", 
        accept_multiple_files=True,
        help="Los archivos se indexarán localmente en tu PC.",
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    # --- PROCESAMIENTO DE ARCHIVOS ---
    if st.button("🚀 Procesar y Aprender", use_container_width=True):
        if not uploaded_files:
            st.warning("Selecciona archivos primero.")
        else:
            valid_files, validation_errors = validate_uploaded_files(uploaded_files)
            
            for error in validation_errors:
                st.error(f"⚠️ {error}")
                
            if valid_files:
                # Usamos el nuevo módulo storage.py
                new_files, old_files = save_uploaded_pdfs(valid_files)
                
                if old_files:
                    st.warning(f"⚠️ Ignorados (ya existían): {', '.join(old_files)}")
                
                if new_files:
                    with st.spinner("Procesando e indexando nuevos documentos..."):
                        vector_db = setup_vector_db() 
                        st.cache_resource.clear() # Limpiamos caché para forzar recarga
                        st.success("¡Biblioteca actualizada!")
                        st.session_state.uploader_key += 1 # Truco UX: vaciar uploader
                        st.rerun()

    st.divider()

    # --- MOSTRAR BIBLIOTECA ACTUAL ---
    st.header("📚 Biblioteca Actual")
    archivos_actuales = get_current_pdf_names()
    
    if vector_db and archivos_actuales:
        # Generar o recuperar título dinámico (state_manager.py)
        if "tema_biblioteca" not in st.session_state or "Biblioteca Activa" in st.session_state.tema_biblioteca:
            with st.spinner("Analizando temática..."):
                st.session_state.tema_biblioteca = generar_titulo_tema(tuple(archivos_actuales))
        
        st.info(f"**Tema detectado:**\n{st.session_state.tema_biblioteca}")
        
        # Listar archivos
        for f in archivos_actuales:
            st.caption(f"✅ {f}")
    else:
        st.info("No hay documentos cargados.")

    st.divider()

    # --- ZONA DE PELIGRO ---
    st.subheader("⚠️ Zona de Peligro")
    confirmar_borrado = st.checkbox("Entiendo que esto borrará toda la biblioteca")
    
    if st.button("🗑️ Limpiar Todo", use_container_width=True, disabled=not confirmar_borrado):
        with st.spinner("Eliminando persistencia y memoria..."):
            clear_system_memory() # Usamos el nuevo módulo state_manager.py
            st.success("Sistema reiniciado con éxito.")
            st.rerun()


# ==========================================
# 3. INTERFAZ DE CHAT PRINCIPAL
# ==========================================
if vector_db:
    st.title("🧠 AI Mentor Hub")
    st.caption("Consultando tu biblioteca personal de documentos y la web.")
else:
    st.title("📚 Bienvenido a tu Biblioteca AI")
    st.markdown("Carga archivos PDF en la barra lateral para comenzar a aprender.")
st.divider()

# Mostrar historial de chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- LÓGICA DE RESPUESTA ---
if prompt := st.chat_input("Haz una pregunta sobre tus documentos..."):
    if not vector_db:
        st.error("⚠️ Error: No hay información cargada para responder. Por favor, sube un PDF.")
    elif not check_rate_limit():
        st.error("⚠️ Has alcanzado el límite de solicitudes. Espera un momento (1 min).")
    else:
        # 1. Mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 2. Sanitización
            prompt_limpio = sanitize_query(prompt)
            
            # 3. Guardrails de Seguridad
            with st.spinner("🛡️ Validando seguridad..."):
                # Usamos el tema detectado dinámicamente como fuente de verdad para la relevancia
                tema_actual = st.session_state.get("tema_biblioteca", "Conocimiento General")
                
                validacion = validar_pregunta(
                    pregunta_usuario=prompt_limpio, 
                    contexto_biblioteca=tema_actual,
                    historial_mensajes=st.session_state.messages[:-1]
                )
            
            # 4. Decisión
            if not validacion.es_seguro or not validacion.es_relevante:
                mensaje_rechazo = f"🛑 **Consulta rechazada:** {validacion.motivo_rechazo}"
                st.warning(mensaje_rechazo)
                st.session_state.messages.append({"role": "assistant", "content": mensaje_rechazo})
            else:
                # 5. Ejecución del Agente
                with st.spinner("🤖 El equipo Multiagente está analizando tu consulta..."):
                    try:
                        full_response = ejecutar_grafo_multiagente(
                            pregunta=prompt_limpio, 
                            vector_db=vector_db, 
                            thread_id=st.session_state.thread_id
                        )
                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    except Exception as e:
                        error_msg = f"❌ Error en el Orquestador: {e}"
                        st.error(error_msg)
                        logger.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

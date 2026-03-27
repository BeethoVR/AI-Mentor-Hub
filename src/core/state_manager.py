import streamlit as st
import os
import shutil
import time
from typing import Optional

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import ChatGoogleGenerativeAI

from config import DATA_DIR, VECTOR_DB_PATH as JSON_DB, MODELO_AGENTE, LLM_TEMP_TITULO, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW, setup_logging
from core.ingestion import setup_vector_db

logger = setup_logging(__name__)

# --- GESTIÓN DE LA BASE DE DATOS ---

@st.cache_resource(show_spinner=False)
def load_active_db() -> Optional[DocArrayInMemorySearch]:
    """
    Carga la BD vectorial:
    1. Del JSON si ya existe (Rápido)
    2. De los PDFs si no hay JSON pero hay archivos
    3. Retorna None si está vacío
    """
    try:
        if os.path.exists(JSON_DB):
            return setup_vector_db()
        elif os.path.exists(DATA_DIR) and any(f.endswith('.pdf') for f in os.listdir(DATA_DIR)):
            return setup_vector_db()
        return None
    except Exception as e:
        logger.error(f"Error al cargar la base de conocimientos: {e}")
        st.error(f"Error al cargar la base de conocimientos: {e}")
        return None

# --- GESTIÓN DE TEMÁTICA ---

@st.cache_data(show_spinner=False)
def generar_titulo_tema(archivos: tuple) -> str:
    """Genera un título dinámico usando Gemini basado en los documentos cargados."""
    if not archivos:
        return "Conocimiento General"
    
    try:
        llm = ChatGoogleGenerativeAI(model=MODELO_AGENTE, temperature=LLM_TEMP_TITULO)
        lista_nombres = ", ".join(archivos)
        
        prompt = (
            f"Based in the names of the loaded documents, generate a very "
            f"attractive/nice and short title (max 4 words) that describes the "
            f"general theme of this library. Answered ONLY with the title, " 
            f"no quotes or explanations. Documents: {lista_nombres}"
        )
        
        respuesta = llm.invoke(prompt)
        
        # Extracción robusta multimodal
        raw_content = respuesta.content
        if isinstance(raw_content, list) and len(raw_content) > 0 and isinstance(raw_content[0], dict):
            title = raw_content[0].get('text', str(raw_content)).strip()
        elif isinstance(raw_content, str):
            title = raw_content.strip()
        else:
            title = str(raw_content).strip()
            
        title = title.replace('"', '').replace("'", "")
        return f"{title}" if title else "Biblioteca Activa"
    except Exception as e:
        logger.warning(f"Error al generar título: {e}")
        return "Biblioteca Activa"

# --- GESTIÓN DE ESTADO Y SEGURIDAD ---

def check_rate_limit() -> bool:
    """Verifica si el usuario puede hacer más requests. Retorna True si está permitido."""
    if "request_times" not in st.session_state:
        st.session_state.request_times = []
        
    now = time.time()
    st.session_state.request_times = [
        t for t in st.session_state.request_times if now - t < RATE_LIMIT_WINDOW
    ]
    if len(st.session_state.request_times) >= RATE_LIMIT_MAX:
        return False
        
    st.session_state.request_times.append(now)
    return True

def clear_system_memory():
    """Elimina persistencia en disco y limpia el caché en memoria."""
    # 1. Borrar PDFs físicos y metadatos
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 2. Limpiar caché de Streamlit y estado
    st.cache_resource.clear()
    st.cache_data.clear()
    
    # 3. Reiniciar variables de sesión seguras
    st.session_state.messages = []
    st.session_state.tema_biblioteca = ""
    st.session_state.request_times = []
    
    logger.info("Memoria del sistema borrada (Hard Reset).")

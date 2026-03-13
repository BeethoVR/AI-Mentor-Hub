import streamlit as st
import os
from dotenv import load_dotenv

# Importamos tu backend
from ingestion import setup_vector_db
from rag_base import consultar_mentor

# 1. Configuración inicial de la página
st.set_page_config(page_title="AI-Mentor Hub", page_icon="🧠", layout="centered")
load_dotenv()

# 2. Caché de la Base de Datos Vectorial
@st.cache_resource(show_spinner="Levantando base de conocimientos...")
def init_db():
    return setup_vector_db()

vector_db = init_db()

# --- NUEVA SECCIÓN: BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    st.markdown("Gestiona tu sesión de estudio con el Mentor.")
    
    # Botón para limpiar el estado de la sesión
    if st.button("🗑️ Nueva Conversación", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola, Roberto! Soy tu mentor virtual. ¿Qué duda técnica tienes hoy?"}
        ]
        st.rerun() # Esto recarga la interfaz inmediatamente para borrar los textos viejos

    st.divider()
    
    st.markdown("**📚 Libros Indexados:**")
    st.markdown("- *AI Engineering* (Chip Huyen)")
    st.markdown("- *AI Agents in Action* (Michael Lanham)")
    
    st.divider()
    st.caption("Bootcamp: Especialización AI Engineer en Python")
# ----------------------------------------------

# 3. Interfaz Principal
st.title("🧠 AI-Mentor Hub")
st.markdown("Tu asistente personal sobre Ingeniería de IA y Agentes Autónomos.")
st.divider()

# 4. Inicializar el historial de chat en memoria
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola, Roberto! Soy tu mentor virtual. ¿Qué duda técnica tienes hoy?"}
    ]

# 5. Dibujar los mensajes guardados en la pantalla
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Capturar la nueva pregunta del usuario
if prompt := st.chat_input("Ej. ¿Cuáles son los desafíos de pasar un modelo a producción?"):
    
    # Añadir pregunta al historial y mostrarla
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 7. Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando la bibliografía técnica..."):
            
            # Llamamos a tu lógica de RAG
            respuesta = consultar_mentor(vector_db, prompt)
            
            if isinstance(respuesta, str):
                # Manejo de errores (ej. Rate Limit)
                st.error(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            else:
                # Formateo Markdown del objeto Pydantic
                respuesta_md = f"**TEMA:** {respuesta.tema.upper()}\n\n"
                respuesta_md += f"{respuesta.explicacion_tecnica}\n\n"
                
                if respuesta.codigo_ejemplo:
                    respuesta_md += f"**Ejemplo de Código:**\n```python\n{respuesta.codigo_ejemplo}\n```\n\n"
                
                respuesta_md += "**📚 Fuentes Consultadas:**\n"
                for ref in respuesta.referencias:
                    respuesta_md += f"- *{ref.libro}* (Cap. {ref.capitulo}): {ref.concepto_clave}\n"
                
                respuesta_md += f"\n**💡 Sugerencia de estudio:** {respuesta.sugerencia_estudio}"
                
                st.markdown(respuesta_md)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_md})
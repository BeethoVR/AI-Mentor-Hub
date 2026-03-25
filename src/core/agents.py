from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from tools.web_search import herramienta_web
from tools.wikipedia_search import herramienta_wiki
from tools.arxiv_search import herramienta_arxiv

from config import MODELO_AGENTE, LLM_TEMP_AGENTE


# --- CONFIGURACIÓN DEL AGENTE REACT CON LANGGRAPH ---
def inicializar_agente_investigador():
    """
    Crea un agente ReAct usando LangGraph y herramientas personalizadas.
    """

    # Lista de herramientas disponibles para el agente
    herramientas = [herramienta_web, herramienta_wiki, herramienta_arxiv]

    llm = ChatGoogleGenerativeAI(
        model=MODELO_AGENTE, 
        temperature=LLM_TEMP_AGENTE
    )
    
    agente = create_react_agent(llm, tools=herramientas)
    
    return agente

@lru_cache(maxsize=1)
def get_agente():
    """Retorna el agente cacheado para evitar re-inicialización."""
    return inicializar_agente_investigador()

def ejecutar_agente(pregunta: str) -> str:
    """
    Ejecuta el agente investigador, procesa la respuesta cruda de LangGraph/Gemini,
    y devuelve un texto limpio listo para la interfaz de usuario.
    """
    # 1. Usamos el agente cacheado
    agente = get_agente()
    
    # 2. Invocamos al agente
    pregunta += ", get back the source reference at the end (URL, Wikipedia title or ArXiv Articule)."
    respuesta_agente = agente.invoke({"messages": [("user", pregunta)]})
    
    # 3. Extraemos el contenido crudo
    raw_content = respuesta_agente['messages'][-1].content
    
    # 4. Limpiamos y formateamos (Encapsulamiento perfecto)
    if isinstance(raw_content, list) and len(raw_content) > 0 and isinstance(raw_content[0], dict):
        return raw_content[0].get('text', str(raw_content))
    
    return str(raw_content)
import logging
from langchain_core.tools import Tool
from ddgs import DDGS

logger = logging.getLogger("ai-mentor-hub.tools")

def buscar_en_internet(query: str) -> str:
    """Busca en DuckDuckGo y devuelve los primeros 3 resultados como texto."""
    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query) if query else "general"
    
    query = query.strip()
    if not query:
        return "Error: Consulta vacía."
    
    logger.info(f"[TOOL] Web Search iniciada con query: {query[:50]}...")
    
    try:
        # Try with DDGS - simpler initialization
        with DDGS() as ddgs:
            resultados = ddgs.text(query, max_results=3)
        
        if not resultados:
            logger.info("[TOOL] Web Search: Sin resultados")
            return "No se encontraron resultados en internet."
        
        texto_formateado = ""
        for r in resultados:
            texto_formateado += f"- {r['title']}: {r['body']}\n"
        logger.info(f"[TOOL] Web Search: {len(resultados)} resultados encontrados")
        return texto_formateado
    
    except TypeError as e:
        logger.error(f"[TOOL] Web Search: Error en la librería - {str(e)}")
        return "Error: El servicio de búsqueda no está disponible temporalmente. Intenta usar Wikipedia o ArXiv."
    except Exception as e:
        error_msg = str(e)
        if "rate" in error_msg.lower() or "limit" in error_msg.lower():
            logger.error(f"[TOOL] Web Search: Rate limit excedido")
            return "Error: Rate limit de DuckDuckGo excedido. Intenta de nuevo en unos segundos."
        logger.error(f"[TOOL] Web Search error: {error_msg}")
        return f"Error al buscar en internet: {error_msg}"

# Envolvemos nuestra función en una Herramienta oficial de LangChain
herramienta_web = Tool(
    name="Busqueda_Internet",
    func=buscar_en_internet,
    description="Useful ONLY when you need updated information from the internet, recent news, or real-time data (like the weather). Input: a clear search query."
)
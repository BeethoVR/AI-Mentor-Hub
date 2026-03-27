import logging
import requests
import xml.etree.ElementTree as ET
from langchain_core.tools import Tool

logger = logging.getLogger("ai-mentor-hub.tools")

def buscar_papers_arxiv(query: str) -> str:
    """Busca papers académicos en ArXiv y devuelve el título y resumen del más relevante."""
    logger.info(f"[TOOL] ArXiv Search iniciada con query: {query[:50]}...")
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', namespace)
        
        if entry is None:
            logger.info("[TOOL] ArXiv: Sin resultados")
            return "No se encontraron papers académicos sobre este tema en ArXiv."
            
        titulo = entry.find('atom:title', namespace).text.replace('\n', ' ')
        resumen = entry.find('atom:summary', namespace).text.replace('\n', ' ')
        
        logger.info(f"[TOOL] ArXiv: Paper encontrado - {titulo[:50]}...")
        return f"Paper encontrado ({titulo}): {resumen[:800]}..."
        
    except Exception as e:
        logger.error(f"[TOOL] ArXiv error: {str(e)}")
        return f"Error al consultar ArXiv: {str(e)}"

herramienta_arxiv = Tool(
    name="Buscar_Papers_ArXiv",
    func=buscar_papers_arxiv,
    description="Useful ONLY when you need to search for scientific research, academic papers, or advanced technical literature about Machine Learning, statistics, or AI. Input: search terms in English (e.g., 'neural networks')."
)
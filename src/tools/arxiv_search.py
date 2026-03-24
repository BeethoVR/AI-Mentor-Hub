import requests
import xml.etree.ElementTree as ET
from langchain_core.tools import Tool

def buscar_papers_arxiv(query: str) -> str:
    """Busca papers académicos en ArXiv y devuelve el título y resumen del más relevante."""
    # API pública de ArXiv (sin autenticación)
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parseamos el XML que devuelve ArXiv
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', namespace)
        
        if entry is None:
            return "No se encontraron papers académicos sobre este tema en ArXiv."
            
        titulo = entry.find('atom:title', namespace).text.replace('\n', ' ')
        resumen = entry.find('atom:summary', namespace).text.replace('\n', ' ')
        
        return f"Paper encontrado ({titulo}): {resumen[:800]}..." # Limitamos a 800 caracteres
        
    except Exception as e:
        return f"Error al consultar ArXiv: {str(e)}"

herramienta_arxiv = Tool(
                            name="Buscar_Papers_ArXiv",
                            func=buscar_papers_arxiv,
                            description="Útil SOLO cuando necesitas buscar investigaciones científicas, papers académicos o literatura técnica avanzada sobre Machine Learning, estadística o IA. Entrada: términos de búsqueda en inglés (ej. 'neural networks')."
                        )
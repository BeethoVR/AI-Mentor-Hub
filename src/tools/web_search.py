from langchain_core.tools import Tool
# Importamos DuckDuckGo directamente, esquivando el bug de LangChain
from duckduckgo_search import DDGS

def buscar_en_internet(query: str) -> str:
    """Busca en DuckDuckGo y devuelve los primeros 3 resultados como texto."""
    try:
        resultados = DDGS().text(query, max_results=3)
        if not resultados:
            return "No se encontraron resultados en internet."
        
        texto_formateado = ""
        for r in resultados:
            texto_formateado += f"- {r['title']}: {r['body']}\n"
        return texto_formateado
    
    except Exception as e:
        return f"Error al buscar en internet: {str(e)}"

# Envolvemos nuestra función en una Herramienta oficial de LangChain
herramienta_web = Tool(
                            name="Busqueda_Internet",
                            func=buscar_en_internet,
                            description="Útil SOLO cuando necesitas información actualizada de internet, noticias recientes, o datos en tiempo real (como el clima). Entrada: una consulta de búsqueda clara."
                        )
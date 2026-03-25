import logging
import requests
from langchain_core.tools import Tool

logger = logging.getLogger("ai-mentor-hub.tools")

def buscar_en_wikipedia(query: str) -> str:
    """Busca un concepto en Wikipedia en español y devuelve el resumen."""
    logger.info(f"[TOOL] Wikipedia Search iniciada con query: {query[:50]}...")
    
    headers = {
        "User-Agent": "AI-Mentor-Hub/1.0 (educational project; contact: info@aimentorhub.local)"
    }
    url = f"https://es.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&utf8=&format=json"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 403:
            logger.error("[TOOL] Wikipedia: Access forbidden (rate limit)")
            return "Error: Wikipedia está limitando el acceso. Intenta de nuevo más tarde."
        
        response.raise_for_status()
        datos = response.json()
        
        resultados = datos.get('query', {}).get('search', [])
        if not resultados:
            logger.info("[TOOL] Wikipedia: Sin resultados")
            return "No se encontraron definiciones exactas en Wikipedia."
        
        mejor_resultado = resultados[0]
        snippet = mejor_resultado['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
        
        logger.info(f"[TOOL] Wikipedia: Resultado encontrado - {mejor_resultado['title']}")
        return f"Definición de Wikipedia ({mejor_resultado['title']}): {snippet}..."
        
    except Exception as e:
        logger.error(f"[TOOL] Wikipedia error: {str(e)}")
        return f"Error al consultar Wikipedia: {str(e)}"

# Envolvemos la función en una Herramienta oficial de LangChain
herramienta_wiki = Tool(
                            name="Buscar_Concepto_Wikipedia",
                            func=buscar_en_wikipedia,
                            description="Útil SOLO cuando necesitas la definición formal, teórica o histórica de un concepto, algoritmo o término (ej. '¿Qué es Machine Learning?'). NO usar para noticias recientes."
                        )
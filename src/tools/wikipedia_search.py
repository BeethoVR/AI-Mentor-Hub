import requests
from langchain_core.tools import Tool

def buscar_en_wikipedia(query: str) -> str:
    """Busca un concepto en Wikipedia en español y devuelve el resumen."""
    # Usamos la API pública de Wikipedia (no requiere API Key)
    url = f"https://es.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&utf8=&format=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Verifica que no haya error HTTP
        datos = response.json()
        
        resultados = datos.get('query', {}).get('search', [])
        if not resultados:
            return "No se encontraron definiciones exactas en Wikipedia."
        
        # Tomamos el mejor resultado y limpiamos las etiquetas HTML residuales
        mejor_resultado = resultados[0]
        snippet = mejor_resultado['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
        
        return f"Definición de Wikipedia ({mejor_resultado['title']}): {snippet}..."
        
    except Exception as e:
        return f"Error al consultar Wikipedia: {str(e)}"

# Envolvemos la función en una Herramienta oficial de LangChain
herramienta_wiki = Tool(
                            name="Buscar_Concepto_Wikipedia",
                            func=buscar_en_wikipedia,
                            description="Útil SOLO cuando necesitas la definición formal, teórica o histórica de un concepto, algoritmo o término (ej. '¿Qué es Machine Learning?'). NO usar para noticias recientes."
                        )
import os
from typing import Dict
from google import genai
from google.genai import types
from langchain_community.vectorstores import DocArrayInMemorySearch
from contracts.schemas import RespuestaMentor

from core.exceptions import RAGQueryError, QuotaExceededError, APIServiceUnavailableError
from config import MODELO_AGENTE, RETRIEVAL_K, LLM_TEMP_RAG, MAX_QUERY_CACHE_SIZE

_query_cache: Dict[str, RespuestaMentor] = {}

def consultar_mentor(vector_db: DocArrayInMemorySearch, pregunta: str) -> RespuestaMentor:
    """
    Realiza la búsqueda semántica y genera la respuesta con la nueva API de Gemini.
    Implementa cacheo básico para evitar llamadas repetidas a la API.
    """
    pregunta_normalizada = pregunta.strip().lower()
    
    if pregunta_normalizada in _query_cache:
        return _query_cache[pregunta_normalizada]
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RAGQueryError("GOOGLE_API_KEY no está configurada. Por favor, configura tu API key en el archivo .env")
    
    try:
        # 1. Retrieval: Buscar en el índice DocArray
        docs = vector_db.similarity_search(pregunta, k=RETRIEVAL_K)
        contexto = "\n\n".join([doc.page_content for doc in docs])

        # 2. Inicializar el cliente de Gemini
        client = genai.Client(api_key=api_key)

        #AI Engineering and Autonomous Agents
        prompt = f"""
        Answer the user's question based ONLY on the provided context.
        
        CONTEXT:
        {contexto}
        
        QUESTION:
        {pregunta}
        
        RULES:
        1. Use ONLY info from context. Never invent, guess, or fabricate.
        2. If context doesn't have the answer, say: "No encontré información sobre ese tema en los documentos cargados."
        3. For recipes: include INGREDIENTS and FULL PREPARATION STEPS (all steps, not partial).
        4. For references: only cite what exists in context. Never create fake chapter names like "Cap. X" - use "Documento" or the actual filename if available.
        5. Keep code examples only if user explicitly asks for code.
        """

        # 3. Generación usando Structured Outputs nativos de la nueva API
        response = client.models.generate_content(
            model=MODELO_AGENTE, # 'gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RespuestaMentor, # Inyección directa del contrato
                temperature=LLM_TEMP_RAG,
            ),
        )
        
        # 4. Validación Final
        if response.text is None:
            raise ValueError("La respuesta del modelo no contiene texto.")
        
        resultado = RespuestaMentor.model_validate_json(response.text)
        
        # Agregar al cache con límite de tamaño
        if len(_query_cache) >= MAX_QUERY_CACHE_SIZE:
            _query_cache.pop(next(iter(_query_cache)))
        _query_cache[pregunta_normalizada] = resultado
        
        return resultado

    except Exception as e:
        if ("EXCEEDED_QUOTA" in str(e)) or ("RESOURCE_EXHAUSTED" in str(e)):
            raise QuotaExceededError("Se ha excedido la cuota de la API. Por favor, revisa tu uso y límites.")
        elif "UNAVAILABLE" in str(e):
            raise APIServiceUnavailableError("El servicio de la API no está disponible en este momento. Por favor, intenta nuevamente más tarde.")
        
        raise RAGQueryError(f"Error en la consulta (RAG): {str(e)}")
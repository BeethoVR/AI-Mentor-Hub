import os
from google import genai
from google.genai import types
from contracts.schemas import RespuestaMentor

from config import MODELO_AGENTE

def consultar_mentor(vector_db, pregunta: str):
    """
    Realiza la búsqueda semántica y genera la respuesta con la nueva API de Gemini.
    """
    try:
        # 1. Retrieval: Buscar en el índice DocArray
        docs = vector_db.similarity_search(pregunta, k=4)
        contexto = "\n\n".join([doc.page_content for doc in docs])

        # 2. Inicializar el NUEVO cliente de Gemini
        # El cliente busca automáticamente os.getenv("GEMINI_API_KEY") o "GOOGLE_API_KEY"
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        #AI Engineering and Autonomous Agents
        prompt = f"""
        You are a AI-Mentor Hub, an expert in the material that is saved/uploaded
        in this RAG/database.
        Base your answer strictly on the books and documents that we have upload in 
        the RAG/database.
        
        TECHNICAL CONTEXT EXTRACTED:
        {contexto}
        
        STUDENT'S QUESTION:
        {pregunta}
        
        INSTRUCTIONS:
        1. Answer in a technical and professional manner, using the information 
            from the context and all the books/documents in the RAG/database.
        2. Cite the book(s) or author(s) if the information is in the context.
        3. If the answer requires code or explaining a pattern (e.g. ReAct), 
            document it clearly and pedagogically, always in kind matter. And if 
            there's something not in the information, say it clearly.
        """

        # 3. Generación usando Structured Outputs nativos de la nueva API
        response = client.models.generate_content(
            model=MODELO_AGENTE, # 'gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RespuestaMentor, # Inyección directa del contrato
                temperature=0.2, # low temperature to be analytical, not creative
            ),
        )
        
        # 4. Validación Final
        if response.text is None:
            raise ValueError("La respuesta del modelo no contiene texto.")
        
        return RespuestaMentor.model_validate_json(response.text)

    except Exception as e:
        if ("EXCEEDED_QUOTA" in str(e)) or ("RESOURCE_EXHAUSTED" in str(e)):
            return "Error: Se ha excedido la cuota de la API. Por favor, revisa tu uso y límites."
        elif "UNAVAILABLE" in str(e):
            return "Error: El servicio de la API no está disponible en este momento. Por favor, intenta nuevamente más tarde."
        
        return f"Error en la consulta (RAG): {str(e)}"
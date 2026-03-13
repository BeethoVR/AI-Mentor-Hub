import os
from google import genai
from google.genai import types
from schemas import RespuestaMentor

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

        prompt = f"""
        Eres el AI-Mentor Hub, un experto en Ingeniería de IA y Agentes Autónomos.
        Basa tu respuesta estrictamente en los libros de Chip Huyen y Michael Lanham.
        
        CONTEXTO TÉCNICO EXTRAÍDO:
        {contexto}
        
        PREGUNTA DEL ESTUDIANTE:
        {pregunta}
        
        INSTRUCCIONES:
        1. Responde de forma técnica y profesional.
        2. Cita el libro o autor si la información está en el contexto.
        3. Si la respuesta requiere código o explicar un patrón (ej. ReAct), documéntalo claramente.
        """

        # 3. Generación usando Structured Outputs nativos de la nueva API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RespuestaMentor, # Inyección directa del contrato
                temperature=0.2, # Baja temperatura para que sea analítico, no creativo
            ),
        )
        
        # 4. Validación Final
        return RespuestaMentor.model_validate_json(response.text)

    except Exception as e:
        return f"Error en la consulta (RAG): {str(e)}"
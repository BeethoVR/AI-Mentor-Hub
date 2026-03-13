import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def consultar_mentor(pregunta):
    # 1. Configurar Acceso
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # 2. Retrieval (Búsqueda)
    # Buscamos los 5 fragmentos más relevantes en los libros
    docs = vector_db.similarity_search(pregunta, k=5)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    # 3. Generation (Gemini)
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = f"""
    Eres el AI-Mentor Hub. Tu conocimiento proviene de los libros de Chip Huyen y Michael Lanham.
    
    CONTEXTO DE LOS LIBROS:
    {contexto}
    
    PREGUNTA DEL ESTUDIANTE:
    {pregunta}
    
    INSTRUCCIÓN: Responde de forma técnica y profesional. Si la respuesta requiere código, 
    usa Python. Cita el libro o concepto principal (ej. 'Según el patrón ReAct...').
    """
    
    response = model.generate_content(prompt)
    return response.text
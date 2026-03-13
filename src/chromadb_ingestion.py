import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def procesar_bibliografia():
    # 1. Cargar PDFs
    archivos = ["data/AI_Engineering_Huyen.pdf", "data/AI_Agents_Lanham.pdf"]
    documentos_totales = []
    
    for ruta in archivos:
        loader = PyPDFLoader(ruta)
        documentos_totales.extend(loader.load())

    # 2. Chunking (Criterio de Ingeniería)
    # Dividimos en pedazos de 1000 caracteres con solapamiento para no perder contexto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documentos_totales)

    # 3. Embeddings y Vector DB
    # Usamos el modelo de Google para que sea compatible con Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("Base de datos vectorial creada con éxito.")

if __name__ == "__main__":
    procesar_bibliografia()
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document

# NUEVA IMPORTACIÓN PARA EMBEDDINGS LOCALES
from langchain_huggingface import HuggingFaceEmbeddings

def setup_vector_db():
    persist_file = "data/processed_docs.json"
    
    # CAMBIO RADICAL: Modelo local 100% gratuito y sin cuotas
    print("[INFO] Cargando modelo de Embeddings local (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(persist_file):
        print("\n[INFO] Cargando fragmentos desde JSON (DocArray)...")
        with open(persist_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks = [Document(page_content=d['page_content'], metadata=d['metadata']) for d in data]
        return DocArrayInMemorySearch.from_documents(chunks, embeddings)

    print("[1/3] Cargando PDFs de Huyen y Lanham...")
    documentos = []
    if not os.path.exists("data"): 
        os.makedirs("data")

    for archivo in os.listdir("data"):
        if archivo.endswith(".pdf"):
            ruta = os.path.join("data", archivo)
            print(f"  -> Leyendo: {archivo}")
            loader = PyPDFLoader(ruta)
            documentos.extend(loader.load())

    if not documentos:
        raise ValueError("No se encontraron PDFs en la carpeta 'data'.")

    print("[2/3] Fragmentando texto...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documentos)

    print(f"[3/3] Generando vectores LOCALMENTE (0 cuotas)... Total: {len(chunks)}")
    vector_db = DocArrayInMemorySearch.from_documents(chunks, embeddings)

    docs_to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]
    with open(persist_file, "w", encoding="utf-8") as f:
        json.dump(docs_to_save, f, ensure_ascii=False, indent=2)

    print(f"[ÉXITO] Base de datos vectorial lista y guardada en {persist_file}")
    return vector_db
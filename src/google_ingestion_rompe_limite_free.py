import os
import json
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


def setup_vector_db():
    persist_file = "data/processed_docs.json"
    
    ## LangChain sigue usando esta clase para los embeddings, el modelo 004 es el más estable
    #embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    # Usamos el modelo exacto que tu API Key reportó como disponible
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 1. Carga Inteligente desde JSON (si existe)
    if os.path.exists(persist_file):
        print("\n[INFO] Cargando fragmentos desde JSON (DocArray)...")
        with open(persist_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Reconstruimos los objetos Document de LangChain
            chunks = [Document(page_content=d['page_content'], metadata=d['metadata']) for d in data]
        return DocArrayInMemorySearch.from_documents(chunks, embeddings)

    # 2. Ingesta desde cero
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
        raise ValueError("No se encontraron PDFs en la carpeta 'data'. Revisa los nombres.")

    print("[2/3] Fragmentando texto...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documentos)
    
    #print("[3/3] Generando embeddings y guardando en JSON...")
    #vector_db = DocArrayInMemorySearch.from_documents(chunks, embeddings)

    # ... (arriba queda igual, donde creas los chunks) ...
    print(f"[3/3] Generando embeddings en lotes para evitar error 429... (Total chunks: {len(chunks)})")
    
    vector_db = None
    batch_size = 90 # Procesamos de a 90 para no pasarnos del límite de 100
    
    for i in range(0, len(chunks), batch_size):
        lote = chunks[i : i + batch_size]
        print(f" -> Procesando lote {i//batch_size + 1} de {(len(chunks)//batch_size) + 1}...")
        
        if vector_db is None:
            vector_db = DocArrayInMemorySearch.from_documents(lote, embeddings)
        else:
            vector_db.add_documents(lote)
            
        # Si aún quedan lotes por procesar, hacemos una pausa estratégica
        if i + batch_size < len(chunks):
            print("    [Pausa de 60 segundos por cuota de API... tómate un café ☕]")
            time.sleep(60)

    # ... (guardar en JSON queda igual) ...
    # Persistencia manual en JSON para evitar errores de Pickling y compatibilidad
    docs_to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]
    with open(persist_file, "w", encoding="utf-8") as f:
        json.dump(docs_to_save, f, ensure_ascii=False, indent=2)

    print(f"[ÉXITO] Base de datos vectorial lista y guardada en {persist_file}")
    return vector_db
    
    
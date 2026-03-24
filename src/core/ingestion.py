import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import setup_logging

logger = setup_logging(__name__)

def setup_vector_db():
    persist_file = "data/processed_docs.json"
    
    logger.info("Cargando modelo de Embeddings local (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    existing_chunks = []
    processed_files = set()

    # 1. Cargar la memoria existente y registrar qué archivos ya conocemos
    if os.path.exists(persist_file):
        logger.info("Leyendo base de datos existente...")
        try:
            with open(persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for d in data:
                    doc = Document(page_content=d['page_content'], metadata=d['metadata'])
                    existing_chunks.append(doc)
                    
                    # PyPDFLoader guarda la ruta original en 'source' (ej. data/libro.pdf)
                    if 'source' in d['metadata']:
                        nombre_archivo = os.path.basename(d['metadata']['source'])
                        processed_files.add(nombre_archivo)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON corrupto en {persist_file}, se recreará: {e}")
            existing_chunks = []
        except Exception as e:
            logger.warning(f"Error al cargar JSON: {e}")
            existing_chunks = []
                    
    # 2. Revisar la carpeta física
    if not os.path.exists("data"): 
        os.makedirs("data")
    
    todos_los_pdfs = [f for f in os.listdir("data") if f.endswith(".pdf")]
    
    # 3. La Magia Incremental: Filtrar SOLO los nuevos
    nuevos_pdfs = [f for f in todos_los_pdfs if f not in processed_files]

    # Si no hay archivos nuevos, devolvemos la BD armada con lo que ya teníamos
    if not nuevos_pdfs:
        if existing_chunks:
            logger.info("No hay archivos nuevos. Cargando BD desde memoria instantáneamente.")
            return DocArrayInMemorySearch.from_documents(existing_chunks, embeddings)
        else:
            logger.info("No hay PDFs en la carpeta 'data'.")
            return None

    # 4. Procesar ÚNICAMENTE los archivos nuevos
    logger.info(f"Indexando {len(nuevos_pdfs)} archivo(s) NUEVO(S)...")
    nuevos_documentos = []
    for archivo in nuevos_pdfs:
        ruta = os.path.join("data", archivo)
        logger.info(f"Leyendo nuevo: {archivo}")
        try:
            loader = PyPDFLoader(ruta)
            nuevos_documentos.extend(loader.load())
        except Exception as e:
            logger.error(f"No se pudo cargar {archivo}: {e}")
            continue

    logger.info("Fragmentando texto nuevo...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    nuevos_chunks = text_splitter.split_documents(nuevos_documentos)

    # 5. Unir la memoria vieja con la memoria nueva
    todos_los_chunks = existing_chunks + nuevos_chunks

    logger.info(f"Generando vectores actualizados... Total de fragmentos: {len(todos_los_chunks)}")
    vector_db = DocArrayInMemorySearch.from_documents(todos_los_chunks, embeddings)

    # 6. Sobrescribir el JSON con la información completa
    docs_to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in todos_los_chunks]
    with open(persist_file, "w", encoding="utf-8") as f:
        json.dump(docs_to_save, f, ensure_ascii=False, indent=2)

    logger.info(f"Base de datos actualizada y persistida en {persist_file}")
    return vector_db
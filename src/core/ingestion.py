import os
import json
import time
from functools import lru_cache
from typing import List, Optional, Set
from langdetect import detect
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import setup_logging, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH, DATA_DIR

logger = setup_logging(__name__)

@lru_cache(maxsize=1)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Loads and caches the HuggingFace embeddings model.
    
    Returns:
        HuggingFaceEmbeddings: The loaded embeddings model.
    """
    try:
        logger.info(f"Cargando modelo de embeddings '{EMBEDDING_MODEL}' (cached)...")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Error al cargar el modelo de embeddings: {e}")
        raise

def detectar_idioma_documento(texto_muestra: str) -> str:
    """Detects the document's language and returns 'Spanish' or 'English'."""
    try:
        iso_code = detect(texto_muestra)
        return "Spanish" if iso_code == 'es' else "English"
    except Exception:
        return "Spanish"  # Default to Spanish for safety

def setup_vector_db() -> Optional[DocArrayInMemorySearch]:
    """
    Sets up the vector database incrementally.
    
    This function:
    1. Loads existing chunks from a JSON file.
    2. Identifies new PDF files in the data directory.
    3. Loads and chunks new PDFs.
    4. Merges old and new chunks.
    5. Re-generates the vector index and persists it back to JSON.
    
    Returns:
        Optional[DocArrayInMemorySearch]: The updated vector database, or None if no documents exist.
    """
    persist_file = VECTOR_DB_PATH
    
    try:
        embeddings = get_embeddings_model()
    except Exception as e:
        logger.error(f"No se pudo inicializar el modelo de embeddings: {e}")
        return None

    existing_chunks: List[Document] = []
    processed_files: Set[str] = set()

    # 1. Cargar la memoria existente y registrar qué archivos ya conocemos
    if os.path.exists(persist_file):
        logger.info("Leyendo base de datos existente...")
        try:
            with open(persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for d in data:
                    doc = Document(page_content=d['page_content'], metadata=d['metadata'])
                    existing_chunks.append(doc)
                    
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
    try:
        if not os.path.exists(DATA_DIR): 
            os.makedirs(DATA_DIR)
        
        todos_los_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    except Exception as e:
        logger.error(f"Error al acceder al directorio de datos: {e}")
        return None
    
    # 3. La Magia Incremental: Filtrar SOLO los nuevos
    nuevos_pdfs = [f for f in todos_los_pdfs if f not in processed_files]

    # Si no hay archivos nuevos, devolvemos la BD armada con lo que ya teníamos
    if not nuevos_pdfs:
        if existing_chunks:
            logger.info("No hay archivos nuevos. Cargando BD desde memoria instantáneamente.")
            try:
                return DocArrayInMemorySearch.from_documents(existing_chunks, embeddings)
            except Exception as e:
                logger.error(f"Error al crear vector db desde chunks existentes: {e}")
                return None
        else:
            logger.info("No hay PDFs en la carpeta 'data'.")
            return None

    # 4. Procesar ÚNICAMENTE los archivos nuevos
    logger.info(f"Indexando {len(nuevos_pdfs)} archivo(s) NUEVO(S)...")
    nuevos_documentos = []
    for archivo in nuevos_pdfs:
        ruta = os.path.join(DATA_DIR, archivo)
        logger.info(f"Leyendo nuevo: {archivo}")
        try:
            loader = PyPDFLoader(ruta)
            nuevos_documentos.extend(loader.load())
        except Exception as e:
            logger.error(f"No se pudo cargar {archivo}: {e}")
            continue

    if not nuevos_documentos:
        logger.warning("No se pudieron cargar nuevos documentos.")
        if existing_chunks:
            return DocArrayInMemorySearch.from_documents(existing_chunks, embeddings)
        return None

    try:
        # Detectar idioma predominante de la nueva carga
        texto_muestra = " ".join([doc.page_content for doc in nuevos_documentos[:3]])
        idioma_detectado = detectar_idioma_documento(texto_muestra)
        logger.info(f"Idioma detectado en nuevos documentos: {idioma_detectado}")

        logger.info("Fragmentando texto nuevo...")
        text_splitter = RecursiveCharacterTextSplitter(
                                                        chunk_size=CHUNK_SIZE,
                                                        chunk_overlap=CHUNK_OVERLAP,
                                                        separators=["\n\n", "\n", ". ", " ", ""],
                                                        add_start_index=True
                                                      )
        nuevos_chunks = text_splitter.split_documents(nuevos_documentos)

        # 5. Unir la memoria vieja con la memoria nueva
        todos_los_chunks = existing_chunks + nuevos_chunks
        
        # Guardar metadatos del proyecto
        metadata = {
            "idioma_rag": idioma_detectado,
            "total_fragmentos": len(todos_los_chunks),
            "ultima_actualizacion": str(time.time()) if 'time' in globals() else str(os.path.getmtime(DATA_DIR))
        }
        ruta_metadata = os.path.join(DATA_DIR, "project_metadata.json")
        with open(ruta_metadata, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Generando vectores actualizados... Total de fragmentos: {len(todos_los_chunks)}")
        vector_db = DocArrayInMemorySearch.from_documents(todos_los_chunks, embeddings)

        # 6. Sobrescribir el JSON con la información completa
        docs_to_save = [{"page_content": d.page_content, "metadata": d.metadata} for d in todos_los_chunks]
        with open(persist_file, "w", encoding="utf-8") as f:
            json.dump(docs_to_save, f, ensure_ascii=False)

        logger.info(f"Base de datos actualizada y persistida en {persist_file}")
        return vector_db
    except Exception as e:
        logger.error(f"Error durante el proceso de indexación: {e}")
        if existing_chunks:
            logger.info("Retornando base de datos con chunks previos debido al error.")
            return DocArrayInMemorySearch.from_documents(existing_chunks, embeddings)
        return None
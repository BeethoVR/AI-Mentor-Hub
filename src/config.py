import os
import logging
import sys

# Rutas del sistema
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "processed_docs.json")

# Modelo de Google (único para todas las operaciones)
MODELO_AGENTE = "gemini-3.1-flash-lite-preview"

# Parámetros de Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Parámetros de chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Parámetros de retrieval
RETRIEVAL_K = 4

# Temperaturas por tarea
LLM_TEMP_RAG = 0.2           # Para respuestas del Mentor
LLM_TEMP_GUARDRAILS = 0.3    # Para validación de preguntas
LLM_TEMP_AGENTE = 0.2        # Para el agente investigador
LLM_TEMP_TITULO = 0.3        # Para generar títulos dinámicos

# Rate limiting
RATE_LIMIT_MAX = 15          # Máximo requests por ventana
RATE_LIMIT_WINDOW = 60       # Ventana de tiempo en segundos

# Cache
MAX_QUERY_CACHE_SIZE = 100   # Máximo preguntas cacheadas

# Sanitización
MAX_QUERY_LENGTH = 2000      # Longitud máxima de consulta

# Configuración de Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(name: str = "ai-mentor-hub") -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
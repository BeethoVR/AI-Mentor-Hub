import os
import logging
import sys

# Rutas del sistema
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Modelos de Google
MODELO_RAPIDO = "gemini-2.5-flash"              # Ideal para Guardrails y RAG
MODELO_AGENTE = "gemini-3.1-flash-lite-preview" # Ideal para LangGraph por sus cuotas altas

# Parámetros globales
TEMPERATURA_ESTRICTA = 0.0

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
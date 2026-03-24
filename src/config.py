import os

# Rutas del sistema
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Modelos de Google
MODELO_RAPIDO = "gemini-2.5-flash"              # Ideal para Guardrails y RAG
MODELO_AGENTE = "gemini-3.1-flash-lite-preview" # Ideal para LangGraph por sus cuotas altas

# Parámetros globales
TEMPERATURA_ESTRICTA = 0.0
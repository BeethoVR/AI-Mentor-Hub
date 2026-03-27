import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.agents import app_grafo, ejecutar_grafo_multiagente

def test_agente_compilacion():
    """Test that the LangGraph agent is compiled correctly."""
    assert app_grafo is not None

def test_ejecutar_grafo_multiagente_mocker(mocker):
    """Test the main entry point for the multi-agent graph with mocks."""
    # Mock the graph's invoke method
    mock_invoke = mocker.patch("core.agents.app_grafo.invoke")
    mock_invoke.return_value = {"borrador": "Respuesta de prueba"}
    
    mock_vector_db = MagicMock()
    
    pregunta = "¿Qué es IA?"
    respuesta = ejecutar_grafo_multiagente(pregunta, mock_vector_db, thread_id="test_thread")
    
    assert respuesta == "Respuesta de prueba"
    mock_invoke.assert_called_once()

# Solo ejecutar si se llama directamente (para pruebas manuales)
if __name__ == "__main__":
    print("🤖 Iniciando el Agente Investigador (Motor: LangGraph)...")
    # Nota: Requiere GOOGLE_API_KEY configurada para ejecución manual real
    from langchain_community.vectorstores import DocArrayInMemorySearch
    from langchain_huggingface import HuggingFaceEmbeddings
    from config import EMBEDDING_MODEL
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = DocArrayInMemorySearch.from_texts(["La IA es inteligencia artificial."], embeddings)
    
    pregunta = "¿Qué es la IA?"
    print(f"\n👤 Pregunta: {pregunta}\n")
    
    try:
        respuesta = ejecutar_grafo_multiagente(pregunta, vector_db, thread_id="manual_test")
        print(f"\n✅ Respuesta Final: {respuesta}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

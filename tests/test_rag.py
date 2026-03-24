import pytest
import os
from unittest.mock import MagicMock, patch

os.environ["GOOGLE_API_KEY"] = "test-api-key"

from core.rag_base import consultar_mentor
from contracts.schemas import RespuestaMentor

def test_consultar_mentor_exitoso(mocker):
    # 1. Mock de la Base de Datos Vectorial
    mock_vector_db = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Texto simulado del libro de Michael Lanham sobre Agentes."
    mock_vector_db.similarity_search.return_value = [mock_doc]

    # 2. Mock del Cliente de Gemini
    mock_client_class = mocker.patch('core.rag_base.genai.Client')
    mock_client_instance = mock_client_class.return_value
    
    mock_response = MagicMock()
    mock_response.text = """
    {
        "tema": "Agentes",
        "explicacion_tecnica": "Explicación simulada",
        "codigo_ejemplo": null,
        "referencias": [{"libro": "Lanham", "capitulo": "1", "concepto_clave": "Test"}],
        "sugerencia_estudio": "Sigue así"
    }
    """
    mock_client_instance.models.generate_content.return_value = mock_response

    # 3. Ejecutamos la función
    resultado = consultar_mentor(mock_vector_db, "¿Qué es un agente?")

    # 4. Validamos que el resultado sea un objeto Pydantic correcto
    assert isinstance(resultado, RespuestaMentor)
    assert resultado.tema == "Agentes"
    mock_vector_db.similarity_search.assert_called_once_with("¿Qué es un agente?", k=4)


def test_consultar_mentor_sin_api_key(mocker):
    """Test que verifica el manejo cuando GOOGLE_API_KEY no está configurada."""
    # Guardar el valor original
    original_key = os.environ.get("GOOGLE_API_KEY")
    
    try:
        # Eliminar la variable de entorno
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        # Importar y probar - debe lanzar error antes de hacer任何事
        from core import rag_base
        # Recargar el módulo para que tome el nuevo entorno
        import importlib
        importlib.reload(rag_base)
        
        with pytest.raises(Exception) as exc_info:
            # Create a mock vector_db
            mock_vec = MagicMock()
            mock_vec.similarity_search.return_value = []
            rag_base.consultar_mentor(mock_vec, "test question")
        
        assert "GOOGLE_API_KEY" in str(exc_info.value)
    finally:
        # Restaurar el valor original
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key
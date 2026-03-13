import pytest
from unittest.mock import MagicMock

# CAMBIO AQUÍ: Quitamos el "src." para que coincida con el entorno real
from rag_base import consultar_mentor
from schemas import RespuestaMentor

def test_consultar_mentor_exitoso(mocker):
    # 1. Mock de la Base de Datos Vectorial
    mock_vector_db = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Texto simulado del libro de Michael Lanham sobre Agentes."
    mock_vector_db.similarity_search.return_value = [mock_doc]

    # 2. Mock del Cliente de Gemini (OJO AL CAMBIO DEL MOCK PATH)
    # Como quitamos "src.", también debemos quitarlo de la ruta del patch
    mock_client_class = mocker.patch('rag_base.genai.Client')
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
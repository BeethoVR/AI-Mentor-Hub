import pytest
from schemas import RespuestaMentor, ReferenciaBibliografica

def test_respuesta_mentor_valida():
    # Simulamos un JSON perfecto que devolvería Gemini
    datos_simulados = {
        "tema": "Patrón ReAct",
        "explicacion_tecnica": "Es un patrón que combina razonamiento y actuación.",
        "codigo_ejemplo": "def react_agent(): pass",
        "referencias": [
            {
                "libro": "AI Engineering de Chip Huyen",
                "capitulo": "Capítulo 4",
                "concepto_clave": "Agentes y Herramientas"
            }
        ],
        "sugerencia_estudio": "Revisar el capítulo 4."
    }
    
    # Si Pydantic falla, esta línea lanzará un error y el test fallará
    respuesta = RespuestaMentor(**datos_simulados)
    
    assert respuesta.tema == "Patrón ReAct"
    assert len(respuesta.referencias) == 1
    assert respuesta.referencias[0].libro == "AI Engineering de Chip Huyen"

def test_respuesta_mentor_invalida():
    # Simulamos una respuesta a la que le falta un campo obligatorio (ej. sugerencia_estudio)
    datos_malos = {
        "tema": "Embeddings",
        "explicacion_tecnica": "Son vectores de números.",
        "codigo_ejemplo": None,
        "referencias": []
    }
    
    # Verificamos que Pydantic efectivamente levante un error de validación
    with pytest.raises(ValueError):
        RespuestaMentor(**datos_malos)
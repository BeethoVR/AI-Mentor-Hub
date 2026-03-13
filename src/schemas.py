from pydantic import BaseModel, Field
from typing import List, Optional

class ReferenciaBibliografica(BaseModel):
    libro: str = Field(description="Nombre del libro (Huyen o Lanham)")
    capitulo: str = Field(description="Capítulo o sección específica")
    concepto_clave: str = Field(description="Concepto técnico principal")

class RespuestaMentor(BaseModel):
    tema: str = Field(description="Tema central de la consulta")
    explicacion_tecnica: str = Field(description="Respuesta detallada basada en el contexto")
    codigo_ejemplo: Optional[str] = Field(description="Fragmento de código Python o seudocódigo si aplica")
    referencias: List[ReferenciaBibliografica] = Field(description="Fuentes exactas de los libros")
    sugerencia_estudio: str = Field(description="Recomendación para profundizar")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv() 


from src.core.agents import inicializar_agente_investigador

print("🤖 Iniciando el Agente Investigador (Motor: LangGraph)...")
agente = inicializar_agente_investigador()

pregunta = "¿Qué clima hace hoy en Aguascalientes, México?"

print(f"\n👤 Pregunta: {pregunta}\n")
print("-" * 50)

# En LangGraph, enviamos un diccionario con la clave 'messages'
respuesta = agente.invoke({"messages": [("user", pregunta)]})

print("-" * 50)
# El resultado es un historial de mensajes. El último es la respuesta del LLM.
print(f"\n✅ Respuesta Final: {respuesta}")
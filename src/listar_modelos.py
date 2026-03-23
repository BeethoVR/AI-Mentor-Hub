import os
from google import genai
from dotenv import load_dotenv

# 1. Cargamos tu variable de entorno
load_dotenv()

# 2. El nuevo estándar: Creamos un Cliente
cliente = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

print("\n\n🔍 Consultando la API de Google AI Studio (Nuevo SDK)...\n\n")

# 3. Usamos el cliente para listar los modelos
for modelo in cliente.models.list():
    # Filtramos para ver solo la familia de modelos Gemini
    #if "3.1" in modelo.name.lower():
    print(f"🔹 ID del Modelo: {modelo.name}")
    print(f"   Nombre: {modelo.display_name}")
    print(f"   Descripción: {modelo.description}")
    print("-" * 50)
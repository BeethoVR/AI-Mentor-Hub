import os
from dotenv import load_dotenv
from google import genai

# Haciendolo con una Funcion
def GetGoogleModels():
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        modelos = client.models.list()
        return modelos
    except Exception as e:
        print(f"Error al obtener modelos: {str(e)}")
        return None
    

#modelos = GetGoogleModels()
#for model in modelos:
#    print(f"- {model.name}")

# Cargar tu .env para leer la GOOGLE_API_KEY
load_dotenv()

client = genai.Client()

print("Modelos disponibles para tu API Key:\n")
# Iterar sobre la lista de modelos y mostrar su nombre técnico
for model in client.models.list():
    print(f"- {model.name}")
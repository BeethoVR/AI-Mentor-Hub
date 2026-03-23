import os
from dotenv import load_dotenv
from src.ingestion_01 import setup_vector_db
from rag_base import consultar_mentor
from google import genai

def main():
    # 1. Cargar variables de entorno
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("[ERROR CRÍTICO] No se encontró GOOGLE_API_KEY en el archivo .env")
        return

    print("\nIniciando motores del AI-Mentor Hub...")
    try:
        

        # 2. Levantar la base de datos vectorial
        vector_db = setup_vector_db()
    except Exception as e:
        print(f"\n[Fallo en la Ingesta] -> {e}")
        return

    print("\n" + "="*50)
    print("      AI-MENTOR HUB (NUEVO SDK GEMINI)")
    print("="*50)
    print("Listo para consultas sobre Ingeniería de IA y Agentes.")
    print("Escribe 'salir' para terminar.\n")

    # 3. Ciclo de interacción
    while True:
        pregunta = input(">> Pregunta: ")
        if pregunta.lower() in ['salir', 'exit', 'quit']:
            print("Cerrando sesión. ¡Éxito en tu proyecto!")
            break
        
        print("\n[Buscando en bibliografía y generando respuesta...]")
        respuesta = consultar_mentor(vector_db, pregunta)
        
        # Parseo de la respuesta estructurada
        if isinstance(respuesta, str):
            print(f"\n[ERROR]: {respuesta}")
        else:
            print(f"\n--- TEMA: {respuesta.tema.upper()} ---")
            print(f"\n{respuesta.explicacion_tecnica}")
            
            if respuesta.codigo_ejemplo:
                print(f"\n[CÓDIGO SUGERIDO]\n{respuesta.codigo_ejemplo}")
            
            print("\n[FUENTES]")
            for ref in respuesta.referencias:
                print(f"- {ref.libro} (Cap: {ref.capitulo}): {ref.concepto_clave}")
                
            print(f"\n[SUGERENCIA] {respuesta.sugerencia_estudio}")
            print("-" * 50)

if __name__ == "__main__":
    main()
import os
from typing import List, Tuple
import logging

from config import DATA_DIR, setup_logging

logger = setup_logging(__name__)

def save_uploaded_pdfs(valid_files: List) -> Tuple[List, List[str]]:
    """
    Saves validated PDF files to the data directory.
    Checks for duplicates before saving.
    
    Args:
        valid_files: A list of validated Streamlit UploadedFile objects.
        
    Returns:
        A tuple containing (list of newly saved files, list of names of files that already existed).
    """
    new_files = []
    old_files = []
    
    try:
        # Aseguramos que la carpeta data/ exista
        os.makedirs(DATA_DIR, exist_ok=True)
        
        for f in valid_files:
            file_route = os.path.join(DATA_DIR, f.name)
            
            # Comprobamos si el archivo ya existe físicamente en la carpeta
            if os.path.exists(file_route):
                old_files.append(f.name)
                logger.info(f"Archivo ignorado (ya existe): {f.name}")
            else:
                # Si no existe, lo guardamos en el disco
                with open(file_route, "wb") as f_uploaded:
                    f_uploaded.write(f.getbuffer())
                new_files.append(f)
                logger.info(f"Nuevo archivo guardado: {f.name}")
                
        return new_files, old_files
        
    except Exception as e:
        logger.error(f"Error crítico al guardar archivos: {e}")
        return [], [f"Error del sistema: {str(e)}"]

def get_current_pdf_names() -> List[str]:
    """Returns a list of PDF filenames currently in the data directory."""
    if os.path.exists(DATA_DIR):
        return [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    return []

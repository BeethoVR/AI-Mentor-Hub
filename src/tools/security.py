import os
from typing import List, Tuple
from config import setup_logging

logger = setup_logging(__name__)

MAX_FILE_SIZE_MB = 36
MAX_FILES_COUNT = 5


def validate_file(file) -> Tuple[bool, str]:
    """
    Validates a single uploaded file for security.
    
    Args:
        file: The uploaded file object (Streamlit).
        
    Returns:
        (is_valid, error_message)
    """
    try:
        if file is None:
            return False, "Archivo no proporcionado"
        
        file_size_mb = file.getbuffer().nbytes / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"Archivo demasiado grande: {file.name} ({file_size_mb:.2f}MB)")
            return False, f"El archivo excede el tamaño máximo de {MAX_FILE_SIZE_MB}MB"
        
        file_name = file.name
        
        if not file_name.lower().endswith('.pdf'):
            return False, "Solo se permiten archivos PDF"
        
        safe_name = os.path.basename(file_name)
        if safe_name != file_name:
            logger.warning(f"Posible intento de path traversal: {file_name}")
            return False, "Nombre de archivo inválido (intento de path traversal)"
        
        if not safe_name or safe_name.startswith('.'):
            return False, "Nombre de archivo inválido"
        
        header = file.getvalue()[:5]
        if not header.startswith(b'%PDF-'):
            logger.warning(f"Header de PDF inválido para: {file_name}")
            return False, "El archivo no es un PDF válido"
        
        return True, ""
    except Exception as e:
        logger.error(f"Error al validar archivo {file.name}: {e}")
        return False, f"Error interno al validar el archivo: {str(e)}"


def validate_uploaded_files(files: List) -> Tuple[List, List[str]]:
    """
    Validates a list of uploaded files and returns valid files and errors.
    
    Args:
        files: List of uploaded file objects.
        
    Returns:
        (valid_files, error_messages)
    """
    errors = []
    
    try:
        if len(files) > MAX_FILES_COUNT:
            msg = f"Se permiten máximo {MAX_FILES_COUNT} archivos por carga. Se procesarán solo los primeros {MAX_FILES_COUNT}."
            logger.warning(msg)
            errors.append(msg)
            files = files[:MAX_FILES_COUNT]
        
        valid_files = []
        for f in files:
            is_valid, error = validate_file(f)
            if is_valid:
                valid_files.append(f)
            else:
                errors.append(f"{f.name}: {error}")
        
        return valid_files, errors
    except Exception as e:
        logger.error(f"Error en validate_uploaded_files: {e}")
        return [], [f"Error general en la validación: {str(e)}"]
import os
from typing import List, Tuple

# Streamlit's UploadedFile type - using duck typing, no import needed
# The file object has .name, .getbuffer(), .getvalue() attributes

MAX_FILE_SIZE_MB = 36
MAX_FILES_COUNT = 5


def validate_file(file) -> Tuple[bool, str]:
    """
    Validates a single uploaded file for security.
    
    Returns:
        (is_valid, error_message)
    """
    if file is None:
        return False, "Archivo no proporcionado"
    
    file_size_mb = file.getbuffer().nbytes / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"El archivo excede el tamaño máximo de {MAX_FILE_SIZE_MB}MB"
    
    file_name = file.name
    
    if not file_name.lower().endswith('.pdf'):
        return False, "Solo se permiten archivos PDF"
    
    safe_name = os.path.basename(file_name)
    if safe_name != file_name:
        return False, "Nombre de archivo inválido (intento de path traversal)"
    
    if not safe_name or safe_name.startswith('.'):
        return False, "Nombre de archivo inválido"
    
    header = file.getvalue()[:5]
    if not header.startswith(b'%PDF-'):
        return False, "El archivo no es un PDF válido"
    
    return True, ""


def validate_uploaded_files(files) -> Tuple[list, list]:
    """
    Validates a list of uploaded files and returns valid files and errors.
    
    Returns:
        (valid_files, error_messages)
    """
    errors = []
    
    if len(files) > MAX_FILES_COUNT:
        errors.append(f"Se permiten máximo {MAX_FILES_COUNT} archivos por carga")
        files = files[:MAX_FILES_COUNT]
    
    valid_files = []
    for f in files:
        is_valid, error = validate_file(f)
        if is_valid:
            valid_files.append(f)
        else:
            errors.append(f"{f.name}: {error}")
    
    return valid_files, errors
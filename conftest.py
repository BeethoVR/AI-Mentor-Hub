import sys
import os

# Esto inyecta dinámicamente la carpeta 'src' en el Path de Python 
# solo durante la ejecución de las pruebas.
carpeta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, carpeta_src)
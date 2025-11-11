# main.py
import os
from src.preprocesamiento import pipeline as preprocesar_datos
from src.modelado import ejecutar_modelado
# otras funciones necesarias 

# --- 1. Definir Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.join(BASE_DIR, 'data', 'data.csv')
RUTA_MODELOS = os.path.join(BASE_DIR, 'models')

# --- 2. Ejecutar Preprocesamiento ---
print("--- Iniciando Preprocesamiento ---")
X_train, X_test, y_train, y_test = preprocesar_datos(
    ruta_csv=RUTA_DATOS, 
    ruta_salida=RUTA_MODELOS
)

print("--- Preprocesamiento Finalizado ---")

# --- 3. Ejecutar Modelado ---
print("\n--- Iniciando Modelado ---")
modelo_logistico, modelo_ridge = ejecutar_modelado(X_train, y_train, X_test, y_test)
print("--- Modelado Finalizado ---")

# --- 4. Ejecutar Validación ---
# ... proximas cosas ...
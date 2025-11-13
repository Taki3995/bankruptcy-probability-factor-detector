# main.py
import os
from src.preprocesamiento import pipeline as preprocesar_datos
from src.modelado import ejecutar_modelado
# otras funciones necesarias 

def main():
    """
    Función principal que orquesta todo el proyecto.
    """
    
    # --- 1.- Definir Rutas ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RUTA_DATOS = os.path.join(BASE_DIR, 'data', 'data.csv')
    RUTA_MODELOS = os.path.join(BASE_DIR, 'models')

    print(f"Directorio base: {BASE_DIR}")
    print(f"Ruta de datos: {RUTA_DATOS}")
    print(f"Ruta de modelos: {RUTA_MODELOS}")
    
    # --- 2.- Ejecutar Preprocesamiento ---
    print("\n" + "="*50)
    print("--- (1) Iniciando Preprocesamiento ---")
    print("="*50)
    
    X_train, X_test, y_train, y_test = preprocesar_datos(ruta_csv=RUTA_DATOS, ruta_salida_modelos=RUTA_MODELOS)

    print("\n--- Preprocesamiento Finalizado ---")

    # --- 3.- Ejecutar Modelado ---
    print("\n" + "="*50)
    print("--- (2) Iniciando Modelado ---")
    print("="*50)
    
    modelo_logistico, modelo_ridge = ejecutar_modelado(X_train, y_train, X_test, y_test, RUTA_MODELOS)  
    print("\n--- Modelado Finalizado ---")

    # --- 4.- Ejecutar Validación (Próximo paso) ---
    print("\n" + "="*50)
    print("--- (3) Validación ---")
    print("="*50)
    print("\n--- Pipeline Completo Ejecutado Exitosamente ---")


if __name__ == "__main__":
    main()
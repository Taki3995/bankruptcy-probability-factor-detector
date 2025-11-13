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
    RUTA_REPORTES = os.path.join(BASE_DIR, 'reports')

    os.makedirs(RUTA_MODELOS, exist_ok=True)
    os.makedirs(RUTA_REPORTES, exist_ok=True)

    print(f"Directorio base: {BASE_DIR}")
    print(f"Ruta de datos: {RUTA_DATOS}")
    print(f"Ruta de modelos: {RUTA_MODELOS}")
    print(f"Ruta de reportes: {RUTA_REPORTES}")
    
    # --- 2.- Ejecutar Preprocesamiento ---
    titulo1 = "Iniciando Preprocesamiento"
    print("\n" + "="*60)
    print(titulo1.center(60))
    print("="*60)
    
    X_train, X_test, y_train, y_test = preprocesar_datos(ruta_csv=RUTA_DATOS, ruta_salida_modelos=RUTA_MODELOS, ruta_salida_reportes=RUTA_REPORTES)

    print("\n--- Preprocesamiento Finalizado ---")

    # --- 3.- Ejecutar Modelado ---
    titulo2 = "Iniciando Modelado"
    print("\n" + "="*60)
    print(titulo2.center(60))
    print("="*60)
    
    modelo_logistico, modelo_ridge = ejecutar_modelado(X_train, y_train, X_test, y_test, RUTA_MODELOS, RUTA_REPORTES)  
    print("\n--- Modelado Finalizado ---")

    # --- 4.- Ejecutar Validación (Próximo paso) ---
    titulo3 = "Validación"
    print("\n" + "="*60)
    print(titulo3.center(60))
    print("="*60)
    print("\n--- Pipeline Completo Ejecutado Exitosamente ---")


if __name__ == "__main__":
    main()
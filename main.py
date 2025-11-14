import os
import joblib
from imblearn.over_sampling import SMOTE

# Importar desde Sources
from src.preprocesamiento import pipeline as preprocesar_datos
from src.modelado import ejecutar_modelado
from src.validacion import validar_modelo as ejecutar_validacion
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
    print("\n" + "="*112)
    print(titulo1.center(112))
    print("="*112)
    
    X_train, X_test, y_train, y_test = preprocesar_datos(ruta_csv=RUTA_DATOS, ruta_salida_modelos=RUTA_MODELOS, ruta_salida_reportes=RUTA_REPORTES)

    print("\n--- Preprocesamiento Finalizado ---")

    # Aplicar SMOTE
    print("\n" + "-"*60)
    texto5 = "Aplicando SMOTE para balancear el set de entrenamiento..."
    print(texto5.center(60))
    print("-"*60)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Shape original de entrenamiento: {X_train.shape}")
    print(f"Shape remuestreado de entrenamiento: {X_train_resampled.shape}")
    print("\nDistribución de clases de entrenamiento (después de SMOTE):")
    print(y_train_resampled.value_counts(normalize=True))
    print("-"*60)

    print("\nGuardando datos de entrenamiento (con SMOTE) para validación...")
    try:
        # X_train_resampled es un DataFrame de pandas, y_train_resampled es una Serie
        joblib.dump(X_train_resampled, os.path.join(RUTA_MODELOS, 'X_train_resampled.joblib'))
        joblib.dump(y_train_resampled, os.path.join(RUTA_MODELOS, 'y_train_resampled.joblib'))
        print(f"Datos guardados exitosamente en carpeta")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")

    # --- 3.- Ejecutar Modelado ---
    titulo2 = "Iniciando Modelado"
    print("\n" + "="*112)
    print(titulo2.center(112))
    print("="*112)

    # Pasar datos remuestreados al modelo
    modelo_logistico, modelo_ridge = ejecutar_modelado(
        X_train_resampled, y_train_resampled,      # <-- Datos balanceados para entrenar
        X_test, y_test,                            # <-- Datos reales (desbalanceados) para probar
        RUTA_MODELOS, 
        RUTA_REPORTES
    )
    
    print("\n--- Modelado Finalizado ---")

    # --- 4.- Ejecutar Validación (Próximo paso) ---
    titulo3 = "Iniciando Validación por Bootstrap"
    print("\n" + "="*112)
    print(titulo3.center(112))
    print("="*112)
    ejecutar_validacion(n_bootstraps=500) 
    print("\n--- Validación Finalizada ---")
    print("\n--- Pipeline Completo Ejecutado Exitosamente ---")


if __name__ == "__main__":
    main()
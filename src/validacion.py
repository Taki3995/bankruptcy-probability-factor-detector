import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import seaborn as sns

from tqdm import tqdm # Para barras de progreso
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# === Funciones de Validación === 

def validar_modelo(n_bootstraps = 1000):
    """
    Ejecuta el remuestro de Bootstrap para estimar la varianza de los coeficientes
    de los modelos MLE y Ridge.
    """

    # === 1 Definir Rutas ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    ruta_modelos = os.path.join(base_dir, 'models')
    ruta_reportes = os.path.join(base_dir, 'reports')
    os.makedirs(ruta_reportes, exist_ok = True)

    print(f"Directorio Base: {base_dir}")
    print(f"Ruta Modelos: {ruta_modelos}")

    # === 2 Cargar Datos de Entrenamiento ===
    try:
        # Cargar los datos de entrenamiento (guardados por main.py)
        X_train = joblib.load(os.path.join(ruta_modelos, 'X_train_resampled.joblib'))
        y_train = joblib.load(os.path.join(ruta_modelos, 'y_train_resampled.joblib'))

        # Cargar los nombres de las columnas (guardados por preprocesamiento.py)
        columnas = joblib.load(os.path.join(ruta_modelos, 'columnas_modelo.joblib'))

        # Cargar los modelos entrenados (guardados por modelado.py via main.py)
        modelo_mle = joblib.load(os.path.join(ruta_modelos, 'modelo_logistico_mle.joblib'))
        modelo_ridge = joblib.load(os.path.join(ruta_modelos, 'modelo_logistico_ridge.joblib'))

    except Exception as e:
        print(f"Ocurrió un error al cargar los archivos: {e}")
        return
    
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns = columnas)
    else:
        X_train.columns = columnas
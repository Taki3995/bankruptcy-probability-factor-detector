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
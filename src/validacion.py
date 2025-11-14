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
        columnas = joblib.load(os.path.join(ruta_modelos, 'columnas_finales.joblib'))

        # Cargar los modelos entrenados (guardados por modelado.py via main.py)
        modelo_mle_base = joblib.load(os.path.join(ruta_modelos, 'modelo_mle.joblib'))
        modelo_ridge = joblib.load(os.path.join(ruta_modelos, 'modelo_ridge_cv.joblib'))

    except Exception as e:
        print(f"Ocurrió un error al cargar los archivos: {e}")
        return
    
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns = columnas)
    else:
        X_train.columns = columnas
    print(f"Datos de entrenamiento cargados: {X_train.shape}")

    # === 3 Remuestreo Bootstrap ===
    modelo_mle = modelo_mle_base

    mejor_C = modelo_ridge.C_[0]
    print(f"Mejor 'C' (inverso de lambda) encontrado por CV: {mejor_C:.4f}")
    
    modelo_ridge_final = LogisticRegression(
        penalty = 'l2',           # L2 = Ridge
        C = mejor_C,              # Usar el mejor C
        solver = 'lbfgs',
        max_iter = 5000,          # Consistente con modelado.py
        random_state = 42
    )

    # === 4 Ejecutar Bucle Bootstrap ===
    print(f"\nIniciando bucle Bootstrap con {n_bootstraps} iteraciones...")
    
    # Listas para almacenar los coeficientes de cada iteración
    coefs_mle = []
    coefs_ridge = []
    
    for i in tqdm(range(n_bootstraps)):
        # 1. Crear una muestra bootstrap (muestreo con reemplazo)
        # Usamos random_state=i para reproducibilidad
        X_sample, y_sample = resample(X_train, y_train, random_state=i)
        
        # 2. Re-entrenar el modelo MLE
        modelo_mle.fit(X_sample, y_sample)
        coefs_mle.append(modelo_mle.coef_[0])
        
        # 3. Re-entrenar el modelo Ridge (con el C óptimo)
        modelo_ridge_final.fit(X_sample, y_sample)
        coefs_ridge.append(modelo_ridge_final.coef_[0])
        
    print("Bucle Bootstrap finalizado.")

    # === 5. Analizar Coeficientes y Calcular IC ===
    print("Calculando Intervalos de Confianza (IC) del 95%...")
    
    # Convertir listas a DataFrames para fácil análisis
    df_coefs_mle = pd.DataFrame(coefs_mle, columns=columnas)
    df_coefs_ridge = pd.DataFrame(coefs_ridge, columns=columnas)
    
    # Calcular los percentiles 2.5% y 97.5%
    ic_mle = df_coefs_mle.quantile([0.025, 0.975])
    ic_ridge = df_coefs_ridge.quantile([0.025, 0.975])
    
    # Crear un DataFrame de resumen
    summary = pd.DataFrame({
        'Coef_MLE_Media': df_coefs_mle.mean(),
        'Coef_MLE_IC_2.5%': ic_mle.loc[0.025],
        'Coef_MLE_IC_97.5%': ic_mle.loc[0.975],
        'Coef_Ridge_Media': df_coefs_ridge.mean(),
        'Coef_Ridge_IC_2.5%': ic_ridge.loc[0.025],
        'Coef_Ridge_IC_97.5%': ic_ridge.loc[0.025],
    })
    
    # Calcular el ANCHO del intervalo (nuestra medida de varianza)
    summary['Ancho_IC_MLE'] = summary['Coef_MLE_IC_97.5%'] - summary['Coef_MLE_IC_2.5%']
    summary['Ancho_IC_Ridge'] = summary['Coef_Ridge_IC_97.5%'] - summary['Coef_Ridge_IC_2.5%']
    
    # Ordenar por el ancho del IC de MLE (para ver los más inestables)
    summary = summary.sort_values(by='Ancho_IC_MLE', ascending=False)
    
    # Guardar reporte
    ruta_csv_summary = os.path.join(ruta_reportes, 'bootstrap_intervalos_confianza.csv')
    summary.to_csv(ruta_csv_summary)
    print(f"\nResumen de IC guardado en: {ruta_csv_summary}")
    
    print("\n--- Top 10 Coeficientes con Mayor Varianza (MLE) ---")
    print(summary[['Ancho_IC_MLE', 'Ancho_IC_Ridge']].head(10))

    # === 6. Visualizar Resultados ===
    print("Generando visualización de IC...")
    
    # Graficaremos los 20 coeficientes más variables del modelo MLE
    df_plot = summary.head(20).sort_values(by='Ancho_IC_MLE') # Ordenar ascendente para el gráfico
    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = np.arange(len(df_plot.index))
    
    # Graficar IC del modelo MLE (azul)
    ax.errorbar(
        x = df_plot['Coef_MLE_Media'], 
        y = y_pos - 0.15, 
        xerr = (df_plot['Coef_MLE_Media'] - df_plot['Coef_MLE_IC_2.5%'], df_plot['Coef_MLE_IC_97.5%'] - df_plot['Coef_MLE_Media']),
        fmt = 'o', label = 'MLE (Sin Regularización)', color = 'blue', capsize = 5
    )
    
    # Graficar IC del modelo Ridge (naranja)
    ax.errorbar(
        x = df_plot['Coef_Ridge_Media'], 
        y = y_pos + 0.15,
        xerr = (df_plot['Coef_Ridge_Media'] - df_plot['Coef_Ridge_IC_2.5%'], df_plot['Coef_Ridge_IC_97.5%'] - df_plot['Coef_Ridge_Media']),
        fmt = 's', label = f'Ridge (C={mejor_C:.2f})', color = 'darkorange', capsize = 5
    )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot.index)
    ax.set_xlabel('Valor del Coeficiente')
    ax.set_ylabel('Variables (Top 20 más variables en MLE)')
    ax.set_title('Comparación de Intervalos de Confianza (95%) - Bootstrap', fontsize = 16)
    ax.axvline(x = 0, color = 'grey', linestyle = '--', linewidth = 0.8)
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    ruta_grafico = os.path.join(ruta_reportes, 'comparacion_ic_bootstrap.png')
    plt.savefig(ruta_grafico)
    plt.close()
    
    print(f"Gráfico de IC guardado en: {ruta_grafico}")

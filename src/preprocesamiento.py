# PREPROCESAMIENTO DEL DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

# --- FUNCIONES AUXILIARES ---

# Carga de datos
def cargar_datos(ruta_csv):
    """
    Carga los datos desde el csv
    """
    df = pd.read_csv(ruta_csv)
    df.columns = df.columns.str.strip()  # limpiar espacios
    print(f"Shape: {df.shape}")
    print("Primeras filas:")
    print(df.head())
    return df

# Exploración inicial
def exploracion_inicial(df, ruta_salida_reportes):
    """
    Ve la distribución de las empresas quebradas vs las no quebradas
    """
    print("\nDistribución de la variable objetivo:")
    print(df['Bankrupt?'].value_counts(normalize=True))

    sns.countplot(x='Bankrupt?', data=df)
    plt.title("Distribución de empresas quebradas vs no quebradas")
    ruta_guardado = os.path.join(ruta_salida_reportes, 'distribucion_objetivo.png')
    plt.savefig(ruta_guardado)
    plt.close() # Cierra la figura para liberar memoria
    print(f"Gráfico de distribución guardado en: {ruta_guardado}")

# Valores Faltantes
def revisar_nulos(df):
    nulos = df.isnull().mean() * 100
    print("\n% de valores nulos por columna (solo >0%):")
    print(nulos[nulos > 0].sort_values(ascending=False))

# Estadísticas básicas y outliers
def resumen_estadistico(df):
    """
    Detecta outliers por columna
    """
    print("\nResumen estadístico global:")
    print(df.describe().T)

    # Detección de outliers usando IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print("\nNúmero de outliers por columna (top 10):")
    print(outliers.sort_values(ascending=False).head(10))

# Correlaciones
def correlaciones_fuertes(df, threshold=0.95):
    """
    Ve cuan semejantes son las columnas entre sí (correlación)
    """
    corr = df.corr().abs()
    altas = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index())
    altas.columns = ['Var1','Var2','Corr']
    return altas[altas['Corr'] > threshold].sort_values(by='Corr', ascending=False)

# Winsorizador (control de outliers)
def winsorizar_df(df, limits=[0.01, 0.01]):
    """
    Técnica estadística para manejar valores atípicos en un conjunto de datos sin eliminarlos por completo,
    acotando los valores extremadamente altos al punto mas alto no conciderado atípico (y viceversa).
    """
    df_w = df.copy()
    cols_to_winsorize = [col for col in df_w.columns if col != 'Bankrupt?'] # Excluir 'Bankrupt?'
    for col in cols_to_winsorize:
        df_w[col] = winsorize(df_w[col], limits=limits)
    return df_w

# Eliminar constantes y redundantes
def limpiar_columnas(df, corr_pairs, prioritarias):
    # Eliminar columnas constantes
    constantes = [col for col in df.columns if df[col].nunique() == 1 and col != 'Bankrupt?']
    print(f"Columnas constantes eliminadas: {constantes}")

    # Ridge maneja este problema
    print("\n--- PASO DE ELIMINACIÓN POR CORRELACIÓN DESACTIVADO ---")
    
    eliminar = constantes # Ahora solo eliminamos las constantes
    
    return df.drop(columns=eliminar, errors="ignore")

# TOP 15 Variables clave
def variables_prioritarias():
    return [
        'ROA(C) before interest and depreciation before interest',
        'ROA(A) before interest and % after tax',
        'Operating Gross Margin',
        'Net Income to Total Assets',
        'Debt ratio %',
        'Net worth/Assets',
        'Liability to Equity',
        'Cash Flow to Total Assets',
        'Cash Flow to Liability',
        'Current Ratio',
        'Quick Ratio',
        'Working Capital to Total Assets',
        'Interest Coverage Ratio (Interest expense to EBIT)',
        'Operating Profit Growth Rate',
        'Gross Profit to Sales'
    ]

# Calculo VIF
def calcular_vif_selectivo(X_imputed, umbral=10.0):
    """
    Calcula VIF iterativamente solo en el set de entrenamiento imputado.
    Devuelve la lista de columnas a mantener.
    """
    variables = X_imputed.copy()
    variables_eliminadas_vif = []
    
    while True:
        vif = pd.DataFrame()
        vif["Variable"] = variables.columns
        vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
        vif = vif.sort_values("VIF", ascending=False).reset_index(drop=True)
        
        max_vif = vif.loc[0, "VIF"]
        
        if max_vif > umbral:
            variable_a_eliminar = vif.loc[0, "Variable"]
            variables = variables.drop(columns=[variable_a_eliminar])
            variables_eliminadas_vif.append(variable_a_eliminar)
        else:
            break
            
    print(f"\nSe eliminaron {len(variables_eliminadas_vif)} variables por alto VIF (>{umbral}):")
    print(variables_eliminadas_vif)
    
    # Devuelve las columnas que sobrevivieron
    return variables.columns.tolist()

# ---- GRÁFICOS ----
def graficar_boxplots(X, prioritarias, ruta_salida_reportes):
    # Filtrar para graficar solo las variables prioritarias que existen en el DataFrame X
    vars_a_graficar = [var for var in prioritarias if var in X.columns]
    
    if not vars_a_graficar:
        print("Ninguna de las variables prioritarias se encuentra en el dataset final para graficar.")
        return

    plt.figure(figsize=(15, 8))
    sns.boxplot(data=X[vars_a_graficar], orient='h')
    plt.title(f"Boxplots de las variables prioritarias (después del preprocesamiento)")
    plt.xlabel("Valor estandarizado")
    plt.ylabel("Variables")
    plt.tight_layout()
    
    ruta_guardado = os.path.join(ruta_salida_reportes, 'boxplots_prioritarias.png')
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Gráfico de boxplots guardado en: {ruta_guardado}")
    
    print("\nVariables prioritarias incluidas en el boxplot:")
    print(vars_a_graficar)

def graficar_correlacion(X, ruta_salida_reportes):
    plt.figure(figsize=(12,10))
    sns.heatmap(X.corr(), cmap='coolwarm', center=0, annot=False)
    plt.title("Correlación entre variables (Train Set, Post-Procesado)")

    ruta_guardado = os.path.join(ruta_salida_reportes, 'heatmap_correlacion.png')
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Gráfico de correlación guardado en: {ruta_guardado}")
    
# -------- FUNCIÓN PRINCIPAL --------
def pipeline(ruta_csv, ruta_salida_modelos, ruta_salida_reportes):
    """
    Pipeline completo de preprocesamiento. Lee datos, limpia, procesa y guarda los transformers (imputer y scaler)
    """
    # 1. Carga y Limpieza inicial
    df = cargar_datos(ruta_csv)
    exploracion_inicial(df, ruta_salida_reportes)
    
    df_w = winsorizar_df(df)
    corr_pairs = correlaciones_fuertes(df_w, threshold=0.95)
    prioritarias_lista = variables_prioritarias()
    df_clean = limpiar_columnas(df_w, corr_pairs, prioritarias_lista)

    # 2. Separar X e y
    X = df_clean.drop(columns=['Bankrupt?'])
    y = df_clean['Bankrupt?']

    # 3. División Train/Test (Estratificada)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Shape antes de VIF: Train {X_train.shape}, Test {X_test.shape}")

    # 4. Imputación (Ajustar en Train, transformar en ambos)
    imputer = SimpleImputer(strategy='median')
    cols_train = X_train.columns
    cols_test = X_test.columns
    
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=cols_train)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=cols_test)
    
    columnas_finales = cols_train.tolist() # Usar todas las columnas en vez de quitar con vif

    """
    # 5. Selección de VIF (Ajustar SOLO en Train)
    columnas_finales = calcular_vif_selectivo(X_train_imputed, umbral=10.0)
    
    # Filtrar ambos datasets para que tengan solo las columnas finales
    X_train_vif = X_train_imputed[columnas_finales]
    X_test_vif = X_test_imputed[columnas_finales]
    print(f"Shape después de VIF: Train {X_train_vif.shape}, Test {X_test_vif.shape}")
    """

    # 6. Escalado Robusto (Ajustar en Train, transformar en ambos)
    scaler = StandardScaler()
    # Quitar vif
    # X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_vif), columns=columnas_finales)
    # X_test_scaled = pd.DataFrame(scaler.transform(X_test_vif), columns=columnas_finales)

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=columnas_finales)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=columnas_finales)
    
    # 7. EDA final (Sobre el Train Set procesado)
    print("\nRealizando EDA sobre el conjunto de entrenamiento preprocesado...")
    graficar_boxplots(X_train_scaled, prioritarias_lista, ruta_salida_reportes)
    graficar_correlacion(X_train_scaled, ruta_salida_reportes)

    # 8. Guardar Transformers (Usando la ruta de salida)
    os.makedirs(ruta_salida_modelos, exist_ok=True) # Asegura que la carpeta 'models/' exista
    
    imputer_path = os.path.join(ruta_salida_modelos, 'imputer.joblib')
    scaler_path = os.path.join(ruta_salida_modelos, 'scaler.joblib')
    cols_path = os.path.join(ruta_salida_modelos, 'columnas_finales.joblib')
    
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(columnas_finales, cols_path)
    print(f"Objetos guardados exitosamente en {ruta_salida_modelos}")

    return X_train_scaled, X_test_scaled, y_train, y_test
# PREPROCESAMIENTO DEL DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import RobustScaler #para proteger de outliers
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize #para limitar outliers
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split


# CARGA DE DATOS

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df.columns = df.columns.str.strip()  # limpiar espacios
    print(f"Shape: {df.shape}")
    print("Primeras filas:")
    print(df.head())
    return df

# EXPLORACION INICIAL

def exploracion_inicial(df):
    print("\nColumnas disponibles:")
    print(df.columns.tolist())

    print("\nDistribución de la variable objetivo:")
    print(df['Bankrupt?'].value_counts(normalize=True))

    sns.countplot(x='Bankrupt?', data=df)
    plt.title("Distribución de empresas quebradas vs no quebradas")
    plt.show()


# DETECTAR VALORES FALTANTES

def revisar_nulos(df):
    nulos = df.isnull().mean() * 100
    print("\n% de valores nulos por columna (solo >0%):")
    print(nulos[nulos > 0].sort_values(ascending=False))


# ESTADISTICAS BASICAS Y OUTLIERS

def resumen_estadistico(df):
    print("\nResumen estadístico global:")
    print(df.describe().T)

    # Detección de outliers usando IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print("\nNúmero de outliers por columna (top 10):")
    print(outliers.sort_values(ascending=False).head(10))

# CORRELACIONES

def correlaciones_fuertes(df, threshold=0.95):
    corr = df.corr().abs()
    altas = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index())
    altas.columns = ['Var1','Var2','Corr']
    altas_filtradas = altas[altas['Corr'] > threshold].sort_values(by='Corr', ascending=False)
    return altas_filtradas

# WINSORIZACION (control de outliers)

def winsorizar_df(df, limits=[0.01, 0.01]):
    df_w = df.copy()
    for col in df_w.drop(columns=['Bankrupt?']).columns:
        df_w[col] = winsorize(df_w[col], limits=limits)
    return df_w


# Calculo VIF
def calcular_vif(X, umbral=10.0):
    variables = X.copy()
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
    
    return variables

# ELIMINAR CONSTANTES Y REDUNDANTES

def limpiar_columnas(df, corr_pairs, prioritarias):
    # Eliminar columnas constantes
    constantes = [col for col in df.columns if df[col].nunique() == 1 and col != 'Bankrupt?']
    print(f"Columnas constantes eliminadas: {constantes}")

    # Eliminar una de cada par altamente correlacionado, protegiendo las prioritarias
    redundantes = set()
    for _, row in corr_pairs.iterrows():
        var1, var2 = row['Var1'], row['Var2']
        
        # Si una es prioritaria y la otra no, eliminar la no prioritaria
        if var1 in prioritarias and var2 not in prioritarias:
            redundantes.add(var2)
        elif var2 in prioritarias and var1 not in prioritarias:
            redundantes.add(var1)
        # Si ninguna es prioritaria, eliminar la segunda (como antes)
        elif var1 not in prioritarias and var2 not in prioritarias:
            redundantes.add(var2)
        # Si ambas son prioritarias, no eliminar ninguna por ahora
        
    eliminar = list(set(constantes) | redundantes)
    print(f"\nColumnas eliminadas automáticamente ({len(eliminar)}):")
    print(sorted(eliminar)) # Ordenar para una salida consistente

    return df.drop(columns=eliminar, errors="ignore")


# TOP 15 VARIABLES CLAVE 

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

# PIPELINE DE PREPROCESAMIENTO

def preprocesar_dataset(df):
     # 1. Winsorización
    df_w = winsorizar_df(df)

    # 2. Reportar y eliminar redundantes
    corr_pairs = correlaciones_fuertes(df_w)
    df_clean = limpiar_columnas(df_w, corr_pairs)

    # 3. Imputación
    imputer = SimpleImputer(strategy='median')
    X = df_clean.drop(columns=['Bankrupt?'])
    y = df_clean['Bankrupt?']
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 4. Escalado robusto
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    return X_scaled, y, imputer, scaler

def graficar_boxplots(X, prioritarias):
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
    plt.show()
    
    print("\nVariables prioritarias incluidas en el boxplot:")
    print(vars_a_graficar)

def graficar_correlacion(X):
    plt.figure(figsize=(12,10))
    sns.heatmap(X.corr(), cmap='coolwarm', center=0, annot=False)
    plt.title("Correlación entre variables preprocesadas")
    plt.show()

# PIPELINE COMPLETO

def pipeline(ruta_csv):
    df = cargar_datos(ruta_csv)
    # Limpieza
    df_w = winsorizar_df(df)
    corr_pairs = correlaciones_fuertes(df_w, threshold=0.95)
    prioritarias_lista = variables_prioritarias()
    df_clean = limpiar_columnas(df_w, corr_pairs, prioritarias_lista)

    # Dividir antes de imputar y escalar
    X_pre_vif = df_clean.drop(columns=['Bankrupt?'])
    y = df_clean['Bankrupt?']

    X_post_vif = calcular_vif(X_pre_vif, umbral=10.0)

    columnas_finales = ['Bankrupt?'] + X_post_vif.columns.tolist()
    df_final = df_clean[columnas_finales]

    X = df_final.drop(columns=['Bankrupt?'])
    y = df_final['Bankrupt?']

    # División estratificada para mantener la proporción de quiebras
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Ajustar Imputer y Scaler solo en X_train
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns) # Solo transform en test

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns) # Solo transform en test
    
    # EDA y Gráficos sobre el set de entrenamiento
    print("\nRealizando EDA sobre el conjunto de entrenamiento preprocesado...")
    graficar_boxplots(X_train_scaled, prioritarias_lista)
    graficar_correlacion(X_train_scaled)

    print("\nGuardando imputer y scaler en archivos...")
    joblib.dump(imputer, 'imputer.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Objetos guardados exitosamente.")

    return X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler



if __name__ == "__main__":
    ruta = "data.csv"
    X_train, X_test, y_train, y_test, imputer, scaler = pipeline(ruta)
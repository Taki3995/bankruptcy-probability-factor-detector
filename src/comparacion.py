import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def generar_reporte_final():
    """
    Carga todos los artefactos y genera un reporte de "herramientas"
    (datos clave) para el análisis técnico final.
    """
    # --- 1. Definir Rutas y Cargar Datos ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
    BASE_DIR = os.path.dirname(SCRIPT_DIR) 
    
    RUTA_MODELOS = os.path.join(BASE_DIR, 'models')
    RUTA_REPORTES = os.path.join(BASE_DIR, 'reports')

    print("Cargando artefactos para el reporte ...")
    try:
        # Cargar modelos
        modelo_mle = joblib.load(os.path.join(RUTA_MODELOS, 'modelo_mle.joblib'))
        modelo_ridge = joblib.load(os.path.join(RUTA_MODELOS, 'modelo_ridge_cv.joblib'))
        
        # Cargar datos de testeo
        X_test = joblib.load(os.path.join(RUTA_MODELOS, 'X_test_scaled.joblib'))
        y_test = joblib.load(os.path.join(RUTA_MODELOS, 'y_test.joblib'))
        
        # Cargar resultados del Bootstrap
        ruta_csv_bootstrap = os.path.join(RUTA_REPORTES, 'bootstrap_intervalos_confianza.csv')
        summary_bootstrap = pd.read_csv(ruta_csv_bootstrap, index_col=0)
        
    except Exception as e:
        print(f"Ocurrió un error al cargar los archivos: {e}")
        return

    # --- 2. Tabla Comparativa de Desempeño ---
    
    # Métricas Modelo MLE
    y_pred_mle = modelo_mle.predict(X_test)
    y_pred_proba_mle = modelo_mle.predict_proba(X_test)[:, 1]
    metricas_mle = {
        'AUC': roc_auc_score(y_test, y_pred_proba_mle),
        'Recall (Quiebra)': recall_score(y_test, y_pred_mle, pos_label=1),
        'Precisión (Quiebra)': precision_score(y_test, y_pred_mle, pos_label=1),
        'F1-Score (Quiebra)': f1_score(y_test, y_pred_mle, pos_label=1)
    }
    
    # Métricas Modelo Ridge
    y_pred_ridge = modelo_ridge.predict(X_test)
    y_pred_proba_ridge = modelo_ridge.predict_proba(X_test)[:, 1]
    metricas_ridge = {
        'AUC': roc_auc_score(y_test, y_pred_proba_ridge),
        'Recall (Quiebra)': recall_score(y_test, y_pred_ridge, pos_label=1),
        'Precisión (Quiebra)': precision_score(y_test, y_pred_ridge, pos_label=1),
        'F1-Score (Quiebra)': f1_score(y_test, y_pred_ridge, pos_label=1)
    }

    # Crear DataFrame de comparación
    df_metricas = pd.DataFrame({
        'Regresión Logística (MLE)': metricas_mle,
        'Regresión Ridge (CV)': metricas_ridge
    }).T # .T transpone para que los modelos sean filas
    df_metricas = df_metricas.round(4) # Redondear

    # --- 3. Métricas de Sesgo-Varianza ---
    
    # Análisis de Varianza (Ancho del IC)
    ancho_medio_mle = summary_bootstrap['Ancho_IC_MLE'].mean()
    ancho_medio_ridge = summary_bootstrap['Ancho_IC_Ridge'].mean()
    reduccion_varianza = 100 * (ancho_medio_mle - ancho_medio_ridge) / ancho_medio_mle

    # Análisis de Sesgo (Contracción de Coeficientes)
    magnitud_media_mle = summary_bootstrap['Coef_MLE_Media'].abs().mean()
    magnitud_media_ridge = summary_bootstrap['Coef_Ridge_Media'].abs().mean()
    reduccion_magnitud = 100 * (magnitud_media_mle - magnitud_media_ridge) / magnitud_media_mle

    # --- 4. Factores Clave (Modelo Ridge) ---
    
    coefs_ridge = summary_bootstrap[['Coef_Ridge_Media']].copy()
    coefs_ridge = coefs_ridge.sort_values(by='Coef_Ridge_Media', ascending=False)
    
    top_riesgo = coefs_ridge.head(5)
    top_proteccion = coefs_ridge.tail(5).sort_values(by='Coef_Ridge_Media', ascending=True)

    # --- 5. Imprimir Reporte ---
    
    print("\n\n" + "="*70)
    print("         HERRAMIENTAS DE ANÁLISIS TÉCNICO FINAL")
    print("="*70)

    print("\n## 1. Comparativa de Desempeño en Test")
    print(df_metricas.to_markdown(floatfmt=".4f"))

    print("\n" + "-"*70)
    print("\n## 2. Métricas de Sesgo-Varianza (Basado en Bootstrap)")
    
    print("\n### Varianza (Ancho Promedio del IC del 95%)")
    print(f"* Ancho promedio IC (MLE):   {ancho_medio_mle:.4f}")
    print(f"* Ancho promedio IC (Ridge): {ancho_medio_ridge:.4f}")
    print(f"* Reducción de Varianza (Ridge vs MLE): {reduccion_varianza:.2f}%")

    print("\n### Sesgo (Contracción Promedio de Coeficientes)")
    print(f"* Magnitud promedio Coef. (MLE):   {magnitud_media_mle:.4f}")
    print(f"* Magnitud promedio Coef. (Ridge): {magnitud_media_ridge:.4f}")
    print(f"* Reducción de Magnitud (Ridge vs MLE): {reduccion_magnitud:.2f}%")

    print("\n" + "-"*70)
    print("\n## 3. Factores Clave (Basado en Coeficientes Modelo Ridge)")

    print("\n**Top 5 Factores de RIESGO (Mayor Probabilidad de Quiebra):**")
    print(top_riesgo.to_markdown(floatfmt=".4f"))

    print("\n**Top 5 Factores de PROTECCIÓN (Menor Probabilidad de Quiebra):**")
    print(top_proteccion.to_markdown(floatfmt=".4f"))
    
    print("\n" + "="*70)
    print("                 Fin del Reporte de Herramientas")
    print("="*70)
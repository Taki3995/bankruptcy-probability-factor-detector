import pandas as pd
import joblib
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def generar_reporte_final():
    """
    Carga todos los artefactos (modelos, datos de test, resultados de bootstrap)
    y genera un análisis técnico comparativo final.
    Esta función es llamada por main.py al final del pipeline.
    """
    
    # === 1. Definir Rutas y Cargar Artefactos ===
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
    BASE_DIR = os.path.dirname(SCRIPT_DIR) 
    
    RUTA_MODELOS = os.path.join(BASE_DIR, 'models')
    RUTA_REPORTES = os.path.join(BASE_DIR, 'reports')

    print("Cargando artefactos para el análisis final...")
    try:
        # Cargar modelos
        modelo_mle = joblib.load(os.path.join(RUTA_MODELOS, 'modelo_mle.joblib'))
        modelo_ridge = joblib.load(os.path.join(RUTA_MODELOS, 'modelo_ridge_cv.joblib'))
        
        # Cargar datos de testeo (del prerrequisito)
        X_test = joblib.load(os.path.join(RUTA_MODELOS, 'X_test_scaled.joblib'))
        y_test = joblib.load(os.path.join(RUTA_MODELOS, 'y_test.joblib'))
        
        # Cargar resultados del Bootstrap (de validacion.py)
        ruta_csv_bootstrap = os.path.join(RUTA_REPORTES, 'bootstrap_intervalos_confianza.csv')
        summary_bootstrap = pd.read_csv(ruta_csv_bootstrap, index_col=0)
        
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo. Asegúrate de haber ejecutado main.py completo al menos una vez.")
        print(f"Detalle: {e}")
        return
    except Exception as e:
        print(f"Ocurrió un error al cargar los archivos: {e}")
        return

    # === 2. Comparativa de Desempeño (Punto 2 de tu plan) ===
    
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

    # === 3. Evaluación del Trade-off Sesgo-Varianza ===
    
    # Análisis de Varianza (Ancho del IC)
    ancho_medio_mle = summary_bootstrap['Ancho_IC_MLE'].mean()
    ancho_medio_ridge = summary_bootstrap['Ancho_IC_Ridge'].mean()
    reduccion_varianza = 100 * (ancho_medio_mle - ancho_medio_ridge) / ancho_medio_mle

    # Análisis de Sesgo (Contracción de Coeficientes)
    magnitud_media_mle = summary_bootstrap['Coef_MLE_Media'].abs().mean()
    magnitud_media_ridge = summary_bootstrap['Coef_Ridge_Media'].abs().mean()
    reduccion_magnitud = 100 * (magnitud_media_mle - magnitud_media_ridge) / magnitud_media_mle

    # === 4. Identificación de Factores Clave ===
    
    # Usaremos los coeficientes PROMEDIO del modelo RIDGE (más estables) para la interpretación
    coefs_ridge = summary_bootstrap[['Coef_Ridge_Media']].copy()
    coefs_ridge = coefs_ridge.sort_values(by='Coef_Ridge_Media', ascending=False)
    
    top_riesgo = coefs_ridge.head(5)
    top_proteccion = coefs_ridge.tail(5).sort_values(by='Coef_Ridge_Media', ascending=True)

    # === 5. Generar Reporte Final en Consola ===
    
    print("\n\n" + "="*70)
    print("      REPORTE DE COMPARACIÓN Y ANÁLISIS FINAL DEL MODELO")
    print("="*70)

    print("\n## 1. Comparativa de Desempeño en Test (Datos Desbalanceados)")
    print(df_metricas.to_markdown(floatfmt=".4f"))
    print("\n* **Observación:** Ambos modelos muestran un rendimiento predictivo casi idéntico.")
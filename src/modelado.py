import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (classification_report, roc_auc_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay)

# --- Funciones de Evaluación Auxiliares ---
def entrenar_evaluar_modelo(modelo, X_train, y_train, X_test, y_test, nombre_modelo, ruta_salida_reportes):
    """
    Entrena un modelo y muestra sus métricas de evaluación clave.
    """
    print(f"--- Evaluando: {nombre_modelo} ---")
    
    # Entrenar
    modelo.fit(X_train, y_train)
    
    # Predecir en el conjunto de prueba
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1] # Probabilidades para la clase 1 (quiebra)
    
    # Calcular y mostrar métricas
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Área Bajo la Curva ROC (AUC): {auc:.4f}")
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['No Quiebra (0)', 'Quiebra (1)']))
    
    # Graficar la Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Quiebra', 'Quiebra'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    filename = f"matriz_confusion_{nombre_modelo.replace(' ', '_').lower()}.png"
    ruta_guardado = os.path.join(ruta_salida_reportes, filename)
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Matriz de confusión guardada en: {ruta_guardado}")
    
    return modelo

def comparar_curvas_roc(modelos_entrenados, nombres, X_test, y_test, ruta_salida_reportes):
    """
    Grafica las curvas ROC para comparar visualmente el rendimiento de los modelos.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for modelo, nombre in zip(modelos_entrenados, nombres):
        RocCurveDisplay.from_estimator(modelo, X_test, y_test, name=nombre, ax=ax)
        
    ax.plot([0, 1], [0, 1], linestyle='--', color='r', label='Clasificador Aleatorio')
    plt.title("Comparación de Curvas ROC", fontsize=16)
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos (Recall)")
    plt.legend()
    plt.grid(True)
    ruta_guardado = os.path.join(ruta_salida_reportes, 'comparacion_roc_auc.png')
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Gráfico ROC comparativo guardado en: {ruta_guardado}")

# --- Función Principal del Módulo ---
def ejecutar_modelado(X_train, y_train, X_test, y_test, ruta_salida_modelos, ruta_salida_reportes):
    """
    Función principal para ejecutar el modelado y la evaluación de los modelos.
    Compara un modelo de Regresión Logística (MLE) con uno de Regresión Ridge (L2) 
    con selección de hiperparámetros por CV.
    """
    # --- Modelo 1: Regresión Logística (MLE) ---
    modelo_mle = LogisticRegression(
        penalty=None,            # Para usar máxima verosimilitud sin ninguna regularización
        class_weight='balanced', # Para manejar el gran desbalance de datos
        solver='lbfgs',          # lbfgs soporta penalty none
        max_iter=5000,           # 'saga' a veces necesita más iteraciones
        random_state=42
    )
    
    # --- Modelo 2: Regresión Ridge con Selección de Lambda (C) vía K-Folds CV ---
    grilla_C = np.logspace(-4, 4, 10) # 10 valores entre 0.0001 y 10000
    
    modelo_ridge_cv = LogisticRegressionCV(
        Cs=grilla_C,              # Valores C (inverso de lambda) a probar (C más pequeño = lambda más grande = más regularización)
        cv=5,                     # k-folds = 5
        penalty='l2',             # l2 es regularización Ridge
        scoring='roc_auc',        # Métrica para optimizar (¡clave para desbalance!)
        class_weight='balanced',
        solver='lbfgs',           # lbfgs por consistencia
        max_iter=5000,
        random_state=42
    )
    
    # --- Entrenar y Evaluar Modelos ---
    print("Entrenando Modelo 1: Regresión Logística (MLE)...")
    modelo_mle_entrenado = entrenar_evaluar_modelo(
        modelo_mle, X_train, y_train, X_test, y_test, "Regresión Logística (MLE)", ruta_salida_reportes)
    
    print("\n" + "="*50 + "\n")
    
    print("Entrenando Modelo 2: Regresión Ridge con K-Folds CV...")
    modelo_ridge_cv_entrenado = entrenar_evaluar_modelo(
        modelo_ridge_cv, X_train, y_train, X_test, y_test, "Regresión Ridge (con CV), ruta_salida_reportes", ruta_salida_reportes
    )
    
    # Reportamos el mejor hiperparámetro encontrado
    print("\n--- Resultados de la Validación Cruzada (Ridge) ---")
    print(f"Mejor C encontrado: {modelo_ridge_cv_entrenado.C_[0]:.4f}")
    
    # --- Comparación Visual ---
    texto4 = "Comparando el rendimiento de los modelos..."
    print("\n" + "="*50)
    print(texto4.center(50))
    print("="*50 + "\n")

    comparar_curvas_roc([modelo_mle_entrenado, modelo_ridge_cv_entrenado], ["Logística (MLE)", "Ridge (CV)"], X_test, y_test, ruta_salida_reportes)
    
    # --- Guardar Modelos Entrenados ---
    print("\nGuardando modelos entrenados...")
    joblib.dump(modelo_mle_entrenado, os.path.join(ruta_salida_modelos, 'modelo_mle.joblib'))
    joblib.dump(modelo_ridge_cv_entrenado, os.path.join(ruta_salida_modelos, 'modelo_ridge_cv.joblib'))
    print("Modelos guardados exitosamente.")
    
    return modelo_mle_entrenado, modelo_ridge_cv_entrenado
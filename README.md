# Proyecto-Estadistica-Avanzada
## Requisitos
- Usar datos reales de dominio especıfico.
- Aplicar al menos 2-3 tecnicas del curso apropiadamente.
- Implementar y evaluar multiples estimadores.
- Aplicar metodos de remuestreo para validacion.
- Interpretar resultados en contexto ingenieril/cientıfico

## Pregunta Central
¿Qué factores predicen mejor la probabilidad de quiebra de empresas?


## Objetivo
Identificar y evaluar los factores que mejor predicen la probabilidad de quiebra de empresas, utilizando técnicas estadísticas avanzadas para construir modelos robustos y validados.


## Datos

Corporate Bankruptcy Prediction:
- **Enlace**: https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction 

- **Características**: 95 variables predictoras (ratios financieros) y 1 variable objetivo (Bankrupt?).

- **Desafío Clave**: Alto desbalance de clases (96.8% No Quiebra, 3.2% Quiebra) y alta multicolinealidad.


## Métodos a Utilizar
1. Estimación y Shrinkage
- Regresión Logística (MLE): Estimador clásico sin regularización (penalty=None).
Regresión Ridge (L2): Estimador con shrinkage L2 para manejar multicolinealidad, con selección de $\lambda$ (lambda) vía Validación Cruzada.Regresión Lasso (L1): Estimador con shrinkage L1 para realizar selección de variables, también con selección de $\lambda$ vía Validación Cruzada.2. Preprocesamiento y Manejo de DatosWinsorización: Para controlar valores atípicos (outliers) extremos.SMOTE: Remuestreo para balancear el set de entrenamiento (50/50).StandardScaler: Estandarización de variables (media 0, varianza 1) para la correcta convergencia de los modelos.3. Validación y RemuestreoValidación Cruzada (K-Folds): Usada dentro de LogisticRegressionCV para seleccionar el mejor hiperparámetro de regularización.Bootstrap: Usado para re-muestrear los datos de entrenamiento y estimar los intervalos de confianza del 95% para los coeficientes de los modelos.4. Comparación y MétricaMétricas Clave: AUC (Area Under the Curve) como métrica principal de rendimiento, apoyada por Precisión, Recall y F1-Score.Análisis Gráfico: Comparación visual del trade-off sesgo-varianza a través de los intervalos de confianza obtenidos por Bootstrap.
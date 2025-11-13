# Plan de Trabajo: Proyecto Estadística Avanzada
## Etapa 1: Preparación y exploración de datos
- Descargar y revisar el dataset de Kaggle
Corporate Bankruptcy Prediction
- Exploración inicial (EDA)
- Identificar variables disponibles (ratios financieros, indicadores contables, etc.)
- Analizar distribución de la variable objetivo (Bankrupt)
- Detectar valores faltantes, outliers y correlaciones
- Preprocesamiento
- Imputación de valores faltantes
- Normalización o estandarización si es necesario
- Codificación de variables si hay categóricas (aunque este dataset es mayormente numérico)

## Etapa 2: Modelado base y estimación clásica
- Aplicar regresión logística MLE
- Estimar coeficientes
- Evaluar significancia estadística
- Interpretar coeficientes en contexto financiero
- Evaluar desempeño inicial
- Métricas: AUC, precisión, recall, F1-score
- Matriz de confusión

## Etapa 3: Técnicas de shrinkage y regularización- Aplicar Ridge Regression
- Comparar coeficientes con MLE
- Analizar reducción de varianza y estabilidad
- Aplicar James-Stein shrinkage
- En variables altamente correlacionadas
- Evaluar impacto en predicción y sesgo

## Etapa 4: Validación y remuestreo- Bootstrap
- Estimar intervalos de confianza para coeficientes
- Evaluar estabilidad de los modelos
- Validación cruzada (k-fold CV)
- Comparar desempeño entre MLE, Ridge y James-Stein
- Seleccionar el modelo más robusto

## Etapa 5: Comparación y análisis técnico- Comparar modelos
- Evaluar trade-off sesgo-varianza
- Comparar métricas de desempeño
- Justificar elección del modelo final
- Interpretación ingenieril
- Traducir resultados a recomendaciones prácticas
- Identificar factores clave de quiebra
- Proponer alertas o criterios de riesgo para empresas

## Etapa 6: Documentación y presentación- Estructurar informe técnico
- Introducción, pregunta, objetivos
- Metodología clara y justificada
- Resultados con gráficos y tablas
- Conclusiones prácticas
- Preparar presentación académica
- Diapositivas con visualizaciones
- Justificación de cada técnica
- Interpretación clara para audiencia no técnica


# PLAN 2.0 (revisar)
## Etapa 1 — Preparación y EDA (mejorada)

Objetivo: dejar un dataset reproducible, documentado y listo para modelado.

Acciones concretas ya realizadas / a ejecutar:

Carga y limpieza: quitar espacios en nombres (df.columns.str.strip()), documentar shape.

Reporte de nulos (hecho — no hay).

Identificación y eliminación automática:

Columnas constantes (nunique==1) → eliminar y listar.

Columnas duplicadas / correlación ≈1 → eliminar y listar.

Pares con correlación > 0.95 → eliminar una columna por par (ya implementado). Imprimir lista de columnas eliminadas por correlación y por ser constantes/duplicadas (hazlo en log).

Outliers: winsorización (1%/99%) o RobustScaler; documentar qué se hizo y por qué.

EDA visual:

Countplot y porcentaje para Bankrupt?.

Heatmap de correlación final (variables retenidas).

Boxplots de las 15 variables con mayor desviación estándar (o las 15 prioritarias).

Distribuciones separadas por clase (boxplot / KDE) para variables clave.

Colinealidad cuantitativa:

Calcular VIF para las variables retenidas; reportar variables con VIF>10.

Decidir: eliminar basado en VIF o agrupar con PCA/score financiero.

Agrupación semántica:

Crear grupos (Rentabilidad, Liquidez, Endeudamiento, Eficiencia, Crecimiento). Guardar estas agrupaciones para análisis interpretativo y para crear agregados si conviene (scores).

Splits reproducibles:

Crear hold-out test set estratificado (ej. 20% estratificado por Bankrupt?) y usar el 80% restante para entrenamiento/validación (CV). Guardar seed.

### preprocesamiento 3.0

Desactivar el vif ya que se tuvieron resultados de auc muy malos %53 

despues de probar primero quitandoel vif, luego haciendo que la limpieza de columnas haga menos cosas dejandole ese trabajo a ridge, y haciendo un remuestreo con Smote para balancear, nos dimos cuenta que muchas de las cosas que hcimos, incluida la winzorizacion, estaban balanceando los daots. es por esto que cabiamos de robust scaler a standard scaler
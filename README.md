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


## Métodos Utilizados

### 1. Estimación y Shrinkage
* **Regresión Logística (MLE):** Estimador clásico sin regularización (`penalty=None`). Se utiliza como modelo base para evaluar el impacto de la multicolinealidad.
* **Regresión Ridge (L2):** Estimador con shrinkage L2. Se utiliza para manejar la alta multicolinealidad entre las 93 variables predictoras que se entregan al modelo.

### 2. Preprocesamiento y Manejo de Datos
* **Winsorización:** Se aplica para controlar los outliers extremos (percentiles 1% y 99%) sin eliminar datos.
* **Imputación de Mediana:** Se utiliza `SimpleImputer` para rellenar valores faltantes (aunque el dataset actual no tiene nulos, se mantiene por robustez).
* **SMOTE (Remuestreo):** Se aplica **solo al set de entrenamiento** (después del split) para corregir el severo desbalance de clases (3.2% vs 96.8%) y crear un dataset 50/50 para el entrenamiento.
* **StandardScaler:** Estandarización de variables (media 0, varianza 1). Este es un paso crítico, implementado para asegurar la correcta convergencia de los solvers del modelo.

### 3. Validación y Remuestreo
* **Validación Cruzada (K-Folds):** Se utiliza en la etapa de modelado (vía `LogisticRegressionCV`) para seleccionar automáticamente el mejor hiperparámetro de regularización `C` (el inverso de $\lambda$) para el modelo Ridge.
* **Bootstrap:** Se implementa en `validacion.py`. Consiste en **re-muestrear con reemplazo** (B=1000 iteraciones) los datos de entrenamiento (ya balanceados con SMOTE) para re-entrenar los modelos MLE y Ridge (con su `C` óptimo) en cada muestra.

### 4. Comparación y Métrica
* **Métricas de Rendimiento:** **AUC (Area Under the Curve)** como métrica principal para evaluar el rendimiento en el set de prueba (desbalanceado). Se complementa con **Precisión, Recall y F1-Score**.
* **Análisis de Estabilidad:** Se calculan los **Intervalos de Confianza (IC) del 95%** para los coeficientes de ambos modelos a partir de las distribuciones generadas por Bootstrap.
* **Análisis Gráfico (Trade-off Sesgo-Varianza):** Se genera un gráfico de **barras de error (`errorbar`)** que compara visualmente los ICs de los coeficientes clave. Esto permite un análisis directo de la estabilidad (ancho del IC) y el shrinkage (media del coeficiente) entre el modelo MLE y el modelo Ridge.


## Categorización de Variables
El dataset de 95 variables consiste en ratios financieros que se pueden agrupar en cuatro categorías principales de análisis, más una de limpieza:

* **A. Rentabilidad:** Mide la capacidad de generar beneficios (ej. `ROA(C)`, `Operating Gross Margin`). Una baja rentabilidad suele ser un predictor temprano.
* **B. Liquidez y Cobertura:** Mide la capacidad de pago a corto plazo (ej. `Current Ratio`, `Quick Ratio`, `Cash Flow to Liability`). La iliquidez es una señal crítica.
* **C. Endeudamiento:** Mide el apalancamiento y el riesgo de insolvencia (ej. `Debt ratio %`, `Liability to Equity`).
* **D. Eficiencia Operativa:** Mide qué tan bien la empresa usa sus activos (ej. `Inventory Turnover`, `Accounts Receivable Turnover`).
* **E. Redundantes/Constantes:** Variables que no aportan información y se eliminan (ej. `Net Income Flag`).

---

## Dificultades y Evolución del Proyecto
El pipeline final es el resultado de un intenso proceso de depuración para resolver un rendimiento inicial de AUC ~0.59 (casi al azar).

* **Dificultad 1: Manejo de Outliers.**
    * **Proceso:** Se implementó `Winsorización` (v2) para controlar los valores extremos (percentiles 1% y 99%). Inicialmente se usó `RobustScaler` pensando en los outliers, pero esto resultó ser un problema.
* **Dificultad 2: Multicolinealidad.**
    * **Proceso:** En la v3, se implementó un filtro **VIF** (Factor de Inflación de la Varianza). Esto fue un error: eliminó 46 variables, destruyendo la señal predictiva y resultando en un AUC de ~0.53.
    * **Solución:** Se desactivó el VIF y la eliminación por correlación. Se decidió entregar todas las variables (93) a los modelos para que la **Regresión Ridge** maneje la multicolinealidad, permitiendo así una comparación justa contra MLE.
* **Dificultad 3: Desbalance de Clases y Convergencia.**
    * **Proceso:** El dataset está muy desbalanceado (3.2% quiebras). El AUC seguía siendo ~0.59.
    * **Solución 1:** Se implementó **SMOTE** (remuestreo) solo en el set de entrenamiento para balancear las clases (50/50).
    * **Solución 2 (El Hallazgo Clave):** El problema final era el escalador. `RobustScaler` no centraba los datos (media 0, varianza 1). Dado que la `Winsorización` ya controlaba los outliers, cambiamos a **`StandardScaler`**. Este cambio fue el que permitió a los solvers del modelo converger, disparando el **AUC final a ~0.94**.

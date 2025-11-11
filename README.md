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
https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction 


## Métodos a Utilizar
Para estimacion y shrinkcage:

- Regresión logística MLE para modelar la probabilidad de quiebra.
- James-Stein shrinkage para mejorar predicción en presencia de muchas variables correlacionadas.
- Ridge regularización para controlar sobreajuste y mejorar generalización.


Para remuestreo y validación:
- Bootstrap para estimar intervalos de confianza de los coeficientes.
- Validación cruzada (CV) para comparar modelos y seleccionar el mejor.


Para comparar:
- Evaluar trade-off sesgo-varianza entre modelos clásicos y regularizados.
- Comparar métricas como AUC, precisión, recall, F1-score.
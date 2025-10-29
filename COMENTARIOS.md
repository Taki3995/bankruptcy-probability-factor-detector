# ETAPA 1
## Preprocesamiento v1

### Columnas se separan en 5 grandes categorias:

A.- RENTABILIDAD (baja rentabilidad suele anticipar problemas financieros.)
- ROA(C) before interest and depreciation before interest → rentabilidad sobre activos antes de intereses y depreciación.

- ROA(A) before interest and depreciation after interest → similar, pero incluye intereses.

- Gross Profit to Sales → margen bruto.

- Operating Gross Margin → rentabilidad operacional.

B.- LIQUIDEZ Y COBERTURA (iliquidez es la señal más temprana de quiebra.)
- Current Ratio (activo corriente / pasivo corriente).

- Quick Ratio (activo corriente sin inventario / pasivo corriente).

- Cash Flow to Liability → flujo de caja sobre deuda total.

- Interest Coverage Ratio → utilidad operacional / gastos de intereses.

C.- ENDEUDAMIENTO (sobreapalancamiento aumenta el riesgo de insolvencia.)
- Debt ratio % → pasivo/activo total.

- Net worth/Assets → patrimonio / activos.

- Liability to Equity → deuda / patrimonio.

- Equity to Liability → inverso del anterior.

D.- EFICIENCIA OPERATIVA 
- Accounts Receivable Turnover → ventas / cuentas por cobrar.

- Inventory Turnover → costo de ventas / inventario.

- Working Capital to Assets → capital de trabajo / activos.

E.- VARIABLES REDUNDANTES / POCO UTILES
- Net Income Flag → constante en todo el dataset (valor = 1).
- Net Value Per Share (A) ≈ Net Value Per Share (C) 

### Otros comentarios
- hay un 3.2% de empresas quebradas, por lo que el dataset esta desbalanceado. Accuracy no nos servira mucho, por lo nos enfocaremos en AUC, recall y F1-Score para evaluar los modelos

- No hay valores nulos por lo que la imputacion no es necesaria. Se deja igualmente para robustez

- hay muchas variables con outliers extremos. es normal, pero puede afectar

- hay pares de columnas coc correlacion mayor al 95%, eliminar en versiones siguientes estos duplicados

- escalar con RobustScaler por los outliers

## Preprocesamiento v2

- implementamos el robustscaler para los outliers

- limpiamos espacios de mas en los nombres de las columnas

- calculamos las correlaciones fuertes (ahora del 95%, antes era del 80%)

- implementamos la winsorizacion para controlar outliers extremos. lo que hace es reemplazar valores extremos por el mas alto en el percentil limite (percentil 5 y 95)

- imprime cuales columnas se eliminan por redundancia, y cuales por alta correlacion

- muestra un countplot de la variable objetivo

- muestra heatmap de correlaciones finales para justificar la sleccion de variables

- revisar boxplot


## Preprocesamiento v3

- CORRECCIÓN: primero dividiremos los datos y luego los ajustaremos, para que solo se ajusten los datos de entrenamiento. Luego se aplica la transformacion a los datos de entrenamiento y a los de prueba.

- MODIFICACION a la funcion limpiar columnas, para que no se eliminen variables prioritarias sin condicionesanteriores.

- Se implementa el calculo del VIF (factor de inflación de la varianza) para detectar y eliminr la multicolinealidad que la correlacion por pares quizas no esta viendo. 

- Se corrige la funcion pipeline para implementar el vif y para que se guarden los transformers (imputer y scalers) y asi poder simplemente cargarlos en otro script.



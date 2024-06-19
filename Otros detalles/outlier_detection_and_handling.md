## 5. Detección y Manejo de Outliers 🚨

La detección y manejo de outliers es un paso crucial en el preprocesamiento de datos por varias razones:

### Importancia de Detectar y Manejar Outliers

1. **Mejora de la Precisión del Modelo**: Los outliers pueden distorsionar la interpretación de los datos y afectar negativamente el rendimiento del modelo. Al identificar y manejar los outliers, se mejora la precisión y la robustez del modelo.

2. **Reducción de Ruido**: Los outliers pueden ser ruido en los datos, lo que puede llevar a interpretaciones incorrectas y decisiones erróneas. Eliminarlos o tratarlos reduce el ruido y permite un análisis más claro y preciso.

3. **Mejora de las Métricas de Evaluación**: En la evaluación de modelos, los outliers pueden influir desproporcionadamente en las métricas, dando una impresión falsa del rendimiento del modelo. Al manejar los outliers, las métricas de evaluación reflejan mejor el rendimiento real del modelo.

4. **Prevención de Problemas Numéricos**: Algunos algoritmos de machine learning pueden ser sensibles a valores extremos, lo que puede causar problemas numéricos y llevar a resultados menos precisos o inestables.

5. **Mejor Convergencia de Algoritmos de Optimización**: Los outliers pueden afectar el proceso de optimización, especialmente en algoritmos basados en gradientes. Al manejarlos adecuadamente, se mejora la convergencia y eficiencia del algoritmo.

### Método del Rango Intercuartílico (IQR)

El método del rango intercuartílico (IQR) es una técnica común para identificar outliers. Aquí está el código para hacerlo:

```python
import numpy as np
import pandas as pd

# Datos de ejemplo
data = {'Feature1': [10, 12, 14, 15, 10, 10, 18, 19, 100, 101]}
df = pd.DataFrame(data)

# Calcular Q1 (cuartil 1) y Q3 (cuartil 3)
Q1 = df['Feature1'].quantile(0.25)
Q3 = df['Feature1'].quantile(0.75)

# Calcular IQR
IQR = Q3 - Q1

# Definir los límites inferior y superior para los outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificar los outliers
outliers = df[(df['Feature1'] < lower_bound) | (df['Feature1'] > upper_bound)]

print("Outliers encontrados:\n", outliers)
```
En este código:

Calculamos el primer cuartil (Q1) y el tercer cuartil (Q3) de la característica.
Determinamos el rango intercuartílico (IQR) como la diferencia entre Q3 y Q1.
Establecemos los límites inferior y superior para identificar los outliers.
Identificamos los outliers como aquellos valores que caen fuera de estos límites.

Manejar los outliers adecuadamente es fundamental para garantizar que los datos sean de alta calidad y que los modelos de machine learning puedan aprender de manera efectiva y precisa.

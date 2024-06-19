# Machine_learning XGBoost Twitch
# Proyecto para Explicar Cantidad de Viewers en Twitch Utilizando XGBoost 🚀

Bienvenido al proyecto que utiliza XGBoost para predecir la cantidad de espectadores en Twitch. Aquí mostraré el proceso para modelar datos para hacer predicciones precisas. ¡Vamos a comenzar! 🎉

## Descripción del Proyecto 📄

XGBoost (Extreme Gradient Boosting) es un algoritmo de aprendizaje automático basado en árboles de decisión. Utiliza el método de boosting, que combina varios árboles de decisión débiles para crear un modelo más robusto y preciso. XGBoost optimiza la predicción minimizando un objetivo de pérdida mediante el uso de técnicas avanzadas de optimización y regularización para mejorar el rendimiento y reducir el sobreajuste. Es conocido por su eficiencia, flexibilidad y capacidad para manejar datos estructurados y no estructurados.

## Pasos del Proyecto 🛠️
1. Recolección de Datos 📊
2. Preprocesamiento de Datos 🧹
3. División de Datos ✂️
4. Selección de Características 🔍
5. Configuración del Modelo ⚙️
6. Entrenamiento del Modelo 🧠
7. Evaluación del Modelo 📈
8. Ajuste de Hiperparámetros 🔧
9. Validación Cruzada 🔄
10. Implementación y Monitoreo 🚀

## Detalle paso a paso:

### 1. Recolección de Datos 📊
En este caso, usaré los datos de las transmisiones de Twitch disponibles en Kaggle. 

[https://www.kaggle.com/datasets/ashishkumarak/twitch-reviews-daily-updated](https://www.kaggle.com/datasets/hibrahimag1/top-1000-twitch-streamers-data-may-2024)

### 2. Preprocesamiento de Datos 🧹
Limpia y prepara los datos, incluyendo la eliminación de valores nulos, codificación de variables categóricas y normalización. Esto asegura que el modelo tenga datos de alta calidad para aprender.

### Pasos,
2.a. Inspección de Datos: Revisar los primeros registros y el resumen estadístico.
2.b. Manejo de Valores Nulos: Identificar y tratar los valores nulos en el dataset.
2.c. Codificación de Variables Categóricas: Convertir las variables categóricas en valores numéricos.
2.d. Normalización/Estandarización: Normalizar o estandarizar las características numéricas.
2.e. Detección y Manejo de Outliers: Identificar y tratar valores atípicos.
2.f. Creación de Nuevas Características: Crear nuevas características si es necesario.

#### 2.a. Inspección de Datos: Revisar los primeros registros y el resumen estadístico.
``` python
import pandas as pd
df = pd.read_csv('Twitch.csv')
# Mostrar los primeros registros
display(df.head())
```
![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/16696030-3982-4e45-a094-184e68a99767)

``` python
# Resumen estadístico
print(df.describe())
```
![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/09af9eb4-cd82-4d21-9223-b12d9a8a567e)

``` python
# Información sobre el dataframe
print(df.info())
```

![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/e84451bb-91ff-403f-aa95-b2c7eedb1216)

#### 2.b. Manejo de Valores Nulos: Identificar y tratar los valores nulos en el dataset.

``` python
# Ejemplo de eliminación de filas con valores nulos
#df.dropna(inplace=True)

# Alternativamente, podemos rellenar valores nulos con la media (para columnas numéricas) o la moda (para columnas categóricas)
# df.fillna(df.mean(), inplace=True)
# df.fillna(df.mode().iloc[0], inplace=True)

#En este caso como se trata de una variable categorica correspondiente al 2ND_MOST_STREAMED_GAME, lo completaré con el texto "Sin información"

# Rellenar valores nulos en la columna '2ND_MOST_STREAMED_GAME' con 'Sin información'
df['2ND_MOST_STREAMED_GAME'].fillna('Sin información', inplace=True)

# Verificar que los valores nulos han sido completados
print(df['2ND_MOST_STREAMED_GAME'].isnull().sum())
```

#### 2.c. Codificación de Variables Categóricas: Convertir las variables categóricas en valores numéricos.

``` python
# Identificar variables categóricas
categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)
```
![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/ca1bf838-393c-48d7-98fd-cf65ac45cec5)

Para, codificar las variables categoricas usaré Label Encoding, este método asigna un número entero único a cada categoría.

``` python
from sklearn.preprocessing import LabelEncoder

# Aplicar Label Encoding a las variables categóricas
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Verificar que las variables han sido codificadas
print(df.head())
```
Este código identifica las columnas categóricas y aplica Label Encoding a cada una de ellas. 
También guarda los codificadores LabelEncoder en un diccionario por si necesitamos revertir la codificación o realizar alguna transformación adicional en el futuro.

![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/576f3f09-8695-4e90-adcf-7d97f3ea24e9)

#### 2.d. Normalización/Estandarización: Normalizar o estandarizar las características numéricas.

Ahora utilizaremos StandardScaler de sklearn.preprocessing para estandarizar las características numéricas del dataset.
Esto va a identificar las columnas numéricas y aplicar la estandarización, es decir, vamos a ajustar los datos para que tengan una media de 0 y una desviación estándar de 1. 
Esto es útil para muchos algoritmos de machine learning que funcionan mejor cuando las características tienen una escala similar.

¿Por qué es importante?, lo explico en más detalle aquí: [normalization_and_standardization.md](Machine_learning_XGBoost_Twitch/Otros detalles/normalization_and_standardization.md).

![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/9c40d5b4-6c64-4ed4-ae46-340b6d75f1b6)

2.e. Detección y Manejo de Outliers: Identificar y tratar valores atípicos.

Para este punto utilizaremos el método del rango intercuartílico (IQR) para identificar y manejar los valores atípicos.
¿Por qué es importante?, lo explico en más detalle aquí: [outlier_detection_and_handling.md](Machine_learning_XGBoost_Twitch/Otros detalles/outlier_detection_and_handling.md).

``` python
import numpy as np

# Calcular el primer cuartil (Q1) y el tercer cuartil (Q3)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Definir límites inferior y superior
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Reemplazar outliers por los límites del IQR
df_no_outliers = df.copy()
for column in df_no_outliers.select_dtypes(include=['float64', 'int64']).columns:
    df_no_outliers[column] = np.where(df_no_outliers[column] < lower_bound[column], lower_bound[column], df_no_outliers[column])
    df_no_outliers[column] = np.where(df_no_outliers[column] > upper_bound[column], upper_bound[column], df_no_outliers[column])

# Opcional: Eliminar outliers
#df_outliers_removed = df[~outliers.any(axis=1)]

# Verificar los cambios
print(df_no_outliers.head())
```
Con esto calculamos el primer (Q1) y tercer cuartil (Q3) para cada columna.
Definimos los límites inferior y superior utilizando el IQR.
Reemplazamos los valores que están por debajo del límite inferior con el valor del límite inferior y los valores que están por encima del límite superior con el valor del límite superior.
Esto conserva todas las filas del dataset mientras limita el impacto de los outliers, otra opción es eliminar los outliers.

2.f. Creación de Nuevas Características: Crear nuevas características si es necesario.
La creación de nuevas características puede ayudar a mejorar el rendimiento del modelo al proporcionar información adicional derivada de las características existentes.

*La creación de nuevas características debe basarse en la comprensión del problema y el conocimiento del dominio, ya que características bien diseñadas pueden mejorar significativamente el rendimiento del modelo.

``` python
# Ratio de juegos por día activo
df_no_outliers['GAMES_PER_ACTIVE_DAY'] = df_no_outliers['TOTAL_GAMES_STREAMED'] / df_no_outliers['ACTIVE_DAYS_PER_WEEK']

# Interacción entre la cantidad de seguidores y el promedio de espectadores por transmisión
df_no_outliers['FOLLOWERS_X_VIEWERS'] = df_no_outliers['TOTAL_FOLLOWERS'] * df_no_outliers['AVG_VIEWERS_PER_STREAM']

# Porcentaje de días activos en una semana
df_no_outliers['ACTIVE_DAYS_PERCENTAGE'] = df_no_outliers['ACTIVE_DAYS_PER_WEEK'] / 7
```

### 3. División de Datos ✂️
Divide los datos en conjuntos de entrenamiento y prueba. Esto nos permitirá evaluar el rendimiento del modelo de manera objetiva.


### 4. Selección de Características 🔍
Identifica y selecciona las características relevantes que se utilizarán en el modelo. Este paso es crucial para mejorar la precisión y eficiencia del modelo.

### 5. Configuración del Modelo ⚙️
Configura los parámetros del modelo XGBoost. La configuración adecuada de los parámetros puede tener un gran impacto en el rendimiento del modelo.

### 6. Entrenamiento del Modelo 🧠
Entrena el modelo con el conjunto de datos de entrenamiento. Aquí es donde el modelo aprende a hacer predicciones basadas en los datos.

### 7. Evaluación del Modelo 📈
Evalúa el rendimiento del modelo utilizando el conjunto de datos de prueba y métricas de evaluación adecuadas como RMSE, MAE, etc.

### 8. Ajuste de Hiperparámetros 🔧
Ajusta los hiperparámetros del modelo para mejorar su rendimiento. Esto puede incluir la optimización de parámetros como learning rate, max depth, etc.

### 9. Validación Cruzada 🔄
Realiza validación cruzada para asegurar la robustez del modelo. Esto ayuda a garantizar que el modelo generalice bien a datos no vistos.

### 10. Implementación y Monitoreo 🚀
Implementa el modelo en producción y monitorea su desempeño en el tiempo. Es importante mantener el modelo actualizado y funcionando correctamente.

## Contribuciones 🤝

¡Las contribuciones son bienvenidas! Si tienes sugerencias o mejoras, por favor abre un issue o un pull request.

## Licencia 📜

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

## Contacto ✉️

Para preguntas o comentarios, por favor contacta a [tu_email@example.com](mailto:tu_email@example.com).

¡Gracias por visitar nuestro proyecto y esperamos que disfrutes trabajando con XGBoost tanto como nosotros! 🌟

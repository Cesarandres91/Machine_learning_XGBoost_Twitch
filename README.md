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

## 1. Recolección de Datos 📊
En este caso, usaré los datos de las transmisiones de Twitch disponibles en Kaggle. 

[https://www.kaggle.com/datasets/ashishkumarak/twitch-reviews-daily-updated](https://www.kaggle.com/datasets/hibrahimag1/top-1000-twitch-streamers-data-may-2024)

## 2. Preprocesamiento de Datos 🧹
Limpia y prepara los datos, incluyendo la eliminación de valores nulos, codificación de variables categóricas y normalización. Esto asegura que el modelo tenga datos de alta calidad para aprender.

  2.a. **Inspección de Datos**: Revisar los primeros registros y el resumen estadístico.
   
   2.b. **Manejo de Valores Nulos**: Identificar y tratar los valores nulos en el dataset.
   
   2.c. **Codificación de Variables Categóricas**: Convertir las variables categóricas en valores numéricos.
   
   2.d. **Normalización/Estandarización**: Normalizar o estandarizar las características numéricas.
   
   2.e. **Detección y Manejo de Outliers**: Identificar y tratar valores atípicos.
   
   2.f. **Creación de Nuevas Características**: Crear nuevas características si es necesario.

### 2.a. Inspección de Datos: Revisar los primeros registros y el resumen estadístico.
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

### 2.b. Manejo de Valores Nulos: Identificar y tratar los valores nulos en el dataset.

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

### 2.c. Codificación de Variables Categóricas: Convertir las variables categóricas en valores numéricos.

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

### 2.d. Normalización/Estandarización: Normalizar o estandarizar las características numéricas.

Ahora utilizaremos StandardScaler de sklearn.preprocessing para estandarizar las características numéricas del dataset.
Esto va a identificar las columnas numéricas y aplicar la estandarización, es decir, vamos a ajustar los datos para que tengan una media de 0 y una desviación estándar de 1. 
Esto es útil para muchos algoritmos de machine learning que funcionan mejor cuando las características tienen una escala similar.

¿Por qué es importante?, lo explico en más detalle aquí: [normalization_and_standardization.md](Machine_learning_XGBoost_Twitch/Otros detalles/normalization_and_standardization.md).

![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/9c40d5b4-6c64-4ed4-ae46-340b6d75f1b6)

### 2.e. Detección y Manejo de Outliers: Identificar y tratar valores atípicos.

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

### 2.f. Creación de Nuevas Características: Crear nuevas características si es necesario.
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

## 3. División de Datos ✂️

Divide los datos en conjuntos de entrenamiento y prueba. Esto nos permitirá evaluar el rendimiento del modelo de manera objetiva.

3.a Estratificación: Si la variable objetivo está desbalanceada, usar la estratificación para asegurar que la proporción de clases se mantenga en ambos conjuntos.
3.b Escalado: Asegurarse de que las transformaciones (como la normalización/estandarización) se apliquen correctamente.
3.c Separación de Validación: Crear un conjunto de validación si es necesario para ajustar los hiperparámetros del modelo.

``` python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definir las características (X) y la variable objetivo (y)
X = df_no_outliers.drop('AVG_VIEWERS_PER_STREAM', axis=1)
y = df_no_outliers['AVG_VIEWERS_PER_STREAM']

# Dividir los datos en conjuntos de entrenamiento y prueba con estratificación (si es necesario)
# Aquí no utilizamos estratificación ya que es una variable continua. Estratificación se usa típicamente para variables categóricas.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar escalado a los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificar las formas de los conjuntos de entrenamiento y prueba
print("Conjunto de entrenamiento:", X_train_scaled.shape, y_train.shape)
print("Conjunto de prueba:", X_test_scaled.shape, y_test.shape)
```
Aquí se define las características (X) eliminando la columna AVG_VIEWERS_PER_STREAM del dataframe y define la variable objetivo (y) como AVG_VIEWERS_PER_STREAM.
Utiliza train_test_split para dividir los datos en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%), con una semilla aleatoria (random_state) de 42 para garantizar la reproducibilidad.
Verifica las formas de los conjuntos de entrenamiento y prueba.

## 4. Selección de Características 🔍
Identifica y selecciona las características relevantes que se utilizarán en el modelo, será una ayuda para mejorar la precisión y eficiencia del modelo.
Primero, entrenaremos un modelo XGBoost inicial para obtener las importancias de las características y luego seleccionaremos las más relevantes.

Correlación de variables,
import pandas as pd
``` python
# 'df_no_outliers' es el dataframe con todas las características y sin outliers
# 'AVG_VIEWERS_PER_STREAM' es la variable objetivo

# Calcular la matriz de correlación
correlation_matrix = df_no_outliers.corr()

# Extraer la correlación de todas las variables con respecto a la variable objetivo
correlation_with_target = correlation_matrix['AVG_VIEWERS_PER_STREAM']

# Mostrar las correlaciones
print("Correlación de todas las variables con respecto a la variable objetivo 'AVG_VIEWERS_PER_STREAM':")
print(correlation_with_target)
```
![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/0d34f889-a161-4633-8fbc-7a4831522f16)

Para poder visualizarlo mejor:

``` python
import seaborn as sns
import matplotlib.pyplot as plt

# Configurar el tamaño del gráfico
plt.figure(figsize=(12, 8))

# Crear un mapa de calor de la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

# Mostrar el gráfico
plt.title('Matriz de Correlación')
plt.show()
```
![descarga (1)](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/13b51fdf-5288-4603-b980-aa20f4030389)



``` python
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

# Definir un modelo XGBoost inicial
initial_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# Entrenar el modelo inicial
initial_model.fit(X_train_scaled, y_train)

# Obtener la importancia de las características
importances = initial_model.feature_importances_

# Crear un selector basado en el modelo
selector = SelectFromModel(initial_model, prefit=True)

# Transformar los datos de entrenamiento y prueba para seleccionar las características importantes
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Verificar las nuevas formas de los conjuntos de entrenamiento y prueba
print("Conjunto de entrenamiento después de la selección de características:", X_train_selected.shape)
print("Conjunto de prueba después de la selección de características:", X_test_selected.shape)

# Mostrar la importancia de las características
xgb.plot_importance(initial_model)
```
![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/69be1f45-c289-499f-90a3-027639a006df)

## 5. Configuración del Modelo ⚙️
Configura los parámetros del modelo XGBoost. La configuración adecuada de los parámetros puede tener un gran impacto en el rendimiento del modelo.
Vamos a configurar el modelo XGBoost con los hiperparámetros iniciales y entrenarlo utilizando las características seleccionadas, para ello
-Define el modelo XGBoost con algunos hiperparámetros iniciales.
-Entrena el modelo utilizando el conjunto de entrenamiento con las características seleccionadas.
-Realiza predicciones sobre el conjunto de prueba.
-Evalúa el rendimiento del modelo utilizando el error cuadrático medio (MSE).
-Muestra la importancia de las características utilizando una gráfica.

``` python
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Definir el modelo XGBoost con algunos hiperparámetros iniciales
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(X_train_selected, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test_selected)

# Evaluar el rendimiento del modelo utilizando el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Mostrar la importancia de las características
xgb.plot_importance(model)
```
![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/46dcc929-f855-43c6-94da-213f96bdc765)


Ajuste de Hiperparámetros
Después de entrenar el modelo inicial, puedes ajustar los hiperparámetros para mejorar el rendimiento del modelo. Esto se puede hacer utilizando técnicas como la búsqueda en cuadrícula (GridSearchCV) o la búsqueda aleatoria (RandomizedSearchCV) de sklearn.model_selection.
*GridSearchCV es una técnica de optimización de hiperparámetros proporcionada por la biblioteca scikit-learn en Python. Su objetivo es encontrar la mejor combinación de hiperparámetros para un modelo de machine learning a través de una búsqueda exhaustiva sobre un conjunto especificado de parámetros, provee:
Búsqueda Exhaustiva: GridSearchCV prueba todas las combinaciones posibles de los hiperparámetros especificados en una cuadrícula (grid) de parámetros.
Validación Cruzada: Utiliza la validación cruzada para evaluar el rendimiento de cada combinación de hiperparámetros. La validación cruzada divide los datos en múltiples subconjuntos y evalúa el modelo varias veces para obtener una estimación robusta del rendimiento.
Selección del Mejor Modelo: Selecciona la combinación de hiperparámetros que da el mejor rendimiento según la métrica de evaluación especificada (por ejemplo, precisión, F1-score, etc.).
Facilita el Ajuste de Hiperparámetros: Permite ajustar automáticamente los hiperparámetros del modelo sin tener que hacerlo manualmente, lo que puede ser tedioso y propenso a errores.

``` python
from sklearn.model_selection import GridSearchCV

# Definir el modelo XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Ajustar GridSearchCV
grid_search.fit(X_train_selected, y_train)

# Mostrar los mejores hiperparámetros
print(f"Best hyperparameters: {grid_search.best_params_}")

# Utilizar el mejor modelo encontrado para predecir
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_selected)

# Evaluar el rendimiento del mejor modelo utilizando el error cuadrático medio
mse_best = mean_squared_error(y_test, y_pred_best)
print(f"Mean Squared Error (Best Model): {mse_best}")
```

![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/f0c9c05b-71ee-4eb7-b4f2-4c11e23437c0)

Lo anterior,
Define un modelo XGBoost.
Define una cuadrícula de hiperparámetros para explorar.
Configura y ajusta GridSearchCV para encontrar los mejores hiperparámetros.
Muestra los mejores hiperparámetros encontrados.
Utiliza el mejor modelo encontrado para predecir y evaluar su rendimiento.



## 6. Entrenamiento del Modelo 🧠
Entrena el modelo con el conjunto de datos de entrenamiento. Aquí es donde el modelo aprende a hacer predicciones basadas en los datos.



## 7. Evaluación del Modelo 📈
Evalúa el rendimiento del modelo utilizando el conjunto de datos de prueba y métricas de evaluación adecuadas como RMSE, MAE, etc.
``` python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predecir en el conjunto de prueba
y_pred = best_model.predict(X_test_selected)

# Calcular las métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar los resultados
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
```
![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/cf950ad0-c358-40f1-a468-c3210f0eb73d)

Los resultados obtenidos son bastante buenos y muestran que el modelo está funcionando bien

Mean Squared Error (MSE): 0.008225852874267157
El MSE es la media de los errores al cuadrado. Un valor más bajo indica que las predicciones están cerca de los valores reales. En este caso, 0.0082 es un valor bastante bajo, lo que sugiere que el modelo tiene buenos resultados.
Root Mean Squared Error (RMSE): 0.09069648766224168

El RMSE es la raíz cuadrada del MSE y proporciona una medida de error en las mismas unidades que la variable de salida. Un RMSE de aproximadamente 0.091 indica que el error típico del modelo es pequeño.
Mean Absolute Error (MAE): 0.052813036632756494

El MAE es la media de los errores absolutos. Un MAE de aproximadamente 0.053 indica que, en promedio, las predicciones del modelo están a 0.053 unidades de los valores reales.
R-squared (R²): 0.9435291447382572
El R² es una medida que indica la proporción de la varianza en la variable dependiente que es explicada por el modelo. Un R² de 0.944 sugiere que el 94.4% de la variación en los datos es explicada por el modelo, lo cual es muy bueno.




## 8. Ajuste de Hiperparámetros 🔧
Ajusta los hiperparámetros del modelo para mejorar su rendimiento. Esto puede incluir la optimización de parámetros como learning rate, max depth, etc.

## 9. Validación Cruzada 🔄 (Cross validation)
Realiza validación cruzada para asegurar la robustez del modelo. Esto ayuda a garantizar que el modelo generalice bien a datos no vistos.

La validación cruzada es una técnica utilizada para evaluar la capacidad de generalización de un modelo de machine learning. 
Proporciona una estimación más robusta del rendimiento del modelo al dividir los datos en múltiples subconjuntos y entrenar y evaluar el modelo en diferentes combinaciones de estos subconjuntos.

Validación Cruzada con XGBoost, utilizaremos la función cross_val_score de sklearn.model_selection para realizar la validación cruzada. 
Para esto, necesitaremos definir el modelo y usar la métrica adecuada (en este caso, utilizaremos el error cuadrático medio negativo).

``` python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Definir el modelo XGBoost con los mejores hiperparámetros encontrados
model = xgb.XGBRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    learning_rate=grid_search.best_params_['learning_rate'],
    max_depth=grid_search.best_params_['max_depth'],
    subsample=grid_search.best_params_['subsample'],
    colsample_bytree=grid_search.best_params_['colsample_bytree'],
    objective='reg:squarederror',
    random_state=42
)

# Definir la métrica de evaluación (error cuadrático medio negativo)
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Realizar la validación cruzada
cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring=scorer)

# Calcular la media y desviación estándar de los scores
mean_cv_score = -cv_scores.mean()
std_cv_score = cv_scores.std()

# Mostrar los resultados
print(f"Cross-Validation Mean Squared Error: {mean_cv_score}")
print(f"Standard Deviation of Cross-Validation MSE: {std_cv_score}")
```

![image](https://github.com/Cesarandres91/Machine_learning_XGBoost_Twitch/assets/102868086/ef9b9af4-bef9-4c63-8e59-620252893a61)

Los resultados de la validación cruzada indican que el modelo tiene un buen desempeño generalizado. Aquí está la interpretación de cada métrica:

Cross-Validation Mean Squared Error (MSE): 0.010974917294077107

Este valor indica el error cuadrático medio promedio a través de las diferentes particiones de la validación cruzada. Un valor de aproximadamente 0.011 es bastante bajo, lo que sugiere que el modelo está haciendo predicciones precisas en general.
Standard Deviation of Cross-Validation MSE: 0.001366182503274942

La desviación estándar del MSE a través de las particiones es aproximadamente 0.0014. Esto es bastante bajo, lo que indica que el modelo es consistente en su rendimiento a través de las diferentes particiones de los datos.
Resumen de los Resultados

Conclusión:
El modelo XGBoost tiene un buen rendimiento predictivo en general.
El modelo es consistente y no muestra grandes variaciones en su desempeño a través de diferentes subconjuntos de datos, lo que sugiere una buena capacidad de generalización.


## 10. Implementación y Monitoreo 🚀
Implementa el modelo en producción y monitorea su desempeño en el tiempo. Es importante mantener el modelo actualizado y funcionando correctamente.

``` python
import joblib

# Guardar el mejor modelo encontrado
joblib.dump(best_model, 'best_xgboost_model.pkl')
print("Modelo guardado exitosamente.")
```
Para cargar el modelo guardado y realizar predicciones en un entorno de producción, utiliza el siguiente código

``` python
# Cargar el modelo guardado
loaded_model = joblib.load('best_xgboost_model.pkl')
print("Modelo cargado exitosamente.")

# Realizar predicciones con el modelo cargado
new_predictions = loaded_model.predict(X_test_selected)

# Evaluar el rendimiento del modelo cargado
mse_loaded_model = mean_squared_error(y_test, new_predictions)
print(f"Mean Squared Error (Loaded Model): {mse_loaded_model}")
```

Monitoreo del Desempeño del Modelo,
Implementa técnicas para monitorear el rendimiento del modelo en producción:

Evaluar las Predicciones Regularmente: Comparar las predicciones del modelo con los resultados reales.
Actualizar el Modelo: Reentrenar el modelo periódicamente con datos nuevos.
Alarmas y Alertas: Configurar alarmas para alertar cuando el rendimiento del modelo cae por debajo de un umbral predefinido.
Registros y Seguimiento de Métricas: Mantener registros de las métricas de rendimiento del modelo para analizar su evolución.

## Contribuciones 🤝

¡Las contribuciones son bienvenidas! Si tienes sugerencias o mejoras, por favor abre un issue o un pull request.

## Licencia 📜

Este proyecto está bajo la Licencia Mozilla Public License Version 2.0. Ver el archivo LICENSE para más detalles.


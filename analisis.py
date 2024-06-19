import pandas as pd
df = pd.read_csv('Twitch.csv')
# Mostrar los primeros registros
display(df.head())

# Resumen estadístico
print(df.describe())

# Información sobre el dataframe
print(df.info())

# Identificar valores nulos
print(df.isnull().sum())

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

# Identificar variables categóricas
categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)


from sklearn.preprocessing import LabelEncoder

# Aplicar Label Encoding a las variables categóricas
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Verificar que las variables han sido codificadas
print(df.head())


from sklearn.preprocessing import StandardScaler

# Identificar columnas numéricas
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print(numeric_columns)

# Estandarizar características numéricas
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Verificar que las características numéricas han sido estandarizadas
print(df.head())


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


# Ratio de juegos por día activo
df_no_outliers['GAMES_PER_ACTIVE_DAY'] = df_no_outliers['TOTAL_GAMES_STREAMED'] / df_no_outliers['ACTIVE_DAYS_PER_WEEK']

# Interacción entre la cantidad de seguidores y el promedio de espectadores por transmisión
df_no_outliers['FOLLOWERS_X_VIEWERS'] = df_no_outliers['TOTAL_FOLLOWERS'] * df_no_outliers['AVG_VIEWERS_PER_STREAM']

# Porcentaje de días activos en una semana
df_no_outliers['ACTIVE_DAYS_PERCENTAGE'] = df_no_outliers['ACTIVE_DAYS_PER_WEEK'] / 7
# Verificar las nuevas características
print(df_no_outliers.head())

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


import joblib

# Guardar el mejor modelo encontrado
joblib.dump(best_model, 'best_xgboost_model.pkl')
print("Modelo guardado exitosamente.")


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel

# Entrenar un modelo inicial para la selección de características
initial_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# Ajustar el modelo inicial
initial_model.fit(X_train_scaled, y_train)

# Crear un selector basado en el modelo
selector = SelectFromModel(initial_model, prefit=True)

# Obtener el soporte de las características seleccionadas
selected_features_mask = selector.get_support()

# Obtener los nombres de las características seleccionadas
selected_features = X.columns[selected_features_mask]

# Mostrar las características seleccionadas
print("Características seleccionadas:")
print(selected_features)

import pandas as pd

# Suponiendo que 'df_no_outliers' es el dataframe con todas las características y sin outliers
# Y que 'AVG_VIEWERS_PER_STREAM' es la variable objetivo

# Calcular la matriz de correlación
correlation_matrix = df_no_outliers.corr()

# Extraer la correlación de todas las variables con respecto a la variable objetivo
correlation_with_target = correlation_matrix['AVG_VIEWERS_PER_STREAM']

# Mostrar las correlaciones
print("Correlación de todas las variables con respecto a la variable objetivo 'AVG_VIEWERS_PER_STREAM':")
print(correlation_with_target)

import seaborn as sns
import matplotlib.pyplot as plt

# Configurar el tamaño del gráfico
plt.figure(figsize=(12, 8))

# Crear un mapa de calor de la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

# Mostrar el gráfico
plt.title('Matriz de Correlación')
plt.show()

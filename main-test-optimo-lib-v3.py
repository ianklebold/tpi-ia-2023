# Implementar el algoritmo de knn optimo sin utilizar librerías que no sean nativas de python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv("dataset_cleaned_fixed.csv")

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop("TenYearCHD", axis=1)
y = data["TenYearCHD"]

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un clasificador KNN estándar con un valor de K fijo
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Entrenar el modelo en el conjunto de entrenamiento
knn.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
predictions = knn.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, predictions)
print("Precisión del modelo KNN Estándar:", accuracy)



#KNN optimo
from sklearn.model_selection import GridSearchCV

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop("TenYearCHD", axis=1)
y = data["TenYearCHD"]

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el rango de valores de K que deseas probar
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# Crear un clasificador KNN óptimo utilizando GridSearchCV
knn_optimo = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn_optimo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
predictions_optimo = knn_optimo.predict(X_test)

# Imprimir la precisión del modelo KNN óptimo
print("Precisión del modelo KNN Óptimo:", knn_optimo.best_score_)

# Imprimir el valor óptimo de K encontrado por GridSearchCV
print("Valor óptimo de K:", knn_optimo.best_params_)



#KNN ponderado
from sklearn.neighbors import KNeighborsClassifier

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop("TenYearCHD", axis=1)
y = data["TenYearCHD"]

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un clasificador KNN ponderado con pesos personalizados
# {'n_neighbors': 11}
k = knn_optimo.best_params_.get('n_neighbors')
knn = KNeighborsClassifier(n_neighbors=k, weights='distance')

# Entrenar el modelo
knn.fit(X_train, y_train)

# Realizar predicciones
predictions = knn.predict(X_test)
# print("Predicción KNN Ponderado:", predictions)

# Imprimir la precisión del modelo KNN
accuracy = accuracy_score(y_test, predictions)
print("Precisión del modelo KNN Ponderado:", accuracy)


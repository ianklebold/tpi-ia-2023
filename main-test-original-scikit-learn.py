import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# # Lee el dataset desde un archivo CSV
dataset = pd.read_csv('resources/dataset_cleaned_fixed.csv')
informacion = dataset.describe()
print(informacion)
# # Divide el dataset en características (X) y etiquetas (y)
id_column = dataset.iloc[:, 0]
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

#### test ####
# import subprocess
# import sys
# import os

# Instalar la libreria ucimlrepo
# subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
# from ucimlrepo import fetch_ucirepo 
  
# # fetch dataset 
# heart_disease = fetch_ucirepo(id=45) 
  
# # data (as pandas dataframes) 

# X = heart_disease.data.features 
# y = heart_disease.data.targets 
  
#### 


# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Tamaño del conjunto de entrenamiento:", len(X_train))
print("Tamaño del conjunto de prueba:", len(X_test))

# Crea los DataFrames para los conjuntos de entrenamiento y prueba
train_data = pd.concat([id_column[X_train.index], X_train, y_train], axis=1)
test_data = pd.concat([id_column[X_test.index], X_test, y_test], axis=1)
# train_data = pd.concat([X_train, y_train], axis=1)
# test_data = pd.concat([X_test, y_test], axis=1)

from sklearn.impute import SimpleImputer

# Crea una instancia del imputador y ajusta los datos de entrenamiento
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)

# Transforma los datos de entrenamiento y prueba
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


# Guarda los conjuntos de entrenamiento y prueba en archivos CSV
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
# train_data.to_csv('train_data_heard.csv', index=False)
# test_data.to_csv('test_data_heard.csv', index=False)

# print("X_train:")
# print(X_train.head())
# print("y_train:")
# print(y_train.head())

# Crea una instancia del clasificador KNN con un valor de K deseado
knn = KNeighborsClassifier(n_neighbors=11)

# Entrena el modelo utilizando el conjunto de entrenamiento
knn.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

from sklearn.model_selection import cross_val_score
# Crea una instancia del clasificador KNN con un valor de K deseado
knn = KNeighborsClassifier(n_neighbors=11)

# Realiza la validación cruzada con 5 particiones (k = 5) y calcula la precisión en cada partición
scores = cross_val_score(knn, X, y, cv=5)

# Muestra la precisión promedio y la desviación estándar de las puntuaciones de validación cruzada
print("Precisión de validación cruzada (promedio):", scores.mean())
print("Desviación estándar de las puntuaciones de validación cruzada:", scores.std())

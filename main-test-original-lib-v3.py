import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Datos de entrenamiento y prueba
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 1, 1, 0])
X_test = np.array([[2.5, 3.5]])

# Crear un clasificador KNN estándar con un valor de K fijo
k = 2
knn = KNeighborsClassifier(n_neighbors=k)

# Entrenar el modelo
knn.fit(X_train, y_train)

# Realizar predicciones
predictions = knn.predict(X_test)
print("Predicción KNN Estándar:", predictions)

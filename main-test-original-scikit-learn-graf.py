import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Lee el dataset desde un archivo CSV
dataset = pd.read_csv('resources/dataset_cleaned_fixed.csv')

# Divide el dataset en características (X) y etiquetas (y)
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea una lista para almacenar las precisión para cada valor de K
precisions = []

# Prueba diferentes valores de K y calcula la precisión para cada uno
for k in range(1, 30):
    # Crea una instancia del clasificador KNN con el valor de K actual
    knn = KNeighborsClassifier(n_neighbors=k)

    # Entrena el modelo utilizando el conjunto de entrenamiento
    knn.fit(X_train, y_train)

    # Realiza predicciones en el conjunto de prueba
    y_pred = knn.predict(X_test)

    # Calcula la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)

    # Agrega la precisión a la lista
    precisions.append(accuracy)

# Encuentra el índice del valor máximo de precisión
max_index = precisions.index(max(precisions))
print("Máxima precisión:", precisions[max_index], "con K =", max_index + 1)

# Grafica los resultados de la precisión para cada valor de K del 1 al 30
plt.plot(range(1, 30), precisions[:30])
plt.scatter(max_index + 1, precisions[max_index], color='red', label='Máximo')
plt.xticks(range(1, 30))
plt.xlabel('Valor de K')
plt.ylabel('Precisión')
plt.title('Precisión en función del valor de K')
plt.legend()
plt.show()

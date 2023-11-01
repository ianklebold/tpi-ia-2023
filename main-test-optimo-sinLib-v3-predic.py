# Implementar el algoritmo de knn optimo sin utilizar librerías que no sean nativas de python
import numpy as np
import pandas as pd
import warnings
# Implementar el algoritmo de knn optimo utilizando librerías de python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv("dataset_cleaned_fixed.csv")

# Crea una clase para representar cada instancia
class Instancia:
    def __init__(self, id, atributos, clase):
        self.id = id
        self.atributos = atributos
        self.clase = clase

# Lee el dataset de entrenamiento desde un archivo CSV
train_data = pd.read_csv('train_data.csv')

# Lee el dataset de prueba desde un archivo CSV
test_data = pd.read_csv('test_data.csv')

# Crea una lista para almacenar las instancias del dataset de entrenamiento
instancias_train = []

# Itera sobre las filas del dataset de entrenamiento
for index, row in train_data.iterrows():
    # Obtiene los atributos de la instancia y los convierte a tipo 'float'
    atributos = [float(value) for value in row[1:-1]]
    # Obtiene el ID de la instancia
    id_instancia = int(row['id'])
    # Obtiene la clase de la instancia
    clase = row[-1]
    # Crea una instancia y la agrega a la lista
    instancia = Instancia(id_instancia, atributos, clase)
    instancias_train.append(instancia)

# Crea una lista para almacenar las instancias del dataset de prueba
instancias_test = []

# Itera sobre las filas del dataset de prueba
for index, row in test_data.iterrows():
    # Obtiene los atributos de la instancia y los convierte a tipo 'float'
    atributos = [float(value) for value in row[1:-1]]
    # Obtiene el ID de la instancia
    id_instancia = int(row['id'])
    # Obtiene la clase de la instancia
    clase = row[-1]
    # Crea una instancia y la agrega a la lista
    instancia = Instancia(id_instancia, atributos, clase)
    instancias_test.append(instancia)


# Definir una función para calcular la distancia euclidiana entre dos instancias
def euclidean_distance(instance1, instance2):
    return np.sqrt(np.sum((np.array(instance1.atributos) - np.array(instance2.atributos)) ** 2))

# Realizar el calculo de la distancia Manhattan
def calcular_distancia_manhattan(instancia, instancia_nueva):
    distancia = 0
    for i in range(len(instancia.atributos)):
        distancia += abs(instancia.atributos[i] - instancia_nueva.atributos[i])
    return distancia

# Definir una función para calcular la distancia de Minkowski entre dos instancias
def minkowski_distance(x1, x2, p):
    distance = 0
    for i in range(len(x1)):
        distance += abs(x1[i] - x2[i]) ** p

    # Deshabilitar la advertencia específica (overflow)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in double_scalars")
        result = distance ** (1/p)
    
    return result

def calcular_distancia_chebyshev(instancia, instancia_nueva):
  distances = []
  for i in range(len(instancia.atributos)):
    distances.append(abs(instancia.atributos[i] - instancia_nueva.atributos[i]))
  return max(distances)

# Definir una función para predecir la clase de una instancia usando KNN estándar
def predict_knn_standard(train_data, test_instance, k, tipo_distancia):
    # print("tipo_distancia: ", tipo_distancia)
    if tipo_distancia == "euclidean":
        distances = [(euclidean_distance(train_instance, test_instance), train_instance.clase) for train_instance in train_data]
    elif tipo_distancia == "manhattan":
        distances = [(calcular_distancia_manhattan(train_instance, test_instance), train_instance.clase) for train_instance in train_data]
    elif tipo_distancia == "minkowski":
        distances = [(minkowski_distance(train_instance.atributos, test_instance.atributos, 3), train_instance.clase) for train_instance in train_data]
    elif tipo_distancia == "chebyshev":
        distances = [(calcular_distancia_chebyshev(train_instance, test_instance), train_instance.clase) for train_instance in train_data]
    else:
        raise ValueError("Tipo de distancia no válido")
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    class_counts = {}
    for neighbor in neighbors:
        _, class_label = neighbor
        if class_label in class_counts:
            class_counts[class_label] += 1
        else:
            class_counts[class_label] = 1
    return max(class_counts, key=class_counts.get)


def predict_knn_weighted(train_data, test_instance, k, tipo_distancia):
    if tipo_distancia == "euclidean":
        distances = [(euclidean_distance(train_instance, test_instance), train_instance.clase) for train_instance in train_data]
    elif tipo_distancia == "manhattan":
        distances = [(calcular_distancia_manhattan(train_instance, test_instance), train_instance.clase) for train_instance in train_data]
    elif tipo_distancia == "minkowski":
        distances = [(minkowski_distance(train_instance.atributos, test_instance.atributos, 3), train_instance.clase) for train_instance in train_data]
    elif tipo_distancia == "chebyshev":
        distances = [(calcular_distancia_chebyshev(train_instance, test_instance), train_instance.clase) for train_instance in train_data]
    else:
        raise ValueError("Tipo de distancia no válido")
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    class_weights = {}
    for neighbor in neighbors:
        distance, class_label = neighbor
        if distance == 0:
            return class_label
        weight = 1 / distance
        if class_label in class_weights:
            class_weights[class_label] += weight
        else:
            class_weights[class_label] = weight
    return max(class_weights, key=class_weights.get)


# Entrenar y probar el modelo KNN estándar
def knn_standard_accuracy(train_data, test_data, k, tipo_distancia):
    correct = 0
    for test_instance in test_data:
        predicted_class = predict_knn_standard(train_data, test_instance, k, tipo_distancia)
        if predicted_class == test_instance.clase:
            correct += 1
    return correct / len(test_data)


k_standard = 11  # Puedes ajustar este valor
tipo_distancia = "manhattan"  # Puedes ajustar este valor
accuracy_standard = knn_standard_accuracy(instancias_train, instancias_test, k_standard, tipo_distancia)
print(f'Precisión del modelo KNN Estándar (k={k_standard}, distancia={tipo_distancia}): {accuracy_standard:.5f}')


# Encontrar el valor óptimo de K para KNN estándar
k_values = [1, 3, 5, 7, 9, 11, 13]  # Puedes ajustar estos valores


def find_optimal_k(train_data, k_values, tipo_distancia):
    best_k = None
    best_accuracy = 0
    for k in k_values:
        accuracy = knn_standard_accuracy(train_data, instancias_test, k, tipo_distancia)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    return best_k

best_k_standard = find_optimal_k(instancias_train, k_values, tipo_distancia)


print(f'Valor óptimo de K para KNN Estándar: {best_k_standard}')

# Entrenar y probar el modelo KNN ponderado
k_weighted = 11  # Puedes ajustar este valor

def knn_weighted_accuracy(train_data, test_data, k, tipo_distancia):
    correct = 0
    for test_instance in test_data:
        predicted_class = predict_knn_weighted(train_data, test_instance, k, tipo_distancia)
        if predicted_class == test_instance.clase:
            correct += 1
    return correct / len(test_data)

accuracy_weighted = knn_weighted_accuracy(instancias_train, instancias_test, k_weighted, tipo_distancia)
print(f'Precisión del modelo KNN Ponderado (k={k_weighted}): {accuracy_weighted:.5f}')

import matplotlib.pyplot as plt

# Definir las distancias a utilizar
tipos_distances = ["manhattan", "euclidean", "minkowski", "chebyshev"]

# Definir una función para calcular las precisiones para diferentes valores de K y distancias
def calculate_precisions_multiple_distances(train_data, test_data, k_values, tipos_distances):
    precisions = {}
    for tipo_distancia in tipos_distances:
        precisions[tipo_distancia] = []
        for k in k_values:
            if tipo_distancia == "euclidean":
                accuracy = knn_standard_accuracy(train_data, test_data, k, tipo_distancia)
            elif tipo_distancia == "manhattan":
                accuracy = knn_standard_accuracy(train_data, test_data, k, tipo_distancia)
            elif tipo_distancia == "minkowski":
                accuracy = knn_standard_accuracy(train_data, test_data, k, tipo_distancia)
            elif tipo_distancia == "chebyshev":
                accuracy = knn_standard_accuracy(train_data, test_data, k, tipo_distancia)
            precisions[tipo_distancia].append(accuracy)
    return precisions

# Definir una función para calcular las precisiones para diferentes valores de K y distancias
def calculate_precisions_multiple_distances_knn_weighted(train_data, test_data, k_values, tipos_distances):
    precisions = {}
    for tipo_distancia in tipos_distances:
        precisions[tipo_distancia] = []
        for k in k_values:
            if tipo_distancia == "euclidean":
                accuracy = knn_weighted_accuracy(train_data, test_data, k, tipo_distancia)
            elif tipo_distancia == "manhattan":
                accuracy = knn_weighted_accuracy(train_data, test_data, k, tipo_distancia)
            elif tipo_distancia == "minkowski":
                accuracy = knn_weighted_accuracy(train_data, test_data, k, tipo_distancia)
            elif tipo_distancia == "chebyshev":
                accuracy = knn_weighted_accuracy(train_data, test_data, k, tipo_distancia)
            precisions[tipo_distancia].append(accuracy)
    return precisions

# Definir los valores de K a graficar
k_values = list(range(1, 13))

# Divide el dataset en características (X) y etiquetas (y)
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular las precisiones para los valores de K y distancias
precisions = calculate_precisions_multiple_distances_knn_weighted(instancias_train, instancias_test, k_values, tipos_distances)

def find_optimal_k_weighted(train_data, k_values, tipo_distancia):
    best_k = None
    best_accuracy = 0
    for k in k_values:
        accuracy = knn_weighted_accuracy(train_data, instancias_test, k, tipo_distancia)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    return best_k

# Encontrar el valor óptimo de K para cada distancia
best_ks = {}
for tipo_distancia in tipos_distances:
    best_ks[tipo_distancia] = find_optimal_k_weighted(instancias_train, k_values, tipo_distancia)

# Graficar los resultados de precisión para los valores de K y distancias
for tipo_distancia in tipos_distances:
    plt.plot(k_values, precisions[tipo_distancia], label=f"{tipo_distancia.capitalize()} (k={best_ks[tipo_distancia]})")
    plt.scatter(best_ks[tipo_distancia], max(precisions[tipo_distancia]), color='red')
    plt.text(best_ks[tipo_distancia], max(precisions[tipo_distancia]), f"{max(precisions[tipo_distancia]):.4f}", ha="center", va="bottom")

# Calcular las precisiones utilizando sklearn
precisions_sklearn = []
for k in range(1, 13):
    knn_sklearn = KNeighborsClassifier(n_neighbors=k)
    knn_sklearn.fit(X_train, y_train)
    y_pred_sklearn = knn_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    precisions_sklearn.append(accuracy_sklearn)

# Encontrar el valor óptimo de K utilizando sklearn
max_index = precisions_sklearn.index(max(precisions_sklearn))
best_k_sklearn = max_index + 1

# Graficar los resultados de la precisión utilizando KNN estándar + sklearn
plt.plot(k_values, precisions_sklearn, label=f"SKlearn (k={best_k_sklearn})")
plt.scatter(best_k_sklearn, max(precisions_sklearn), color='red')
plt.text(best_k_sklearn, max(precisions_sklearn), f"{max(precisions_sklearn):.4f}", ha="center", va="bottom")

# Configurar las etiquetas y título del gráfico
plt.xlabel('Valor de K')
plt.ylabel('Precisión')
plt.title('Precisión en función del valor de K y distancia')

# Mostrar la leyenda y el gráfico
plt.legend()
plt.show()

# Implementar el algoritmo de knn optimo sin utilizar librerías que no sean nativas de python
import numpy as np
import pandas as pd
import math
import warnings

# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv("dataset_cleaned_fixed.csv")

# Crea una clase para representar cada instancia
class Instancia:
    def __init__(self, id, atributos, clase):
        self.id = id
        self.atributos = atributos
        self.clase = clase

# Cargar los datos desde archivos CSV y realizar la división en conjuntos de entrenamiento y prueba

# Asegúrate de que los datos estén en el formato correcto
# instancias_train y instancias_test deben ser listas de instancias de la clase Instancia

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

# Definir una función para predecir la clase de una instancia usando KNN estándar
def predict_knn_standard(train_data, test_instance, k):
    distances = [(calcular_distancia_manhattan(train_instance, test_instance), train_instance.clase) for train_instance in train_data]
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

# Definir una función para predecir la clase de una instancia usando KNN ponderado
def predict_knn_weighted(train_data, test_instance, k):
    distances = [(calcular_distancia_manhattan(train_instance, test_instance), train_instance.clase) for train_instance in train_data]
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    class_weights = {}
    for neighbor in neighbors:
        distance, class_label = neighbor
        if distance == 0:
            return class_label  # No ponderar si la distancia es 0
        weight = 1 / distance
        if class_label in class_weights:
            class_weights[class_label] += weight
        else:
            class_weights[class_label] = weight
    return max(class_weights, key=class_weights.get)

# Entrenar y probar el modelo KNN estándar


def knn_standard_accuracy(train_data, test_data, k):
    correct = 0
    for test_instance in test_data:
        predicted_class = predict_knn_standard(train_data, test_instance, k)
        if predicted_class == test_instance.clase:
            correct += 1
    return correct / len(test_data)

k_standard = 11  # Puedes ajustar este valor
accuracy_standard = knn_standard_accuracy(instancias_train, instancias_test, k_standard)
print(f'Precisión del modelo KNN Estándar (k={k_standard}): {accuracy_standard:.5f}')

# Encontrar el valor óptimo de K para KNN estándar
k_values = [1, 3, 5, 7, 9, 11, 13]  # Puedes ajustar estos valores


def find_optimal_k(train_data, k_values):
    best_k = None
    best_accuracy = 0
    for k in k_values:
        accuracy = knn_standard_accuracy(train_data, instancias_test, k)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    return best_k

best_k_standard = find_optimal_k(instancias_train, k_values)


print(f'Valor óptimo de K para KNN Estándar: {best_k_standard}')

# Entrenar y probar el modelo KNN ponderado
k_weighted = 11  # Puedes ajustar este valor

def knn_weighted_accuracy(train_data, test_data, k):
    correct = 0
    for test_instance in test_data:
        predicted_class = predict_knn_weighted(train_data, test_instance, k)
        if predicted_class == test_instance.clase:
            correct += 1
    return correct / len(test_data)

accuracy_weighted = knn_weighted_accuracy(instancias_train, instancias_test, k_weighted)
print(f'Precisión del modelo KNN Ponderado (k={k_weighted}): {accuracy_weighted:.5f}')

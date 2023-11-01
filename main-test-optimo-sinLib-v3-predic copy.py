# Implementar el algoritmo de knn optimo sin utilizar librerías que no sean nativas de python
import numpy as np
import pandas as pd
import warnings

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
test_data = pd.read_csv('test_data_v2.csv')

# Crea una lista para almacenar las instancias del dataset de entrenamiento
instancias_train = []


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
    
    # Calcula las distancias entre la instancia actual y todas las instancias de entrenamiento utilizando las cuatro funciones de distancia
    distances_euclidean = []
    distances_manhattan = []
    distances_minkowski = []
    distances_chebyshev = []
    
    for instancia_train in instancias_train:
        distance_euclidean = euclidean_distance(instancia_train, Instancia(0, atributos, 0))
        distance_manhattan = calcular_distancia_manhattan(instancia_train, Instancia(0, atributos, 0))
        distance_minkowski = minkowski_distance(instancia_train.atributos, atributos, 3)
        distance_chebyshev = calcular_distancia_chebyshev(instancia_train, Instancia(0, atributos, 0))
        
        distances_euclidean.append((instancia_train, distance_euclidean))
        distances_manhattan.append((instancia_train, distance_manhattan))
        distances_minkowski.append((instancia_train, distance_minkowski))
        distances_chebyshev.append((instancia_train, distance_chebyshev))

    # Ordena las distancias por orden ascendente para obtener los K vecinos más cercanos
    k = 2
    
    neighbors_euclidean = sorted(distances_euclidean, key=lambda x: x[1])[:k]
    neighbors_manhattan = sorted(distances_manhattan, key=lambda x: x[1])[:k]
    neighbors_minkowski = sorted(distances_minkowski, key=lambda x: x[1])[:k]
    neighbors_chebyshev = sorted(distances_chebyshev, key=lambda x: x[1])[:k]

    # Realiza la predicción de la clase utilizando KNN estándar y KNN ponderado con las cuatro funciones de distancia
    prediction_euclidean_standard = max(set([neighbor[0].clase for neighbor in neighbors_euclidean]), key=[neighbor[0].clase for neighbor in neighbors_euclidean].count)
    prediction_manhattan_standard = max(set([neighbor[0].clase for neighbor in neighbors_manhattan]), key=[neighbor[0].clase for neighbor in neighbors_manhattan].count)
    prediction_minkowski_standard = max(set([neighbor[0].clase for neighbor in neighbors_minkowski]), key=[neighbor[0].clase for neighbor in neighbors_minkowski].count)
    prediction_chebyshev_standard = max(set([neighbor[0].clase for neighbor in neighbors_chebyshev]), key=[neighbor[0].clase for neighbor in neighbors_chebyshev].count)

    prediction_euclidean_weighted = sum([neighbor[0].clase / neighbor[1] for neighbor in neighbors_euclidean]) / sum([1 / neighbor[1] for neighbor in neighbors_euclidean])
    prediction_manhattan_weighted = sum([neighbor[0].clase / neighbor[1] for neighbor in neighbors_manhattan]) / sum([1 / neighbor[1] for neighbor in neighbors_manhattan])
    prediction_minkowski_weighted = sum([neighbor[0].clase / neighbor[1] for neighbor in neighbors_minkowski]) / sum([1 / neighbor[1] for neighbor in neighbors_minkowski])
    prediction_chebyshev_weighted = sum([neighbor[0].clase / neighbor[1] for neighbor in neighbors_chebyshev]) / sum([1 / neighbor[1] for neighbor in neighbors_chebyshev])

    # Imprime la predicción de la instancia actual utilizando KNN estándar y KNN ponderado con las cuatro funciones de distancia
    print(f"Instancia {id_instancia}:")
    print(f"Predicción Euclidean Standard: {prediction_euclidean_standard}")
    print(f"Predicción Manhattan Standard: {prediction_manhattan_standard}")
    print(f"Predicción Minkowski Standard: {prediction_minkowski_standard}")
    print(f"Predicción Chebyshev Standard: {prediction_chebyshev_standard}")
    
    print(f"Predicción Euclidean Weighted: {prediction_euclidean_weighted}")
    print(f"Predicción Manhattan Weighted: {prediction_manhattan_weighted}")
    print(f"Predicción Minkowski Weighted: {prediction_minkowski_weighted}")
    print(f"Predicción Chebyshev Weighted: {prediction_chebyshev_weighted}")

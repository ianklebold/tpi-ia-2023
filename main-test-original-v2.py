# Authors:
#   Kalbermatter, Alan
#   Fernandez, Ian
#   Hadad Menegaz, Jorge Miguel

import pandas as pd
import math
import warnings

# Lee el dataset de entrenamiento desde un archivo CSV
train_data = pd.read_csv('train_data.csv')

# Lee el dataset de prueba desde un archivo CSV
test_data = pd.read_csv('test_data.csv')

def rename_columns_dataset(dataset):
    dataset.rename(columns={
        # 'id': 'ID',
        'age': 'Edad',
        'education': 'Educación',
        'sex': 'Sexo',
        'is_smoking': 'Fumador',
        'cigsPerDay': 'Cigarrillos por Día',
        'BPMeds': 'Toma Medicamentos Pre ART',
        'prevalentStroke': 'Derrame Cerebral',
        'prevalentHyp': 'Hipertenso',
        'diabetes': 'Diabetes',
        'totChol': 'Colesterol Total',
        'sysBP': 'Presión Arterial Sistólica',
        'diaBP': 'Presión Arterial Diastólica',
        'BMI': 'IMC',
        'heartRate': 'Frecuencia Cardíaca',
        'glucose': 'Nivel de Glucosa',
        'TenYearCHD': 'Predicción'
    }, inplace=True)

# Renombrar las columnas
rename_columns_dataset(train_data)
rename_columns_dataset(test_data)

# Normalización min-max
def min_max_normalize(data):
    normalized_data = data.copy()
    for feature in data.columns:
        max_value = data[feature].max()
        min_value = data[feature].min()
        normalized_data[feature] = (data[feature] - min_value) / (max_value - min_value)
    return normalized_data


# Crea una clase para representar cada instancia
class Instancia:
    def __init__(self, id, atributos, clase):
        self.id = id
        self.atributos = atributos
        self.clase = clase


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

#Mostrar detalles de los vecinos encontrados por instancia en cada K
def mostrar_vecinos(vecinos_mas_cercanos, k):
    df_resultados = pd.DataFrame({
        'ID': [vecino.id for vecino in vecinos_mas_cercanos],
        'Edad': [vecino.atributos[0] for vecino in vecinos_mas_cercanos],
        'Educación': [vecino.atributos[1] for vecino in vecinos_mas_cercanos],
        'Sexo': [vecino.atributos[2] for vecino in vecinos_mas_cercanos],
        'Fumador': [vecino.atributos[3] for vecino in vecinos_mas_cercanos],
        'Cigarrillos por Día': [vecino.atributos[4] for vecino in vecinos_mas_cercanos],
        'Toma Medicamentos Pre ART': [vecino.atributos[5] for vecino in vecinos_mas_cercanos],
        'Derrame Cerebral': [vecino.atributos[6] for vecino in vecinos_mas_cercanos],
        'Hipertenso': [vecino.atributos[7] for vecino in vecinos_mas_cercanos],
        'Diabetes': [vecino.atributos[8] for vecino in vecinos_mas_cercanos],
        'Colesterol Total': [vecino.atributos[9] for vecino in vecinos_mas_cercanos],
        'Presión Arterial Sistólica': [vecino.atributos[10] for vecino in vecinos_mas_cercanos],
        'Presión Arterial Diastólica': [vecino.atributos[11] for vecino in vecinos_mas_cercanos],
        'IMC': [vecino.atributos[12] for vecino in vecinos_mas_cercanos],
        'Frecuencia Cardíaca': [vecino.atributos[13] for vecino in vecinos_mas_cercanos],
        'Nivel de Glucosa': [vecino.atributos[14] for vecino in vecinos_mas_cercanos],
        'Predicción': [vecino.clase for vecino in vecinos_mas_cercanos],
        'Distancia': [vecino.distancia for vecino in vecinos_mas_cercanos]
    })

    # Imprime el DataFrame con los resultados
    print("------------------------------------\n")
    print("Resultados para k = ", k)
    print(df_resultados)
    print("------------------------------------\n")


# Función para calcular la distancia entre dos instancias
def calcular_distancia(instancia, instancia_nueva):
    distancia = 0
    for i in range(len(instancia.atributos)):
        distancia += (instancia.atributos[i] - instancia_nueva.atributos[i]) ** 2
    return math.sqrt(distancia)


#Realizar el calculo de la distancia Manhattan
def calcular_distancia_manhattan(instancia, instancia_nueva):
    distancia = 0
    for i in range(len(instancia.atributos)):
        distancia += abs(instancia.atributos[i] - instancia_nueva.atributos[i])
    return distancia


def minkowski_distance(x1, x2, p):
    distance = 0
    for i in range(len(x1)):
        distance += abs(x1[i] - x2[i]) ** p

    # Deshabilitar la advertencia específica (overflow)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in double_scalars")
        result = distance ** (1/p)
    
    return result


# Función para encontrar los k vecinos más cercanos utilizando la distancia de Minkowski
def find_nearest_neighbors_minkowski(test_instance, train_data, train_labels, k, p):
    distances = []
    for i in range(len(train_data)):
        dist = minkowski_distance(test_instance, train_data[i], p)
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return neighbors

##########################################################################
##########################################################################
##########################################################################


if __name__ == "__main__":
    # Mostrar al usuario las 3 opciones de distancia disponibles para calcular la distancia entre instancias
    print("Seleccione la distancia a utilizar para calcular la distancia entre instancias:")
    print("1. Distancia Euclidiana")
    print("2. Distancia Manhattan")
    print("3. Distancia de Minkowski")
    opcion = int(input("Opción: "))
    if opcion == 1:
        tipo_calculo = "calcular_distancia_euclidiana"
    elif opcion == 2:
        tipo_calculo = "calcular_distancia_manhattan"
    elif opcion == 3:
        tipo_calculo = "calcular_distancia_minkowski"    
    else:
        print("Opción inválida.")
        exit()
    print("Distancia seleccionada:", tipo_calculo)
    
    
##########################################################################
# Clasificación del conjunto de prueba utilizando la distancia euclidiana
##########################################################################

# Función para calcular la distancia euclidiana entre dos instancias
def euclidean_distance(test_instance, instance):
    # Inicializar la distancia
    distance = 0
    # Iterar para cada atributo de la instancia
    for i, attribute in enumerate(instance):
        # Sumar la diferencia de cada atributo al cuadrado
        distance += pow(test_instance[i] - attribute, 2)
    # Calcular la raíz cuadrada de la distancia
    return math.sqrt(distance)

# Función para encontrar los k vecinos más cercanos utilizando la distancia euclidiana
def find_nearest_neighbors_euclidean(test_instance, train_data, train_labels, k):
    # Lista para almacenar las distancias
    distances = []
    # Iterar para cada instancia de entrenamiento
    for i, instance in enumerate(train_data):
        # Calcular la distancia euclidiana entre la instancia de prueba y la instancia de entrenamiento
        distance = euclidean_distance(test_instance, instance)
        # Almacenar la distancia y el índice de la instancia de entrenamiento
        distances.append((i, distance))
    # Ordenar las distancias
    distances.sort(key=lambda x: x[1])
    # Obtener los k vecinos más cercanos
    neighbors = distances[:k]
    # Obtener las etiquetas de los vecinos más cercanos
    labels = [train_labels[neighbor[0]] for neighbor in neighbors]
    return labels

if tipo_calculo == 'calcular_distancia_euclidiana':
    # Lista para almacenar las predicciones
    predictions = []
    k = 24  # Número de vecinos a considerar

    for i in range(len(test_data)):
        test_instance = test_data.iloc[i].values[:-1]
        neighbors = find_nearest_neighbors_euclidean(test_instance, train_data.values[:, :-1], train_data.values[:, -1], k)
        labels = [neighbor for neighbor in neighbors]
        prediction = max(set(labels), key=labels.count)
        predictions.append(prediction)
        
    # Cálculo de la precisión del modelo
    correct_predictions = 0
    for i, prediction in enumerate(predictions):
        if prediction == test_data.values[i][-1]:
            correct_predictions += 1
    accuracy = correct_predictions / len(predictions)
    print("Precisión del modelo: {:.2f}%".format(accuracy * 100))
        
##########################################################################
##########################################################################
##########################################################################
            
##########################################################################
# Clasificación del conjunto de prueba utilizando la distancia de Minkowski
##########################################################################

if tipo_calculo == 'calcular_distancia_minkowski':
    # Lista para almacenar las predicciones
    predictions = []
    k = 24  # Número de vecinos a considerar
    p = 0.0001  # Valor de p para la distancia de Minkowski (buen numero)
    # p = 1000 # Valor de p para la distancia de Minkowski (buen numero)

    for i in range(len(test_data)):
        test_instance = test_data.iloc[i].values[:-1]
        neighbors = find_nearest_neighbors_minkowski(test_instance, train_data.values[:, :-1], train_data.values[:, -1], k, p)
        labels = [neighbor[1] for neighbor in neighbors]
        prediction = max(set(labels), key=labels.count)
        predictions.append(prediction)

    # Cálculo de la precisión del modelo
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == test_data.iloc[i][-1]:
            correct_predictions += 1
    accuracy = correct_predictions / len(predictions)
    print("Precisión del modelo: {:.2f}%".format(accuracy * 100))

########################################################################
########################################################################
########################################################################

########################################################################
# Clasificación del conjunto de prueba utilizando la distancia Manhattan
########################################################################

# Función para calcular la distancia Manhattan entre dos instancias
def calcular_distancia_manhattan(train_data, test_instance):
    # Inicializa la distancia en 0
    distance = 0
    # Calcula la distancia Manhattan entre la instancia de prueba y la instancia actual del conjunto de entrenamiento
    for i in range(len(test_instance)):
        distance += abs(test_instance[i] - train_data[i])
    # Retorna la distancia Manhattan entre la instancia de prueba y la instancia actual del conjunto de entrenamiento
    return distance

# Función para encontrar los k vecinos más cercanos utilizando la distancia Manhattan
def find_nearest_neighbors_manhattan(test_instance, train_data, train_labels, k):
    # Lista para almacenar los vecinos más cercanos
    neighbors = []
    for i in range(len(train_data)):
        # Calcula la distancia Manhattan entre la instancia de prueba y la instancia actual del conjunto de entrenamiento
        distance = calcular_distancia_manhattan(train_data[i], test_instance)
        # Agrega la instancia actual del conjunto de entrenamiento y su distancia a la lista de vecinos
        neighbors.append((train_data[i], train_labels[i], distance))
    # Ordena la lista de vecinos por distancia
    neighbors.sort(key=lambda x: x[2])
    # Retorna los k vecinos más cercanos
    return neighbors[:k]

if tipo_calculo == "calcular_distancia_manhattan":
    # Lista para almacenar las predicciones
    predictions = []
    k = 11  # Número de vecinos a considerar

    for i in range(len(test_data)):
        test_instance = test_data.iloc[i].values[:-1]
        neighbors = find_nearest_neighbors_manhattan(test_instance, train_data.values[:, :-1], train_data.values[:, -1], k)
        labels = [neighbor[1] for neighbor in neighbors]
        prediction = max(set(labels), key=labels.count)
        predictions.append(prediction)

    # Cálculo de la precisión del modelo
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == test_data.iloc[i][-1]:
            correct_predictions += 1
    accuracy = correct_predictions / len(predictions)
    print("Precisión del modelo: {:.2f}%".format(accuracy * 100))
    

# Authors:
#   Kalbermatter, Alan
#   Fernandez, Ian
#   Hadad Menegaz, Jorge Miguel

import pandas as pd
import math
import random


# Lee el dataset desde un archivo CSV
# dataset = pd.read_csv('resources/dataset_cleaned_fixed.csv')

# Lee el dataset de entrenamiento desde un archivo CSV
train_data = pd.read_csv('train_data.csv')

# Lee el dataset de prueba desde un archivo CSV
test_data = pd.read_csv('test_data.csv')

# Renombrar las columnas
train_data.rename(columns={
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

test_data.rename(columns={
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

# Normalización min-max
def min_max_normalize(data):
    normalized_data = data.copy()
    for feature in data.columns:
        max_value = data[feature].max()
        min_value = data[feature].min()
        normalized_data[feature] = (data[feature] - min_value) / (max_value - min_value)
    return normalized_data

# Aplicar one-hot encoding a las variables categóricas
train_data = pd.get_dummies(train_data, columns=['Sexo', 'Fumador'])
test_data = pd.get_dummies(test_data, columns=['Sexo', 'Fumador'])

# # Normalizar los datos de entrenamiento
# train_data = min_max_normalize(train_data)

# # Normalizar los datos de prueba
# test_data = min_max_normalize(test_data)


# Crea una clase para representar cada instancia
class Instancia:
    def __init__(self, id, atributos, clase):
        self.id = id
        self.atributos = atributos
        self.clase = clase

# Crea una lista para almacenar las instancias
instancias = []

# # Itera sobre las filas del dataset
# for index, row in dataset.iterrows():
#     # Obtiene los atributos de la instancia y los convierte a tipo 'float'
#     atributos = [float(value) for value in row[1:-1]]
#     # Obtiene el ID de la instancia
#     id_instancia = int(row['id'])
#     # Obtiene la clase de la instancia
#     clase = row[-1]
#     # Crea una instancia y la agrega a la lista
#     instancia = Instancia(id_instancia, atributos, clase)
#     instancias.append(instancia)

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

#https://www.sciencedirect.com/topics/mathematics/manhattan-distance#:~:text=1)%20Manhattan%20Distance%20%3D%20%7C%20x,in%20a%20higher-dimensional%20space
#Realizar el calculo de la distancia Manhattan
def calcular_distancia_manhattan(instancia, instancia_nueva):
    distancia = 0
    for i in range(len(instancia.atributos)):
        distancia += abs(instancia.atributos[i] - instancia_nueva.atributos[i])
    return distancia

# Función para calcular la distancia de Minkowski
def minkowski_distance(x1, x2, p):
    distance = 0
    for i in range(len(x1)):
        distance += abs(x1[i] - x2[i]) ** p
    return distance ** (1/p)

# Función para encontrar los k vecinos más cercanos utilizando la distancia de Minkowski
def find_nearest_neighbors_minkowski(test_instance, train_data, train_labels, k, p):
    distances = []
    for i in range(len(train_data)):
        dist = minkowski_distance(test_instance, train_data[i], p)
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return neighbors

# Función para encontrar los k vecinos más cercanos a una instancia nueva
def encontrar_vecinos_mas_cercanos(instancias, instancia_nueva, k):
    distancias = []
    for instancia in instancias:
        distancia = calcular_distancia(instancia, instancia_nueva)
        distancias.append((instancia, distancia))
    distancias.sort(key=lambda x: x[1])
    vecinos = [instancia[0] for instancia in distancias[:k]]
    return vecinos

# Ejemplo de uso

# Obtener 6 instancias aleatorias del dataset
instancias_random_test = random.sample(instancias_test, 8)

# Lista para almacenar los resultados de las instancias aleatorias
resultados_random = []

# Lista para almacenar los resultados de las coincidencias
resultados = []

# Lista para almacenar los vecinos más cercanos por cada k para cada instancia
vecinos_por_k = []

# Lista para almacenar las coincidencias por cada k para cada instancia
coincidencias_por_k = []

cantidad_de_vecinos = 11

# # Iterar para cada instancia aleatoria
# for i, instancia_nueva in enumerate(instancias_test, start=1):
#     # Lista para almacenar los resultados para esta instancia aleatoria
#     resultados_instancia = []

#     # Lista para almacenar los vecinos más cercanos para cada k
#     vecinos_k_actual = []
#     coincidencias_k_actual = []

#     # Iterar para k de 1 a 5
#     for k in range(1, cantidad_de_vecinos + 1):
#         # Encuentra los vecinos más cercanos
#         vecinos_mas_cercanos = encontrar_vecinos_mas_cercanos(instancias_train, instancia_nueva, k)

#         # Calcula la distancia para cada vecino
#         for vecino in vecinos_mas_cercanos:
#             vecino.distancia = calcular_distancia_manhattan(vecino, instancia_nueva)

#         # Ordena los vecinos por distancia
#         vecinos_mas_cercanos.sort(key=lambda x: x.distancia)

#         # Crea un DataFrame para mostrar los resultados
#         # mostrar_vecinos(vecinos_mas_cercanos, k)

#         # Almacena el ID del último vecino más cercano para esta instancia y k
#         vecinos_k_actual.append(vecinos_mas_cercanos[-1].id)

#         # Almacena la cantidad de coincidencias
#         coincidencias_k_actual.append(sum(1 for vecino in vecinos_mas_cercanos if vecino.clase == instancia_nueva.clase))

#         # Cuenta cuántas predicciones coinciden con el valor de la clase
#         coincidencias = sum(1 for vecino in vecinos_mas_cercanos if vecino.clase == instancia_nueva.clase)

#         # Almacena los resultados
#         resultados_instancia.append({'K': f'k={k}', 'Coincidencias': coincidencias})

#     vecinos_por_k.append([f'k={i}'] + vecinos_k_actual)
#     coincidencias_por_k.append([f'k={i}'] + coincidencias_k_actual)

#     # Crea un DataFrame para mostrar los resultados de coincidencias con la clase para esta instancia
#     df_coincidencias_instancia = pd.DataFrame(resultados_instancia)
#     print(f"\nResultados para la instancia i{i}:")
#     print(df_coincidencias_instancia)
    
#     # Calcula la precisión del modelo para esta instancia
#     precision = sum(1 for vecino in vecinos_mas_cercanos if vecino.clase == instancia_nueva.clase) / k
#     resultados_random.append({'i': f'i{i}', 'Precisión': precision})

# # Crea un DataFrame para mostrar los resultados de coincidencias con la clase para todas las instancias
# df_resultados_random = pd.DataFrame(resultados_random)
# print("\nResultados de precisión para todas las instancias:")
# print(df_resultados_random)

# # Calcula la precisión promedio del modelo
# precision_promedio = df_resultados_random['Precisión'].mean()
# print("Precisión promedio del modelo:", precision_promedio)

##########################################################################
# Clasificación del conjunto de prueba utilizando la distancia de Minkowski
##########################################################################

predictions = []
k = 5  # Número de vecinos a considerar
p = 4  # Valor de p para la distancia de Minkowski

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
    

# # Ajuste de columnas para el DataFrame de vecinos por k
# columnas = ['K'] + [f'i{i}' for i in range(1, cantidad_de_vecinos + 1)]
# # Añadimos una lista vacía para ajustar las columnas
# vecinos_por_k.append([''] * len(columnas))
# coincidencias_por_k.append([''] * len(columnas))

# # Crear el DataFrame para mostrar los vecinos por cada k
# df_vecinos_por_k = pd.DataFrame(vecinos_por_k, columns=columnas)
# # Eliminar la última fila del DataFrame df_vecinos_por_k
# df_vecinos_por_k = df_vecinos_por_k.drop(df_vecinos_por_k.index[-1])

# # Crear el DataFrame para mostrar las coincidencias por cada k
# df_coincidencias_por_k = pd.DataFrame(coincidencias_por_k, columns=columnas)
# # Eliminar la última fila del DataFrame df_coincidencias_por_k
# df_coincidencias_por_k = df_coincidencias_por_k.drop(df_coincidencias_por_k.index[-1])

# print("\nVecinos más cercanos por cada k ANTES TEST para cada instancia:")
# print(df_vecinos_por_k.shape)
# print(df_vecinos_por_k.head())
# print("\nCoincidencias por cada k ANTES TEST para cada instancia:")
# print(df_coincidencias_por_k.shape)
# print(df_coincidencias_por_k.head())

# # Verificar si el DataFrame tiene la misma cantidad de filas que de columnas
# if df_coincidencias_por_k.shape[0] + 1 != df_coincidencias_por_k.shape[1]:
#     # Calculate the difference between the number of rows and columns
#     diff = abs(df_coincidencias_por_k.shape[0] - df_coincidencias_por_k.shape[1]) - 1
    
#     # Add rows or columns as needed
#     if df_coincidencias_por_k.shape[0] + 1 < df_coincidencias_por_k.shape[1]:
#         for i in range(diff):
#             df_coincidencias_por_k = df_coincidencias_por_k.append([pd.Series([None] * len(df_coincidencias_por_k.columns))], ignore_index=True)
#     elif df_coincidencias_por_k.shape[0] + 1 > df_coincidencias_por_k.shape[1]:
#         for i in range(diff):
#             df_coincidencias_por_k[len(df_coincidencias_por_k.columns)] = [None] * len(df_coincidencias_por_k)
        
#     print("\nCoincidencias por cada k TEST para cada instancia:")
#     print(df_coincidencias_por_k)
    
# if df_vecinos_por_k.shape[0] + 1 != df_vecinos_por_k.shape[1]:
#     # Calculate the difference between the number of rows and columns
#     diff = abs(df_vecinos_por_k.shape[0] - df_vecinos_por_k.shape[1]) - 1
    
#     # Add rows or columns as needed
#     if df_vecinos_por_k.shape[0] + 1 < df_vecinos_por_k.shape[1]:
#         print("La cantidad de columnas es mayor que la cantidad de filas")
#         for i in range(diff):
#             df_vecinos_por_k = df_vecinos_por_k.append([pd.Series([None] * len(df_vecinos_por_k.columns))], ignore_index=True)
#     elif df_vecinos_por_k.shape[0] + 1 > df_vecinos_por_k.shape[1]:
#         print("La cantidad de filas es mayor que la cantidad de columnas")
#         for i in range(diff):
#             df_vecinos_por_k[len(df_vecinos_por_k.columns)] = [None] * (len(df_vecinos_por_k) + 1)
    
#     print("\nVecinos más cercanos  TEST por cada k para cada instancia:")
#     print(df_vecinos_por_k)
    
#     # Rellenar los valores faltantes con None o NaN
#     # df_vecinos_por_k.fillna(0, inplace=True)
#     # df_coincidencias_por_k.fillna(0, inplace=True)


# # Transponer solo el contenido debajo de las columnas de las ix
# print("\nVecinos más cercanos por cada k DESPUES TEST para cada instancia:")
# #mostrar cantidad de filas y columnas
# print(df_vecinos_por_k.shape)
# print(df_coincidencias_por_k.shape)
# df_vecinos_por_k.iloc[:, 1:] = df_vecinos_por_k.iloc[:, 1:].T
# df_coincidencias_por_k.iloc[:, 1:] = df_coincidencias_por_k.iloc[:, 1:].T

# # Imprimir el DataFrame con los resultados de vecinos por k
# print("\nVecinos más cercanos por cada k para cada instancia:")
# print(df_vecinos_por_k)

# # Imprimir el DataFrame con los resultados de coincidencias por k
# print("\nCoincidencias por cada k para cada instancia:")
# print(df_coincidencias_por_k)

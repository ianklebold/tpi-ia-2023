import numpy as np
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt

# Paso 1: Definir una función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Paso 2: Definir la función KNN
def knn(train_data, test_instance, k):
    distances = []

    # Calcular la distancia entre la instancia de prueba y todos los puntos de entrenamiento
    for i, train_instance in enumerate(train_data):
        dist = euclidean_distance(test_instance, train_instance[:-1])  # Excluye la etiqueta
        distances.append((i, dist))

    # Ordenar las distancias y obtener los índices de los k vecinos más cercanos
    distances.sort(key=lambda x: x[1])
    nearest_neighbors_indices = [x[0] for x in distances[:k]]

    # Obtener las etiquetas de los k vecinos más cercanos
    nearest_labels = [train_data[i][-1] for i in nearest_neighbors_indices]

    # Realizar la predicción tomando la etiqueta más común entre los vecinos más cercanos
    prediction = max(set(nearest_labels), key=nearest_labels.count)

    return prediction

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrenamiento (cada fila contiene [edad, presion, colesterol, etiqueta])
    train_data = np.array([[25, 120, 200, 0],  # 0 representa 'sano'
                           [30, 140, 240, 1],  # 1 representa 'enfermo'
                           [35, 130, 220, 0],
                           [45, 150, 250, 1]])

    # Datos de prueba
    test_instance = np.array([40, 135, 230])

    # Número de vecinos a considerar
    k = 3

    # Realizar la predicción
    prediction = knn(train_data, test_instance, k)

    # Mapear la predicción de nuevo a etiquetas
    labels = ['sano', 'enfermo']
    predicted_label = labels[prediction]

    print(f"La predicción para la instancia de prueba es: {predicted_label}")

# Cargar datos desde un archivo CSV
data = pd.read_csv('sample_data/data_cardiovascular_risk.csv')  # Reemplaza 'tudataset.csv' con el nombre de tu archivo CSV

# Puedes imprimir una vista previa de los datos para verificar que se hayan cargado correctamente
print(data.head())

# Extraer las columnas que deseas graficar
id = data['id']
age = data['age']
education = data['education']
sex = data['sex']
is_smoking = data['is_smoking']
cigsPerDay = data['cigsPerDay']
BPMeds = data['BPMeds']
prevalentStroke = data['prevalentStroke']
prevalentHyp = data['prevalentHyp']
diabetes = data['diabetes']
totChol = data['totChol']
sysBP = data['sysBP']
diaBP = data['diaBP']
BMI = data['BMI']
heartRate = data['heartRate']
glucose = data['glucose']
TenYearCHD = data['TenYearCHD']

df = pd.read_csv('/content/data_cardiovascular_risk.csv')

# reemplaza los valores faltantes con el promedio de la columna correspondiente
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# convierte las columnas 'education', 'prevalentStroke', 'prevalentHyp', 'diabetes', y 'TenYearCHD' a tipo int
df[['education', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD', 'cigsPerDay', 'BPMeds', 'heartRate', 'totChol', 'glucose']] = df[['education', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD', 'cigsPerDay', 'BPMeds', 'heartRate', 'totChol', 'glucose']].astype(int)

# escribe los datos de vuelta al archivo csv
df.to_csv('dataset_cleaned.csv', index=False)

df[df['TenYearCHD'] == 1]

df[df['TenYearCHD'] == 0]


# Función para limpiar los datos reemplazando None por 0
def clean_data(data):
    cleaned_data = []
    for row in data:
        cleaned_row = [0 if val is None else val for val in row]
        cleaned_data.append(cleaned_row)
    return cleaned_data

# Función para calcular la distancia euclidiana entre dos puntos, manejando valores faltantes
def euclidean_distance(point1, point2):
    # Asegurarse de que ambas listas tengan la misma longitud
    max_length = max(len(point1), len(point2))
    point1 = point1 + [0] * (max_length - len(point1))
    point2 = point2 + [0] * (max_length - len(point2))

    squared_distance = 0
    for i in range(max_length):
        squared_distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(squared_distance)

# Función para encontrar los k vecinos más cercanos
def get_neighbors(train_data, test_instance, k):
    distances = []
    for i, train_instance in enumerate(train_data):
        dist = euclidean_distance(test_instance, train_instance[:-1])  # Excluye la etiqueta
        distances.append((i, dist))
    distances.sort(key=lambda x: x[1])
    print(len(test_instance))
    print(train_data[i][len(test_instance)-1])

    return [train_data[i][len(test_instance)-1] for i, _ in distances[:k]]

# Función para realizar la predicción
def predict(train_data, test_instance, k):
    neighbors = get_neighbors(train_data, test_instance, k)
    # Realizar la predicción tomando la etiqueta más común entre los vecinos más cercanos
    prediction = max(set(neighbors), key=neighbors.count)
    return prediction

# Funcion para codificar el atributo 'sex'
def encode_sex(sex):
    if sex == 'M':
      return 0
    elif sex == 'F':
      return 1
    else:
      return -1

# Funcion para codificar el atributo 'is_smoking'
def encode_is_smoking(is_smoking):
    if is_smoking == "YES":
      return 1
    elif is_smoking == "NO":
      return 0
    else:
      return -1

# Cargar los datos desde tu archivo CSV
data = []
with open('sample_data/data_cardiovascular_risk.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    headers = next(csv_reader)  # Leer la primera fila que contiene los encabezados
    for row in csv_reader:
        # Verificar y manejar valores faltantes (puedes personalizar este manejo)
        for i in range(len(row)):
            if row[i] == '':
                row[i] = None
            elif i == 3:  # Índice del atributo 'sex'
                row[i] = encode_sex(row[i])
            elif i == 4:  # Índice del atributo 'is_smoking'
                row[i] = encode_is_smoking(row[i])
            else:
                try:
                    row[i] = float(row[i])
                except ValueError:
                    row[i] = None  # Manejo de valores no numéricos

        data.append(row)

# Llamar a clean_data para limpiar los datos antes de continuar.
data = clean_data(data)

# Dividir los datos en conjuntos de entrenamiento y prueba (puedes personalizar el split)
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

# Número de vecinos a considerar
k = 3

# Realizar predicciones en el conjunto de prueba
predictions = []
for test_instance in test_data:
    prediction = predict(train_data, test_instance, k)
    predictions.append(prediction)

# Evaluar el rendimiento del modelo y otros calculos...






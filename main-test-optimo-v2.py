import pandas as pd
import math
from sklearn.model_selection import train_test_split

# Lee el dataset desde un archivo CSV
dataset = pd.read_csv('resources/dataset_cleaned_fixed.csv')

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

# Divide el dataset en características (X) y etiquetas (y)
id_column = dataset.iloc[:, 0]
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Función para calcular la distancia Euclidiana entre dos instancias
def euclidean_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)):
        distance += (instance1[i] - instance2[i]) ** 2
    return math.sqrt(distance)

# Clase para representar una instancia
class Instance:
    def __init__(self, attributes, label):
        self.attributes = attributes
        self.label = label
        

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

# Función para encontrar los k vecinos más cercanos
def find_nearest_neighbors(train_data, test_instance, k):
    distances = []

    for train_instance in train_data:
        distance = euclidean_distance(train_instance.attributes, test_instance.attributes)
        distances.append((train_instance, distance))

    distances.sort(key=lambda x: x[1])
    neighbors = [x[0] for x in distances[:k]]
    return neighbors

# Función para calcular la precisión
def calculate_accuracy(predictions, true_labels):
    correct_predictions = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    return correct_predictions / len(true_labels)

# Rango de valores de k a probar
k_values = range(1, 21)

best_k = None
best_accuracy = 0

# Realizar la búsqueda de hiperparámetros
for k in k_values:
    predictions = []

    for test_instance in X_val.values:
        neighbors = find_nearest_neighbors(X_train.values, test_instance, k)
        # Realiza una predicción basada en los vecinos más cercanos
        # Puedes utilizar la moda de las etiquetas de los vecinos como predicción
        prediction = max(set(neighbor.label for neighbor in neighbors), key=(neighbor.label for neighbor in neighbors).count)
        predictions.append(prediction)

    accuracy = calculate_accuracy(predictions, y_val)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("El valor óptimo de k es:", best_k)

# Entrenar el modelo final con el valor óptimo de k en el conjunto de entrenamiento
# ...

# Opcional: Evaluar el modelo en el conjunto de prueba
# ...

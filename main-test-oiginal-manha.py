import pandas as pd
import numpy as np

# Función para calcular la distancia Manhattan entre dos puntos
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2), axis=1)

# Función para normalizar los datos utilizando la técnica de min-max scaling
def min_max_normalize(data):
    min_values = data.min()
    max_values = data.max()
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data

# Lee el dataset de entrenamiento desde un archivo CSV
train_data = pd.read_csv('train_data.csv')

# Lee el dataset de prueba desde un archivo CSV
test_data = pd.read_csv('test_data.csv')

# Renombrar las columnas
train_data.rename(columns={
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

# Seleccionar las columnas relevantes para el algoritmo KNN
features = ['Edad', 'Educación', 'Sexo', 'Fumador', 'Cigarrillos por Día', 
            'Toma Medicamentos Pre ART', 'Derrame Cerebral', 'Hipertenso', 
            'Diabetes', 'Colesterol Total', 'Presión Arterial Sistólica', 
            'Presión Arterial Diastólica', 'IMC', 'Frecuencia Cardíaca', 
            'Nivel de Glucosa']

# Normalizar las columnas relevantes en el dataset de entrenamiento y prueba
train_data[features] = min_max_normalize(train_data[features])
test_data[features] = min_max_normalize(test_data[features])

print(train_data.head())
print(test_data.head())

# Función para realizar la predicción utilizando el algoritmo KNN
def knn_predict(train_data, test_data, k):
    predictions = []
    
    for i in range(len(test_data)):
        distances = manhattan_distance(test_data[features].iloc[i].values, train_data[features].values)
        
        # Obtener los índices de los k vecinos más cercanos
        nearest_indices = np.argsort(distances)[:k]
        
        # Obtener las clases correspondientes a los vecinos más cercanos
        nearest_classes = train_data['Predicción'].iloc[nearest_indices]
        
        # Realizar la predicción basada en la mayoría de vecinos cercanos
        prediction = np.argmax(np.bincount(nearest_classes))
        predictions.append(prediction)
        
        print(f'Predicción {i + 1}: {predictions[i]}')
    
    return predictions

# Ejemplo de uso del algoritmo KNN con distancia Manhattan y k=5
k = 24
predictions = knn_predict(train_data, test_data, k)

# Imprimir las predicciones obtenidas
print(predictions)

# Calcular la precisión del modelo
correct_predictions = np.sum(test_data['Predicción'] == predictions)
total_predictions = len(test_data)

accuracy = correct_predictions / total_predictions
print(f'Precisión del modelo: {accuracy:.2%}')

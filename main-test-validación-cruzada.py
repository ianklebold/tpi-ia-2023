import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

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

# Función para normalizar los datos utilizando la técnica de min-max scaling
def min_max_normalize(data):
    min_values = data.min()
    max_values = data.max()
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data

# Seleccionar las características y la etiqueta de clase
features = ['Edad', 'Educación', 'Sexo', 'Fumador', 
            'Cigarrillos por Día', 'Toma Medicamentos Pre ART', 
            'Derrame Cerebral', 'Hipertenso', 
            'Diabetes', 'Colesterol Total', 
            'Presión Arterial Sistólica', 
            'Presión Arterial Diastólica', 
            'IMC', 
            'Frecuencia Cardíaca', 
            'Nivel de Glucosa']
X_train = train_data[features]
y_train = train_data['Predicción']

# Normalizar los datos de entrenamiento
X_train_normalized = min_max_normalize(X_train)

# Configurar los valores de k para probar
k_values = range(1, 31)

# Realizar validación cruzada para cada valor de k
scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    cv_scores = cross_val_score(knn, X_train_normalized, y_train, cv=10)
    scores.append(cv_scores.mean())

# Encontrar el valor de k con el mayor puntaje
best_k = k_values[np.argmax(scores)]
print(f'Mejor valor de k: {best_k}')

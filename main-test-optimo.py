import pandas as pd
import math
import random
import subprocess

# # Instala scikit-learn usando pip
# subprocess.run(["pip", "install", "sklearn"])

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Lee el dataset desde un archivo CSV
dataset = pd.read_csv('resources\dataset_cleaned_fixed.csv')

# Renombrar las columnas
dataset.rename(columns={
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

# Crea una clase para representar cada instancia
class Instancia:
    def __init__(self, id, atributos, clase):
        self.id = id
        self.atributos = atributos
        self.clase = clase

# Crea una lista para almacenar las instancias
instancias = []

# Itera sobre las filas del dataset
for index, row in dataset.iterrows():
    # Obtiene los atributos de la instancia y los convierte a tipo 'float'
    atributos = [float(value) for value in row[1:-1]]
    # Obtiene el ID de la instancia
    id_instancia = int(row['id'])
    # Obtiene la clase de la instancia
    clase = row[-1]
    # Crea una instancia y la agrega a la lista
    instancia = Instancia(id_instancia, atributos, clase)
    instancias.append(instancia)

# Función para calcular la distancia euclidiana entre dos instancias
def calcular_distancia(instancia1, instancia2):
    distancia = 0
    for i in range(len(instancia1.atributos)):
        distancia += (instancia1.atributos[i] - instancia2.atributos[i]) ** 2
    return math.sqrt(distancia)

# Función para encontrar los k vecinos más cercanos a una instancia nueva
def encontrar_vecinos_mas_cercanos(instancias, instancia_nueva, k):
    distancias = []
    for instancia in instancias:
        distancia = calcular_distancia(instancia, instancia_nueva)
        distancias.append((instancia, distancia))
    distancias.sort(key=lambda x: x[1])
    vecinos = [instancia[0] for instancia in distancias[:k]]
    return vecinos

# Función para encontrar el k óptimo usando validación cruzada
def encontrar_k_optimo(instancias, instancia_nueva, max_k=10):
    mejores_k = []
    mejores_scores = []
    for k in range(1, max_k + 1):
        vecinos_mas_cercanos = encontrar_vecinos_mas_cercanos(instancias, instancia_nueva, k)
        clases_vecinos = [vecino.clase for vecino in vecinos_mas_cercanos]
        # Crea un clasificador k-NN
        clasificador_knn = KNeighborsClassifier(n_neighbors=k)
        # Calcula el score usando validación cruzada
        score = np.mean(cross_val_score(clasificador_knn, [instancia_nueva.atributos], [instancia_nueva.clase], cv=5))
        mejores_k.append(k)
        mejores_scores.append(score)
    # Devuelve el k con el mejor score
    k_optimo = mejores_k[np.argmax(mejores_scores)]
    return k_optimo

# Mostrar detalles de los vecinos encontrados por instancia en cada K    
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

# Ejemplo de uso
instancias_random = random.sample(instancias, 6)

vecinos_por_k = []
coincidencias_por_k = []

for i, instancia_nueva in enumerate(instancias_random, start=1):
    resultados_instancia = []
    vecinos_k_actual = []
    coincidencias_k_actual = []

    for k in range(1, 7):
        vecinos_mas_cercanos = encontrar_vecinos_mas_cercanos(instancias, instancia_nueva, k)

        for vecino in vecinos_mas_cercanos:
            vecino.distancia = calcular_distancia(vecino, instancia_nueva)

        vecinos_mas_cercanos.sort(key=lambda x: x.distancia)

        mostrar_vecinos(vecinos_mas_cercanos, k)

        vecinos_k_actual.append(vecinos_mas_cercanos[-1].id)

        coincidencias_k_actual.append(sum(1 for vecino in vecinos_mas_cercanos if vecino.clase == instancia_nueva.clase))

        coincidencias = sum(1 for vecino in vecinos_mas_cercanos if vecino.clase == instancia_nueva.clase)
        resultados_instancia.append({'K': f'k={k}', 'Coincidencias': coincidencias})

    vecinos_por_k.append([f'k={i}'] + vecinos_k_actual)
    coincidencias_por_k.append([f'k={i}'] + coincidencias_k_actual)

    df_coincidencias_instancia = pd.DataFrame(resultados_instancia)
    print(f"\nResultados para la instancia i{i}:")
    print(df_coincidencias_instancia)

columnas = ['K'] + [f'i{i}' for i in range(1, 7)]
vecinos_por_k.append([''] * len(columnas))
coincidencias_por_k.append([''] * len(columnas))

df_vecinos_por_k = pd.DataFrame(vecinos_por_k, columns=columnas)
df_vecinos_por_k = df_vecinos_por_k.drop(df_vecinos_por_k.index[-1])

df_coincidencias_por_k = pd.DataFrame(coincidencias_por_k, columns=columnas)
df_coincidencias_por_k = df_coincidencias_por_k.drop(df_coincidencias_por_k.index[-1])

df_vecinos_por_k.iloc[:, 1:] = df_vecinos_por_k.iloc[:, 1:].values.T
df_coincidencias_por_k.iloc[:, 1:] = df_coincidencias_por_k.iloc[:, 1:].values.T

print("\nVecinos más cercanos por cada k para cada instancia:")
print(df_vecinos_por_k)

print("\nCoincidencias por cada k para cada instancia:")
print(df_coincidencias_por_k)

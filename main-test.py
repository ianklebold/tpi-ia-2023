import pandas as pd
import math

# Lee el dataset desde un archivo CSV
dataset = pd.read_csv('resources\dataset_cleaned_fixed.csv')

# Crea una clase para representar cada instancia
class Instancia:
    def __init__(self, atributos, clase):
        self.atributos = atributos
        self.clase = clase

# Crea una lista para almacenar las instancias
instancias = []

# Itera sobre las filas del dataset
for index, row in dataset.iterrows():
    # Obtiene los atributos de la instancia y los convierte a tipo 'float'
    atributos = [float(value) for value in row[1:-1]]
    # Obtiene la clase de la instancia
    clase = row[-1]
    # Crea una instancia y la agrega a la lista
    instancia = Instancia(atributos, clase)
    instancias.append(instancia)

# Función para calcular la distancia entre dos instancias
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

# Ejemplo de uso
# atributos_nueva_instancia = [50, 1, 1, 1, 15, 0, 0, 0, 0, 200, 120, 80, 23.0, 75, 85]
# 35,2,0,1,30,0,0,0,0,245,148.0,84.0,23.74,60,73
atributos_nueva_instancia = [35,2,0,1,30,0,0,0,0,245,148.0,84.0,23.74,60,73]
instancia_nueva = Instancia(atributos_nueva_instancia, None)
k = 5

# Encuentra los vecinos más cercanos
vecinos_mas_cercanos = encontrar_vecinos_mas_cercanos(instancias, instancia_nueva, k)

# Calcula la distancia para cada vecino
for vecino in vecinos_mas_cercanos:
    vecino.distancia = calcular_distancia(vecino, instancia_nueva)

# Ordena los vecinos por distancia
vecinos_mas_cercanos.sort(key=lambda x: x.distancia)

# Crea un DataFrame para mostrar los resultados
df_resultados = pd.DataFrame({
    'Age': [vecino.atributos[0] for vecino in vecinos_mas_cercanos],
    'Education': [vecino.atributos[1] for vecino in vecinos_mas_cercanos],
    'Sex': [vecino.atributos[2] for vecino in vecinos_mas_cercanos],
    'Is Smoking': [vecino.atributos[3] for vecino in vecinos_mas_cercanos],
    'Cigs/Day': [vecino.atributos[4] for vecino in vecinos_mas_cercanos],
    'BPMeds': [vecino.atributos[5] for vecino in vecinos_mas_cercanos],
    'Prev. Stroke': [vecino.atributos[6] for vecino in vecinos_mas_cercanos],
    'Prev. Hyp': [vecino.atributos[7] for vecino in vecinos_mas_cercanos],
    'Diabetes': [vecino.atributos[8] for vecino in vecinos_mas_cercanos],
    'Tot Chol': [vecino.atributos[9] for vecino in vecinos_mas_cercanos],
    'Sys BP': [vecino.atributos[10] for vecino in vecinos_mas_cercanos],
    'Dia BP': [vecino.atributos[11] for vecino in vecinos_mas_cercanos],
    'BMI': [vecino.atributos[12] for vecino in vecinos_mas_cercanos],
    'Heart Rate': [vecino.atributos[13] for vecino in vecinos_mas_cercanos],
    'Distance': [vecino.distancia for vecino in vecinos_mas_cercanos]
})

# Imprime el DataFrame con los resultados
print(df_resultados)

import pandas as pd
import math

# Lee el dataset desde un archivo CSV
dataset = pd.read_csv('resources\dataset_cleaned_fixed.csv')

# Renombrar las columnas
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
#51,1,1,1,15,0,0,0,0,212,146.0,89.0,24.49,100,132
atributos_nueva_instancia = [55,1,0,0,0,0,0,1,0,243,142.0,92.0,30.24,70,85]
instancia_nueva = Instancia(-1, atributos_nueva_instancia, 0.0)  # Usamos -1 para indicar que no tiene un ID asignado aún, Clase Predicción = 1.0 o 0.0

# Lista para almacenar los resultados
resultados = []

# Iterar para k de 1 a 5
for k in range(1, 6):
    # Encuentra los vecinos más cercanos
    vecinos_mas_cercanos = encontrar_vecinos_mas_cercanos(instancias, instancia_nueva, k)

    # Calcula la distancia para cada vecino
    for vecino in vecinos_mas_cercanos:
        vecino.distancia = calcular_distancia(vecino, instancia_nueva)

    # Ordena los vecinos por distancia
    vecinos_mas_cercanos.sort(key=lambda x: x.distancia)
    
    # Crea un DataFrame para mostrar los resultados
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

    # Cuenta cuántas predicciones coinciden con el valor de la clase
    coincidencias = sum(1 for vecino in vecinos_mas_cercanos if vecino.clase == instancia_nueva.clase)

    # Almacena los resultados
    resultados.append({'K': k, 'Coincidencias': coincidencias})

# Crea un DataFrame para mostrar los resultados de coincidencias con la clase
df_coincidencias = pd.DataFrame(resultados)

# Imprime el DataFrame con los resultados de coincidencias
# print con estrellas ♫
print("===================================\n")
print("Resultados de coincidencias con la clase para cada k:")
print(df_coincidencias)
print("===================================\n")


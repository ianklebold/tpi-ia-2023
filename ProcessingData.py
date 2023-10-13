import pandas as pd

# Lee el dataset desde un archivo CSV
dataset = pd.read_csv('resources/dataset_cleaned_fixed.csv')

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


# Llevarlo a un package aparte
class Instancia:
    def __init__(self, id, atributos, clase):
        self.id = id
        self.atributos = atributos
        self.clase = clase


# Crea una lista para almacenar las instancias
def getinstances():
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
    return instancias

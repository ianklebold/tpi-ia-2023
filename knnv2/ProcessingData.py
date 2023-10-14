import pandas as pd


class Dataset:
    def __init__(self, data=None):
        if data is None:
            data = list()
        self.data = data


# Llevarlo a un package aparte
class Instancia:
    def __init__(self, id, atributos, clase):
        self.id = id
        self.atributos = atributos
        self.clase = clase


def getDataset():
    # Lee el dataset desde un archivo CSV
    dataset = pd.read_csv('../resources/dataset_cleaned_fixed.csv')
    newDataset = Dataset(dataset)
    newDataset.data = dataset

    # Renombrar las columnas
    newDataset.data.rename(columns={
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

    return newDataset


# Crea una lista para almacenar las instancias
def getInstances():
    instances = list()
    dataset = getDataset()
    # Itera sobre las filas del dataset
    for index, row in dataset.data.iterrows():
        # Obtiene los atributos de la instancia y los convierte a tipo 'float'
        attributes = [float(value) for value in row[1:-1]]
        # Obtiene el ID de la instancia
        id_instance = int(row['id'])
        # Obtiene la clase de la instancia
        instance_class = row[-1]
        # Crea una instancia y la agrega a la lista
        instances.append(Instancia(id_instance, attributes, instance_class))
    return instances

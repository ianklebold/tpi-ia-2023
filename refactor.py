import pandas as pd
import math
import random


# -------
# dataset section
class DataTreatment:
  dataset = pd.read_csv('resources/dataset_cleaned_fixed.csv')

  # column rename
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

  def get_ds_rows(self, dataset):
    # Crea una lista para almacenar las instances
    instances = []

    # Iterate over dataset rows
    for index, row in dataset.iterrows():
      # Obtiene los atributos de la instancia y los convierte a tipo 'float'
      attribute = [float(value) for value in row[1:-1]]
      # Obtiene el ID de la instancia
      id_instance = int(row['id'])
      # Obtiene la clase de la instancia
      _class = row[-1]
      # Crea una instancia y la agrega a la lista
      instance = Instance(id_instance, attribute, _class)
      instances.append(instance)

  # Mostrar detalles de los vecinos encontrados por instancia en cada K
  def list_neighbors(self, knn, k):
    df_resultados = pd.DataFrame({
      'ID': [vecino.id for vecino in knn],
      'Edad': [vecino.atributos[0] for vecino in knn],
      'Educación': [vecino.atributos[1] for vecino in knn],
      'Sexo': [vecino.atributos[2] for vecino in knn],
      'Fumador': [vecino.atributos[3] for vecino in knn],
      'Cigarrillos por Día': [vecino.atributos[4] for vecino in knn],
      'Toma Medicamentos Pre ART': [vecino.atributos[5] for vecino in knn],
      'Derrame Cerebral': [vecino.atributos[6] for vecino in knn],
      'Hipertenso': [vecino.atributos[7] for vecino in knn],
      'Diabetes': [vecino.atributos[8] for vecino in knn],
      'Colesterol Total': [vecino.atributos[9] for vecino in knn],
      'Presión Arterial Sistólica': [vecino.atributos[10] for vecino in knn],
      'Presión Arterial Diastólica': [vecino.atributos[11] for vecino in knn],
      'IMC': [vecino.atributos[12] for vecino in knn],
      'Frecuencia Cardíaca': [vecino.atributos[13] for vecino in knn],
      'Nivel de Glucosa': [vecino.atributos[14] for vecino in knn],
      'Predicción': [vecino.clase for vecino in knn],
      'Distancia': [vecino.distancia for vecino in knn]
    })

    # Imprime el DataFrame con los resultados
    print("------------------------------------\n")
    print("Resultados para k = ", k)
    print(df_resultados)
    print("------------------------------------\n")


# class to represent each instance
class Instance:
  def __init__(self, id, attribute, _class):
    self.id = id
    self.attribute = attribute
    self._class = _class


# class KNN:



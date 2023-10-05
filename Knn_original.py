# Authors:
#   Kalbermatter, Alan
#   Fernandez, Ian
#   Hadad Menegaz, Jorge Miguel

import pandas as pd
import math
from tratamiento_datos.TratamientoDatos import limpiar_datos

# Función para calcular la distancia entre dos instancias
def calcular_distancia(instancia1, instancia2):
    distancia = 0
    for i in range(len(instancia1.atributos)):
        distancia += (instancia1.atributos[i] - instancia2.atributos[i]) ** 2
    return math.sqrt(distancia)


# Lee el dataset desde un archivo CSV ORIGINAL - SIN TRATAR
dataset = pd.read_csv('resources\data_cardiovascular_risk.csv')

dataset = limpiar_datos(dataset)


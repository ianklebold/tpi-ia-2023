# Authors:
#   Kalbermatter, Alan
#   Fernandez, Ian
#   Hadad Menegaz, Jorge Miguel

from domain.Dataset import Dataset
from domain.Conjunto import Conjunto
from domain.Clase import Clase
from domain.Instancia import Instancia

import math
import pandas as pd

def calcular_distancia(instancia1, instancia2):
    distancia = 0

    for i in range(len(instancia1.atributos)):
        distancia += (instancia1.atributos[i] - instancia2.atributos[i]) ** 2

    return math.sqrt(distancia)

def d(clase_analizada, funcion_clase):
    if funcion_clase == clase_analizada:
        return 1
    else:
        return 0


def knn(dataset, instancia_nueva, k):
    distancias = []

    for conjunto in dataset.conjuntos:
        for clase in conjunto.clases:
            for instancia in clase.instancias:
                distancia = calcular_distancia(instancia, instancia_nueva)
                distancias.append((distancia, instancia.clase))

    distancias.sort(key=lambda x: x[0])
    vecinos = distancias[:k]

    clases_vecinos = [vecino[1] for vecino in vecinos]
    clase_ganadora = max(set(clases_vecinos), key=clases_vecinos.count)

    return clase_ganadora

if __name__ == '__main__':

    df = pd.read_csv('resources\dataset_cleaned.csv')
    # print(df.head(2))
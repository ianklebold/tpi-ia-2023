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
    # # Crear un conjunto
    # conjunto1 = Conjunto("Conjunto 1")
    
    # instancia1 = Instancia([1, 2, 3])

    # # Crear una clase
    # clase1 = Clase("Clase 1")

    # # Crear instancias
    # instancia1 = Instancia([1, 2, 3])
    # instancia2 = Instancia([4, 5, 6])

    # # Agregar instancias a la clase
    # clase1.agregar_instancia(instancia1)
    # clase1.agregar_instancia(instancia2)

    # # Agregar la clase al conjunto
    # conjunto1.agregar_clase(clase1)

    # # Agregar el conjunto al dataset
    # # Crear el dataset
    # dataset = Dataset()

    # # Agregar el conjunto al dataset
    # dataset.agregar_conjunto(conjunto1)
    
    # # Imprimir el dataset completo
    # for conjunto in dataset.conjuntos:
    #     print("Conjunto: ", conjunto.nombre)
    #     for clase in conjunto.clases:
    #         print("Clase: ", clase.nombre)
    #         for instancia in clase.instancias:
    #             print("Instancia: ", instancia.atributos)

    def d(clase_analizada, funcion_clase):
        if funcion_clase == clase_analizada:
            return 1
        else:
            return 0

    # Suponiendo que tienes una lista ordenada de instancias y sus clases
    lista_instancias = [('x1', 'c1'), ('x2', 'c2'), ('x3', 'c1')]

    # Suponiendo que K = 3
    k = 3

    # Realizar la sumatoria para cada clase
    print("   x    |    c1    |    c2    ")
    print("-------------------------------")
    print("   x1   |    1     |    0     ")
    print("   x2   |    0     |    1     ")
    print("   x3   |    1     |    0     ")
    print("-------------------------------")
    print("Suma c1 |    2     |    0     ")
    print("Suma c2 |    0     |    1     ")
    print('lo que me devuelve for instancia in lista_instancias[:k] es: ')
    print([instancia for instancia in lista_instancias[:k]])
    
    suma_c1 = sum([d("c1", instancia[1]) for instancia in lista_instancias[:k]])
    suma_c2 = sum([d("c2", instancia[1]) for instancia in lista_instancias[:k]])

    # Imprimir los resultados
    print("Suma para clase c1:", suma_c1)
    print("Suma para clase c2:", suma_c2)

    # Determinar la clase ganadora
    if suma_c1 > suma_c2:
        clase_ganadora = "c1"
    else:
        clase_ganadora = "c2"

    print("La clase ganadora es:", clase_ganadora)


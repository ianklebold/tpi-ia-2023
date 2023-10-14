import ProcessingData as processingData
import math
import random

def calcular_distancia(instancia, instancia_nueva):
    distancia = 0
    for i in range(len(instancia.atributos)):
        distancia += (instancia.atributos[i] - instancia_nueva.atributos[i]) ** 2
    return math.sqrt(distancia)

def encontrar_vecinos_mas_cercanos(instancias, k):
    # Esto lo debe ingresar el usuario
    instancia_nueva = processingData.getinstances().pop(random.randint(0, len(processingData.getinstances())))

    training_examples = []
    for instancia in instancias:
        distancia = calcular_distancia(instancia, instancia_nueva)
        training_examples.append((instancia, distancia))
    training_examples.sort(key=lambda x: x[1])
    vecinos_mas_cercanos = [instancia[0] for instancia in training_examples[:k]]
    return vecinos_mas_cercanos


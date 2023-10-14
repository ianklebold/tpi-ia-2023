import ProcessingData as processingData
import math
import random


def calculateDistance(instance, new_instance):
    distance = 0
    for i in range(len(instance.atributos)):
        distance += (instance.atributos[i] - new_instance.atributos[i]) ** 2
    return math.sqrt(distance)


def findNeighborMoreEarn(k):
    # Esto lo debe ingresar el usuario
    new_instance = processingData.getInstances().pop(random.randint(0, len(processingData.getInstances())))

    training_examples = []
    for instance in processingData.getInstances():
        distance = calculateDistance(instance, new_instance)
        training_examples.append((instance, distance))
    training_examples.sort(key=lambda x: x[1])
    neighbor_more_earns = [instance[0] for instance in training_examples[:k]]
    return neighbor_more_earns

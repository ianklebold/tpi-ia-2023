import math
import random

from knnv2 import ProcessingData


def calculateDistance(instance, new_instance):
    distance = 0
    for i in range(len(instance.atributos)):
        distance += (instance.atributos[i] - new_instance.atributos[i]) ** 2
    return math.sqrt(distance)


def findNeighborMoreEarn(k):
    # Esto lo debe ingresar el usuario
    instances = ProcessingData.getInstances()
    new_instance = instances.pop(random.randint(0, len(instances)))

    training_examples = []
    for instance in instances:
        distance = calculateDistance(instance, new_instance)
        training_examples.append((instance, distance))
    training_examples.sort(key=lambda x: x[1])
    neighbor_more_earns = [instance[0] for instance in training_examples[:k]]
    return neighbor_more_earns

def calculateDistanceManhattan(instance, new_instance):
    distance = 0
    for i in range(len(instance.atributos)):
      distance += abs(instance.atributos[i] - new_instance.atributos[i])
    return distance


def calculateChebyshevDistance(instance, new_instance):
  distances = []
  for i in range(len(instance.atributos)):
    distances.append(abs(instance.atributos[i] - new_instance.atributos[i]))
  return max(distances)



import math
import random
from knnv2 import ProcessingData
from enum import Enum

class DistanceMethod(Enum):
    EUCLIDEAN = 1
    MANHATTAN = 2
    CHEBYSHEV = 3
    MINKOWSKI = 4


def calculateDistance(instance, new_instance):
    distance = 0
    for i in range(len(instance.atributos)):
        distance += (instance.atributos[i] - new_instance.atributos[i]) ** 2
    return math.sqrt(distance)


def findNeighborMoreEarn(k, method):
    instances = ProcessingData.getInstances()
    new_instance = instances.pop(random.randint(0, len(instances)))
    distance_functions = {
      DistanceMethod.EUCLIDEAN: calculateDistance,
      DistanceMethod.MANHATTAN: calculateDistanceManhattan,
      DistanceMethod.CHEBYSHEV: calculateChebyshevDistance,
      DistanceMethod.MINKOWSKI: calculateMinkowskiDistance(new_instance, instances)
    }

    if method in distance_functions:
        distance_function = distance_functions[method]
    else:
        raise ValueError("Método de cálculo de distancia no válido")

    training_examples = []
    for instance in instances:
        distance = distance_function(instance, new_instance)
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

def calculateMinkowskiDistance(x1, x2):
    p = 0.0001
    distance = 0
    for i in range(len(x1)):
        distance += abs(x1[i] - x2[i]) ** p
    return distance ** (1/p)

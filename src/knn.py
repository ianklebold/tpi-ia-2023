import math


def calculate_distance(instance, new_instance):
    distance = 0
    for i in range(len(instance.attribute)):
        distance += (instance.attributes[i] - new_instance.atributos[i]) ** 2

    return math.sqrt(distance)


def training_algorithm(instances, new_instance, k):
    distances = []
    for instance in instances:
        distance = calculate_distance(instance, new_instance)
        distances.append((instance, distance))
    # distancias.sort(key=lambda x: x[1])
    # vecinos = [instancia[0] for instancia in distancias[:k]]
    return distances


def sort_distances(distances):
    distances.sort(key=lambda x: x[1])
    return distances


def knn(self, instances, new_instance, k):
    distances = training_algorithm(self, instances, new_instance)
    sorted_distances = sort_distances(distances)
    neighbors = [instance[0] for instance in sorted_distances[:k]]
    return neighbors
